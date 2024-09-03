from tinygrad.helpers import fetch

from transformers import AutoTokenizer
from typing import Tuple, Union, Optional, Dict, Any
from tinygrad import Tensor, Variable, nn, dtypes, TinyJit
from tinygrad.helpers import getenv
import math, sys
from examples.mamba import MambaMixer
from tinygrad.nn.state import load_state_dict
from tqdm import tqdm
from tinygrad.helpers import Context

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype=dtypes.bfloat16) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  # freqs = Tensor.arange(end, dtype=dtypes.int64).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  freqs = Tensor.einsum("i,j->ij", Tensor.arange(end, dtype=dtypes.int64), freqs)
  # TODO: move dtype outside this
  return Tensor.stack(freqs.cos().cast(dtype), freqs.sin().cast(dtype), dim=-1).reshape(1, end, 1, dim//2, 2).realize()

# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, c, d):
  a,b = A[..., 0:1], A[..., 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return ro.cat(co, dim=-1)

def apply_rotary_emb_new(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> Tuple[Tensor, Tensor]:
  assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 5
  c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
  xq_out = complex_mult(xq, c, d)
  xk_out = complex_mult(xk, c, d)
  return xq_out.flatten(3), xk_out.flatten(3)

class Attention:
  def __init__(self, dim, n_heads, n_kv_heads, max_context, window_size, linear=nn.Linear):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = dim // n_heads
    # self.n_rep = self.n_heads // self.n_kv_heads #Should be 1, don't see multi grouped head
    self.max_context = max_context
    self.window_size = window_size

    self.attn = linear(dim, 3*dim, bias=False) 
    self.proj = linear(dim, dim, bias=False)

  def __call__(self, x:Tensor, start_pos:Union[Variable,int], mask:Optional[Tensor]=None) -> Tensor:
    B,T,C = x.shape
    qkv = self.attn(x)
    qkv = qkv.view(B, T, self.n_heads, 3, self.head_dim) # (B, T, n_query_groups, total_qkv, hs)
    q, k, v = qkv.split((1, 1, 1), dim=-2)
    q = q.reshape(B,  T, -1, self.head_dim)# (B, T, nh_q, hs)
    k = k.reshape(B,  T, -1, self.head_dim)  
    v = v.reshape(B,  T, -1, self.head_dim)
    keys, values = k, v

    if getenv('ROPE'):
      if not hasattr(self, "freq_cis"): self.freq_cis = precompute_freqs_cis(self.head_dim, self.max_context).cast(dtypes.float32).contiguous().realize()
      freq_cis = self.freq_cis.shrink((None, (start_pos, start_pos+T),None,None,None))
      q, keys = apply_rotary_emb_new(q, keys, freq_cis)

    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, B, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype).contiguous().realize()
      if isinstance(x.device, tuple):
        # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded
        self.cache_kv.shard_((x.device), axis=3 if getenv("SHARD_KVCACHE") else None).realize()

    # update the cache
    assert k.dtype == v.dtype == self.cache_kv.dtype, f"{k.dtype=}, {v.dtype=}, {self.cache_kv.dtype=}"
    self.cache_kv.shrink((None, None, (start_pos, start_pos+T), None, None)).assign(Tensor.stack(keys, values)).realize()

    keys = self.cache_kv[0].shrink((None, (0, start_pos+T), None, None)) if start_pos > 0 else keys
    values = self.cache_kv[1].shrink((None, (0, start_pos+T), None, None)) if start_pos > 0 else values
    q, keys, values = q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    y = q.scaled_dot_product_attention(keys, values, is_causal=T>1, attn_mask=mask).transpose(1, 2)
    y = y.reshape(B, T, -1)
    return self.proj(y)

class SwiGLU:
  def __init__(self, dim):
    self.w1 = nn.Linear(dim, 4*dim, bias=False)
    self.w2 = nn.Linear(dim, 4*dim, bias=False)
    self.w3 = nn.Linear(4*dim, dim, bias=False)
  def __call__(self, x:Tensor): return self.w3(self.w1(x).silu() * self.w2(x))

class SambaLayer:
  def __init__(self, dim, layer_idx, n_heads, max_context, window_size):
    self.layer_idx = layer_idx
    self.norm_1 = nn.RMSNorm(dim, 1e-5)
    if layer_idx%2==0: self.attn = MambaMixer(dim)
    else: self.attn = Attention(dim, n_heads, n_heads, max_context, window_size)
    self.norm_2 = nn.RMSNorm(dim,1e-5)
    self.swiglu = SwiGLU(dim)
  def __call__(self, x:Tensor, start_pos=0):
    n_1 = self.norm_1(x)
    if self.layer_idx%2==0: h = self.attn(n_1)
    else: h = self.attn(n_1, start_pos)
    x = x + h
    n_2 = self.norm_2(x)
    h = self.swiglu(n_2)
    x = x + h
    return x

class Samba:
  def __init__(self, dim, n_heads, n_layers, vocab_size, max_context, window_size):
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)
    self.wte = nn.Embedding(vocab_size, dim)
    self.layers = [SambaLayer(dim, i, n_heads, max_context, window_size) for i in range(n_layers)]
    self.ln_f = nn.RMSNorm(dim, 1e-5)
    self.forward_jit = TinyJit(self.__call__)
  def __call__(self, x:Tensor, start_pos=0):
    out = self.wte(x)
    for l in self.layers:
      out = l(out, start_pos)
    out = self.ln_f(out)
    return self.lm_head(out)

  def load_model(self):
    import torch
    # path = fetch('https://ml-modelstore-public.s3.ap-northeast-2.amazonaws.com/iter-1003200-ckpt.pth')
    path = fetch('https://ml-modelstore-public.s3.ap-northeast-2.amazonaws.com/samba_instruct.pth')
    print('path: ', path)
    d = torch.load(path, weights_only=True)
    print(d.keys())
    print(len(d['model'].keys()))
    tens_weigts = {}
    for k,v in d['model'].items():
      # print(k, v.shape)
      if 'transformer.wte' in k:
        tens_weigts[k.replace('transformer.wte', 'wte')] = Tensor(v.cpu().numpy())
      elif 'transformer.h' in k:
        tens_weigts[k.replace('transformer.h', 'layers').replace('mlp.swiglu', 'swiglu')] = Tensor(v.cpu().numpy())
      elif 'transformer.ln_f' in k:
        tens_weigts[k.replace('transformer.ln_f', 'ln_f')] = Tensor(v.cpu().numpy())
      else:
        tens_weigts[k] = Tensor(v.cpu().numpy())
    load_state_dict(self, tens_weigts)

  def load_tokenizer(self):
    tokenizer_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer
def llama3_prompt_format(text):
    prompt = f"""<|start_header_id|>user<|end_header_id|>
{text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    return prompt
if __name__ == '__main__':
  samba = Samba(1024, 8, 8, 128256, 4096, 2048)

  samba.load_model()
  tokenizer = samba.load_tokenizer()
  # REF_NOPE
  # <|begin_of_text|>1,2,3,4,5,6,7,8,8,9,10,11,12,13,14,15,16,17,17,17,19,19,21,21,21,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,25,25,26,26,26,26
  prompt = "1,2,3,4,5,6,"
  prompt = llama3_prompt_format('Why is the sky blue?')
  output = prompt
  prompt_tok = tokenizer.encode(prompt)
  sp = 0
  for i in (bar:=tqdm(range(3000))):
    tens = Tensor([prompt_tok[sp:]])
    if i != 0 :
      # with Context(BEAM=2):
      out = samba.forward_jit(tens, Variable("start_pos", 0, 4096).bind(sp))
    else: out = samba(tens, sp)
    logits = out[:, -1, :]
    tok = logits.softmax(-1).argmax()
    tok_str = tokenizer.decode(tok_item := tok.item())
    # print('GEN_TOK: ', tok_str, '||', tok.item())
    output+=tok_str
    bar.set_description(f'GEN_TOK: {tok_str}||')

    sp = len(prompt_tok)
    prompt_tok.append(tok.item())
    if tok_item==128001: break
  print(output)