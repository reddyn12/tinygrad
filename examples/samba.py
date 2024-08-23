#%%
from tinygrad.helpers import fetch
from tinygrad.nn.state import torch_load
import pickle
import torch
path = fetch('https://ml-modelstore-public.s3.ap-northeast-2.amazonaws.com/iter-1003200-ckpt.pth', 'samba_weights.pth', '/raid/weights/')
print('path: ', path)
d = torch.load(path)
print(d.keys())
print(len(d['model'].keys()))

#%%
from transformers import AutoTokenizer

tokenizer_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
prompt = 'This is a test prompt that '
inputs = tokenizer.encode(prompt)
print(inputs)

# %%
for k,v in d['model'].items():
  print(k, v.shape, v.dtype)
# %%
d['optimizer'].keys()
# %%
from typing import Tuple, Union, Optional, Dict, Any
from tinygrad import Tensor, Variable, nn, dtypes
from tinygrad.helpers import getenv
import math
def repeat_kv(x:Tensor, n_rep:int) -> Tensor:
  bs, seqlen, n_kv_heads, head_dim = x.shape
  if n_rep == 1: return x
  # NOTE: this is different from x.repeat((1, 1, n_rep, 1))
  return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype=dtypes.bfloat16) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2, dtype=dtype)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end, dtype=dtype).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
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

def _rotate_half(x: Tensor) -> Tensor:
  x1, x2 = x.chunk(2, dim=-1)
  return Tensor.cat(-x2, x1, dim=-1)

def _apply_rotary_pos_emb(x: Tensor, pos_sin: Tensor, pos_cos: Tensor) -> Tensor:
  a1 =  (x * pos_cos)
  a2 = (_rotate_half(x) * pos_sin)
  return a1+a2
def sliding_window_attention(query:Tensor, key:Tensor, value:Tensor, window_size ,attn_mask:Optional[Tensor]=None):
  """
  Computes sliding window attention.
  """
  bsz, n_heads, seqlen, head_dim = query.shape

  # compute attention weights for each window
  attn_weights = []
  for i in range(0, seqlen, window_size):
    window_query = query[:, :, i:i+window_size, :]
    window_key = key[:, :, i:i+window_size, :]
    window_value = value[:, :, i:i+window_size, :]

    window_attn = window_query.matmul(window_key.transpose(-2,-1)) / math.sqrt(head_dim)
    if attn_mask is not None:
      window_attn += attn_mask[:, :, i:i+window_size, i:i+window_size]

    window_attn = window_attn.softmax(-1)
    window_attn = window_attn.matmul(window_value)

    attn_weights.append(window_attn)

  # concatenate attention weights from each window
  attn = Tensor.cat(*attn_weights, dim=2)

  return attn
class Attention:
  def __init__(self, dim, n_heads, n_kv_heads, max_context, window_size, linear=nn.Linear):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads
    # self.n_rep = self.n_heads // self.n_kv_heads
    self.max_context = max_context
    self.window_size = window_size

    self.attn = linear(dim, 3*dim, bias=False) 
    self.proj = linear(dim, dim, bias=False)
    # self.mask = Tensor.ones((4096, 4096), dtype=dtypes.bool).tril().unsqueeze(0).unsqueeze(0)
    # self.freq_cis = precompute_freqs_cis(self.head_dim, window_size*2, 10000)

  def __call__(self, x:Tensor, start_pos:Union[Variable,int], mask:Optional[Tensor]=None) -> Tensor:
    B,T,C = x.shape
    qkv = self.attn(x)
    qkv = qkv.view(B, T, 8, 3, self.head_dim) # (B, T, n_query_groups, total_qkv, hs)
    q, k, v = qkv.split((1, 1, 1), dim=-2)
    q = q.reshape(B,  T, -1, self.head_dim)#.transpose(1, 2)  # (B, T, nh_q, hs)
    k = k.reshape(B,  T, -1, self.head_dim)#.transpose(1, 2)  
    v = v.reshape(B,  T, -1, self.head_dim)#.transpose(1, 2)
    keys, values = k, v
    # freq_cis = precompute_freqs_cis(128, 4096, dtype=dtypes.float32)[:, start_pos:start_pos+T]
    # q, keys = apply_rotary_emb_new(q, keys, freq_cis)

    # if not hasattr(self, "cache_kv"):
    #   self.cache_kv = Tensor.zeros(2, B, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype).contiguous().realize()
    #   if isinstance(x.device, tuple):
    #     # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded
    #     self.cache_kv.shard_((x.device), axis=3 if getenv("SHARD_KVCACHE") else None).realize()

    # # update the cache
    # assert k.dtype == v.dtype == self.cache_kv.dtype, f"{k.dtype=}, {v.dtype=}, {self.cache_kv.dtype=}"
    # self.cache_kv.shrink((None, None, (start_pos, start_pos+T), None, None)).assign(Tensor.stack(k, v)).realize()

    # keys = self.cache_kv[0].shrink((None, (0, start_pos+T), None, None)) if start_pos > 0 else k
    # values = self.cache_kv[1].shrink((None, (0, start_pos+T), None, None)) if start_pos > 0 else v
    q, keys, values = q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    y = q.scaled_dot_product_attention(keys, values, is_causal=True, attn_mask=mask).transpose(1, 2)
    y = y.reshape(B, T, -1)
    return self.proj(y)
    # bsz, seq_len, embed_dim = x.shape
    # # print('ATTENTION_SHAPE', x.shape)
    # xqkv = self.attn(x)
    # xq, xk, xv = xqkv.split([self.attn.weight.shape[0]//3, self.attn.weight.shape[0]//3, self.attn.weight.shape[0]//3], dim=2)


    # xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    # xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    # xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

    # # xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    # freq_constant = 10000
    # head_dim=128
    # seqlen=4096
    # inv_freq = 1.0 / (freq_constant ** (Tensor.arange(0, head_dim, 2) / head_dim))
    # # print(inv_freq.shape)
    # pos_index_theta = Tensor.einsum("i,j->ij", Tensor.arange(seq_len), inv_freq)
    # emb = Tensor.cat(pos_index_theta, pos_index_theta, dim=-1)
    # cos_emb, sin_emb = emb.cos()[None,:,  None, :], emb.sin()[None,:,  None, :]
    # # cos_emb, sin_emb = emb.cos()[:, None, None, :], emb.sin()[:, None, None, :]
    # # freq_constant = 10000
    # # inv_freq = 1.0 / (freq_constant ** (Tensor.arange(0, self.head_dim, 2) / self.head_dim))
    # # pos_index_theta = Tensor.einsum("i,j->ij", Tensor.arange(seq_len), inv_freq)
    # # emb = Tensor.cat(pos_index_theta, pos_index_theta, dim=-1)
    # # cos_emb, sin_emb = emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
    # # print('ROPE:', emb.shape, cos_emb.shape, sin_emb.shape)
    # # xq = _apply_rotary_pos_emb(xq, sin_emb, cos_emb)
    # # xk = _apply_rotary_pos_emb(xk, sin_emb, cos_emb)
    # freq_cis = precompute_freqs_cis(head_dim, seqlen)[:,:seq_len,:,:,:]
    # print('freq_cis', freq_cis.shape)
    # xq, xk = apply_rotary_emb_new(xq, xk, freq_cis)
    # keys = xk
    # values = xv

    # # # create kv cache
    # # if not hasattr(self, "cache_kv"):
    # #   self.cache_kv = Tensor.zeros(2, bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype).contiguous().realize()
    # #   if isinstance(x.device, tuple):
    # #     # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded
    # #     self.cache_kv.shard_((x.device), axis=3 if getenv("SHARD_KVCACHE") else None).realize()

    # # # update the cache
    # # assert xk.dtype == xv.dtype == self.cache_kv.dtype, f"{xk.dtype=}, {xv.dtype=}, {self.cache_kv.dtype=}"
    # # self.cache_kv.shrink((None, None, (start_pos, start_pos+seq_len), None, None)).assign(Tensor.stack(xk, xv)).realize()

    # # keys = self.cache_kv[0].shrink((None, (0, start_pos+seq_len), None, None)) if start_pos > 0 else xk
    # # values = self.cache_kv[1].shrink((None, (0, start_pos+seq_len), None, None)) if start_pos > 0 else xv

    # # keys, values = repeat_kv(keys, 1), repeat_kv(values, 1)
    # xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    # # attn = xq.scaled_dot_product_attention(keys, values, is_causal=True).transpose(1, 2)
    # # mask_new = Tensor.ones((4096, 4096), dtype=dtypes.bool).tril().unsqueeze(0).unsqueeze(0)
    # attn = sliding_window_attention(xq, keys, values, self.window_size, mask)
    # attn = attn.reshape(bsz, seq_len, -1)
    # return self.proj(attn)
# %%
class SwiGLU:
  def __init__(self, dim):
    self.w1 = nn.Linear(dim, 4*dim, bias=False)
    self.w2 = nn.Linear(dim, 4*dim, bias=False)
    self.w3 = nn.Linear(4*dim, dim, bias=False)
  def __call__(self, x:Tensor):
    return self.w3(self.w1(x).silu() * self.w2(x))
# %%
from examples.mamba import MambaMixer
from tinygrad.nn.state import load_state_dict
class SambaLayer:
  def __init__(self, dim, layer_idx):
    self.layer_idx = layer_idx
    self.norm_1 = nn.RMSNorm(dim, 1e-5)
    if layer_idx%2==0:
      self.attn = MambaMixer(dim)
    else:
      # mask = 
      self.attn = Attention(dim, 8, 8, 4096, 2048)
    self.norm_2 = nn.RMSNorm(dim,1e-5)
    self.swiglu = SwiGLU(dim)
  def __call__(self, x:Tensor, start_pos=0):
    # print('LAYER_IDX', self.layer_idx, x.shape)
    n_1 = self.norm_1(x)
    if self.layer_idx%2==0:
      h = self.attn(n_1)
      delattr(self.attn, 'conv_state')
    else:
      seqlen = x.shape[1]
      # mask = Tensor.full((1, 1, seqlen, start_pos+seqlen), float("-inf")).triu(start_pos+1).realize() if seqlen > 1 else None
      mask=None
      h = self.attn(n_1, start_pos, mask)
    
    x = x + h
    n_2 = self.norm_2(x)
    h = self.swiglu(n_2)
    x = x + h
    return x
    
class Samba:
  def __init__(self, dim, n_heads, n_layers, vocab_size):
    self.lm_head = nn.Linear(dim, vocab_size, bias=False)
    self.wte = nn.Embedding(vocab_size, dim)
    self.layers = [SambaLayer(dim, i) for i in range(n_layers)]
    self.ln_f = nn.RMSNorm(dim, 1e-5)
  def __call__(self, x:Tensor, start_pos=0):
    out = self.wte(x)
    # out = out.sequential(self.layers)
    for l in self.layers:
      out = l(out, start_pos)
    out = self.ln_f(out)
    return self.lm_head(out)
t = Samba(1024, 8, 8, 128256)
print(len(nn.state.get_state_dict(t).keys()))
# for k,v in nn.state.get_state_dict(t).items():
#   print(k, v.shape)
tens_weigts = {}
for k,v in d['model'].items():
  print(k, v.shape)
  if 'transformer.wte' in k:
    tens_weigts[k.replace('transformer.wte', 'wte')] = Tensor(v.cpu().numpy())
  elif 'transformer.h' in k:
    tens_weigts[k.replace('transformer.h', 'layers').replace('mlp.swiglu', 'swiglu')] = Tensor(v.cpu().numpy())
  elif 'transformer.ln_f' in k:
    tens_weigts[k.replace('transformer.ln_f', 'ln_f')] = Tensor(v.cpu().numpy())
  else:
    tens_weigts[k] = Tensor(v.cpu().numpy())
for k,v in tens_weigts.items():
  print(k, v.shape)
load_state_dict(t, tens_weigts)
# %%
# tens = (Tensor.arange(10)+100).reshape(1, -1)
# REF_NOPE
# <|begin_of_text|>1,2,3,4,5,6,7,8,8,9,10,11,12,13,14,15,16,17,17,17,19,19,21,21,21,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,25,25,26,26,26,26
prompt_tok = tokenizer.encode("1,2,3,4,5,6,")
sp = 0
for i in range(30):
  # tens = Tensor([prompt_tok[sp:]])
  sp=0
  tens = Tensor([prompt_tok], dtype=dtypes.int64)
  # if i<4: print('encode_TENS', tens.dtype, tens.shape, tens.numpy())
  out = t(tens, sp)
  # print('out',out.shape)
  logits = out[:, -1, :]
  # print('logits',logits.shape)
  tok = logits.softmax(-1).argmax()#.multinomial()
  tok_str = tokenizer.decode(tok.item())

  print('GEN_TOK: ', tok_str, '||', tok.item())
  sp = len(prompt_tok)
  prompt_tok.append(tok.item())


# %%
