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

# %%
for k,v in d['model'].items():
  print(k, v.shape, v.dtype)
# %%
d['optimizer'].keys()
# %%
from typing import Tuple, Union, Optional, Dict, Any
from tinygrad import Tensor, Variable, nn
from tinygrad.helpers import getenv
import math

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

  def __call__(self, x:Tensor, start_pos:Union[Variable,int], mask:Optional[Tensor]=None) -> Tensor:
    bsz, seq_len, embed_dim = x.shape
    print('ATTENTION_SHAPE', x.shape)
    xqkv = self.attn(x)
    xq, xk, xv = xqkv.split([self.attn.weight.shape[0]//3, self.attn.weight.shape[0]//3, self.attn.weight.shape[0]//3], dim=2)


    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

    # xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    
    # freq_constant = 10000
    # inv_freq = 1.0 / (freq_constant ** (Tensor.arange(0, self.head_dim, 2) / self.head_dim))
    # pos_index_theta = Tensor.einsum("i,j->ij", Tensor.arange(seq_len), inv_freq)
    # emb = Tensor.cat(pos_index_theta, pos_index_theta, dim=-1)
    # cos_emb, sin_emb = emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
    # print('ROPE:', emb.shape, cos_emb.shape, sin_emb.shape)
    # xq = _apply_rotary_pos_emb(xq, sin_emb, cos_emb)
    # xk = _apply_rotary_pos_emb(xk, sin_emb, cos_emb)

    # create kv cache
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype).contiguous().realize()
      if isinstance(x.device, tuple):
        # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded
        self.cache_kv.shard_((x.device), axis=3 if getenv("SHARD_KVCACHE") else None).realize()

    # update the cache
    assert xk.dtype == xv.dtype == self.cache_kv.dtype, f"{xk.dtype=}, {xv.dtype=}, {self.cache_kv.dtype=}"
    self.cache_kv.shrink((None, None, (start_pos, start_pos+seq_len), None, None)).assign(Tensor.stack(xk, xv)).realize()

    keys = self.cache_kv[0].shrink((None, (0, start_pos+seq_len), None, None)) if start_pos > 0 else xk
    values = self.cache_kv[1].shrink((None, (0, start_pos+seq_len), None, None)) if start_pos > 0 else xv

    # keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    # attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2)
    attn = sliding_window_attention(xq, keys, values, self.window_size, mask)
    attn = attn.reshape(bsz, seq_len, -1)
    return self.proj(attn)
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
    self.norm_1 = nn.RMSNorm(dim)
    if layer_idx%2==0:
      self.attn = MambaMixer(dim)
    else:
      self.attn = Attention(dim, 8, 8, 4092, 2048)
    self.norm_2 = nn.RMSNorm(dim)
    self.swiglu = SwiGLU(dim)
  def __call__(self, x:Tensor):
    print('LAYER_IDX', self.layer_idx)
    n_1 = self.norm_1(x)
    if self.layer_idx%2==0:
      h = self.attn(n_1)
    else:
      h = self.attn(n_1, 0)
    
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
    self.ln_f = nn.RMSNorm(dim)
  def __call__(self, x:Tensor):
    out = self.wte(x)
    out = out.sequential(self.layers)
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
tens = (Tensor.arange(10)+100).reshape(1, -1)
out = t(tens)
print(out.shape)
# %%
