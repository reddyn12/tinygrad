from typing import Any
from tinygrad import Tensor
from tinygrad.nn import Conv2d, Embedding, LayerNorm, Linear
from tinygrad.nn.state import get_state_dict, torch_load, load_state_dict
from tinygrad.helpers import fetch
class CLIP:
  def __init__(self):
    self.visual = CLIP_Vision()
    text = CLIP_Text()
    self.ln_final = text.ln_final
    self.transformer = text.transformer
    self.token_embedding = text.embeddings.token_embedding
    self.positional_embedding = Tensor.empty(77, 1024)
    self.text_projection = text.text_projection
  
  def encode_image(self, x:Tensor):
    return self.visual(x)
  
  def encode_text(self, text:Tensor):
    x = self.token_embedding(text)
    x = x + self.positional_embedding
    x = x.permute(1,0,2)
    
    x = self.transformer(x)
    x = x.permute(1,0,2)
    x = self.ln_final(x)
    x = x[Tensor.arange(x.shape[0]), text.argmax(axis=-1)]
    
    return x@self.text_projection
  
  def get_clip_score(self, img:Tensor, txt:Tensor):
    img_feat = self.encode_image(img)
    norm = (img_feat**2).sum(-1)**0.5
    img_feat = img_feat/norm.unsqueeze(-1)
    
    txt_feat = self.encode_text(txt)
    norm = (txt_feat**2).sum(-1)**0.5
    txt_feat = txt_feat/norm.unsqueeze(-1)
    
    return img_feat@txt_feat.T
    
  def load_pretrained(self):
    w = torch_load(fetch('https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin', 'clip_h-14.bin'))
    w_new = {}
    for k,v in w.items():
      new_k = k.replace('in_proj_', 'in_proj.')
      w_new[new_k] = v
    load_state_dict(self, w_new)
    
    

class CLIP_Text:
  def __init__(self, width=1024, layers=24):
    self.embeddings = CLIPTextEmbeddings()
    self.transformer = CLIPEncoder(width, 16, layers)
    self.ln_final = LayerNorm(width)
    self.text_projection = Tensor.empty((width,width))

  def __call__(self, input_ids):
    x = self.embeddings(input_ids, Tensor.arange(input_ids.shape[1]).reshape(1, -1))
    x = self.transformer(x, Tensor.full((1, 1, 77, 77), float("-inf")).triu(1))
    x = self.ln_final(x)
    # text gloabl pool
    x = x[Tensor.arange(x.shape[0]), input_ids.argmax(axis=-1)]
    return x @ self.text_projection



class CLIP_Vision:
  def __init__(self, image_size=224, layers=32, width=1280, head_width=80, patch_size=14):
    grid_size = image_size//patch_size
    
    self.conv1 = Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
    self.class_embedding = Tensor.empty(width)
    self.positional_embedding = Tensor.empty(grid_size**2+1, width)
    self.transformer = CLIPEncoder(width, head_width,layers)
    self.ln_pre = LayerNorm(width)
    self.ln_post = LayerNorm(width)
    self.proj = Tensor.empty(width, 1024)
  def __call__(self, x:Tensor) -> Any:
    x = self.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    
    x = Tensor.cat(self.class_embedding.reshape(1,1,-1).expand((x.shape[0], -1, -1)), x, dim=1)
    
    x = x + self.positional_embedding
    x = self.ln_pre(x)
    x = x.permute(1,0,2)
    x = self.transformer(x)
    x = x.permute(1,0,2)
    
    x = self.ln_post(x)
    pooled, tokens = x[:, 0], x[:, 1:]
    
    return pooled@self.proj
    
    
    
class CLIPMLP:
  def __init__(self, width):
    self.c_fc = Linear(width, width*4)
    self.c_proj = Linear(width*4, width)

  def __call__(self, hidden_states):
    hidden_states = self.c_fc(hidden_states)
    hidden_states = hidden_states.quick_gelu()
    hidden_states = self.c_proj(hidden_states)
    return hidden_states

class CLIPAttention:
  # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
  def __init__(self, embed_dim=1024, num_heads=8):
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = self.embed_dim // self.num_heads

    self.in_proj = Linear(self.embed_dim, self.embed_dim*3)
    self.out_proj = Linear(self.embed_dim, self.embed_dim)

  def __call__(self, hidden_states, causal_attention_mask=None):
    L, N, C = hidden_states.shape
    q,k,v = self.in_proj(hidden_states).chunk(3,-1)
    q,k,v = [x.reshape(L, N * self.num_heads, -1).transpose(0, 1) for x in (q,k,v)]
    attn_output = Tensor.scaled_dot_product_attention(q, k, v, attn_mask=causal_attention_mask)
    return self.out_proj(attn_output.transpose(0, 1).reshape(L, N, C))

class CLIPEncoderLayer:
  def __init__(self, width=1024, num_heads=8):
    self.attn = CLIPAttention(width, num_heads)
    self.ln_1 = LayerNorm(width)
    self.mlp = CLIPMLP(width)
    self.ln_2 = LayerNorm(width)

  def __call__(self, hidden_states, causal_attention_mask):
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    hidden_states = self.attn(hidden_states, causal_attention_mask)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states

class CLIPEncoder:
  def __init__(self, width=1024, num_heads=8, layers=24):
    self.resblocks = [CLIPEncoderLayer(width, num_heads) for i in range(layers)]

  def __call__(self, hidden_states, causal_attention_mask):
    for l in self.resblocks:
      hidden_states = l(hidden_states, causal_attention_mask)
    return hidden_states

class CLIPTextEmbeddings:
  def __init__(self):
    self.token_embedding = Embedding(49408, 1024)
    self.positional_embedding = Embedding(77, 1024)

  def __call__(self, input_ids, position_ids):
    return self.token_embedding(input_ids) + self.positional_embedding(position_ids)
