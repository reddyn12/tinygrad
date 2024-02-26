# import torch
# import torch.nn as nn
# import torch.nn.functional as F

from typing import Any
from tinygrad import Tensor, dtypes, nn
import tinygrad

# import numpy as np
# from einops import rearrange, repeat


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    batch_size, batch_length = seq.shape[:2]
    # subsequent_mask = (1 - torch.triu(
    #     torch.ones((1, batch_length, batch_length), device=seq.device), diagonal=1)).bool()

    subsequent_mask = (
        (1 - Tensor.ones((1, batch_length, batch_length), device=seq.device))
        .triu(1)
        .cast(dtypes.bool)
    )
    return subsequent_mask


def get_subsequent_mask_with_batch_length(batch_length, device):
    """For masking out the subsequent info."""
    # subsequent_mask = (1 - torch.triu(torch.ones((1, batch_length, batch_length), device=device), diagonal=1)).bool()
    subsequent_mask = (
        (1 - Tensor.ones((1, batch_length, batch_length))).triu(1).cast(dtypes.bool)
    )
    return subsequent_mask


def get_vector_mask(batch_length, device):
    # mask = torch.ones((1, 1, batch_length), device=device).bool()
    # # mask = torch.ones((1, batch_length, 1), device=device).bool()
    mask = Tensor.ones((1, 1, batch_length), device=device).cast(dtypes.bool)
    return mask


class ScaledDotProductAttention:
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        # super().__init__()
        self.temperature = temperature
        self.dropoutVal = attn_dropout

    def forward(self, q, k, v, mask: Tensor = None):
        # attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = Tensor.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn_mask = (mask == 0).where(-1e9, 0)
            attn = attn + attn_mask

        # attn = self.dropout(F.softmax(attn, dim=-1))
        attn = attn.softmax(axis=-1).dropout(self.dropoutVal)
        # output = torch.matmul(attn, v)
        output = Tensor.matmul(attn, v)

        return output, attn

    def __call__(self, q, k, v, mask=None):
        return self.forward(q, k, v, mask)


class MultiHeadAttention:
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropoutVal = dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q: Tensor, k, v, mask: Tensor = None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        sz_b, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # print('SHAPEEE:', self.w_qs(q).shape)
        q = self.w_qs(q).reshape((sz_b, len_q, n_head, d_k))
        k = self.w_ks(k).reshape((sz_b, len_k, n_head, d_k))
        v = self.w_vs(v).reshape((sz_b, len_v, n_head, d_v))

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().reshape((sz_b, len_q, -1))
        # q = self.dropout(self.fc(q))
        q = self.fc(q).dropout(self.dropoutVal)
        q = q + residual

        q = self.layer_norm(q)

        return q, attn

    def __call__(self, q, k, v, mask=None):
        return self.forward(q, k, v, mask=mask)


class PositionwiseFeedForward:
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropoutVal = dropout

    def forward(self, x):
        residual = x

        # x = self.w_2(F.relu(self.w_1(x)))
        x = self.w_2(self.w_1(x).relu())
        x = x.dropout(self.dropoutVal)
        x = x + residual

        x = self.layer_norm(x)

        return x

    def __call__(self, x):
        return self.forward(x)


class AttentionBlock:
    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        self.slf_attn = MultiHeadAttention(
            num_heads,
            feat_dim,
            feat_dim // num_heads,
            feat_dim // num_heads,
            dropout=dropout,
        )
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

    def __call__(self, enc_input, slf_attn_mask=None):
        return self.forward(enc_input, slf_attn_mask)


class AttentionBlockKVCache:
    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        self.slf_attn = MultiHeadAttention(
            num_heads,
            feat_dim,
            feat_dim // num_heads,
            feat_dim // num_heads,
            dropout=dropout,
        )
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, q, k, v, slf_attn_mask=None):
        output, attn = self.slf_attn(q, k, v, mask=slf_attn_mask)
        output = self.pos_ffn(output)
        return output, attn

    def __call__(self, q, k, v, slf_attn_mask=None):
        return self.forward(q, k, v, slf_attn_mask)


class PositionalEncoding1D:
    def __init__(self, max_length: int, embed_dim: int):
        self.max_length = max_length
        self.embed_dim = embed_dim

        # print('posEncINIT',self.max_length, self.embed_dim)
        self.pos_emb = nn.Embedding(int(self.max_length), int(embed_dim))

    def forward(self, feat: Tensor):
        # pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        # a = Tensor.arange(self.max_length).reshape(self.max_length, 1)
        # print('PosEncoding1D', self.max_length, a, a.shape)

        # reshpae done to remove np.int64 shape
        pos_emb = self.pos_emb(
            Tensor.arange(self.max_length).reshape(self.max_length, 1)
        ).squeeze()

        # print('pre',feat.shape, pos_emb.shape)
        # pos_emb = repeat(pos_emb, "L D -> B L D", B=feat.shape[0])
        pos_emb = pos_emb.repeat((feat.shape[0], 1, 1))  # .realize()
        # print('post',feat.shape, pos_emb.shape)
        feat = feat + pos_emb[:, : feat.shape[1], :]
        return feat

    def forward_with_position(self, feat: Tensor, position):
        assert feat.shape[1] == 1
        # pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        # pos_emb = self.pos_emb(Tensor.arange(self.max_length, device=feat.device))
        pos_emb = self.pos_emb(
            Tensor.arange(self.max_length).reshape(self.max_length, 1)
        ).squeeze()
        # pos_emb = repeat(pos_emb, "L D -> B L D", B=feat.shape[0])
        pos_emb = pos_emb.repeat((feat.shape[0], 1, 1))

        feat = feat + pos_emb[:, position : position + 1, :]
        return feat

    def __call__(self, feat: Tensor):
        return self.forward(feat)