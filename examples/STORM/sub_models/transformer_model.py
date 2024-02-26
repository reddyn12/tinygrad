# import torch
# import torch.nn as tnn
# import torch.nn.functional as F

from typing import Any
from tinygrad import Tensor, dtypes, nn
import tinygrad

# from einops import repeat, rearrange

from sub_models.attention_blocks import get_vector_mask
from sub_models.attention_blocks import (
    PositionalEncoding1D,
    AttentionBlock,
    AttentionBlockKVCache,
)


class StochasticTransformer:
    def __init__(
        self,
        stoch_dim,
        action_dim,
        feat_dim,
        num_layers,
        num_heads,
        max_length,
        dropout,
    ):
        # super().__init__()
        self.action_dim = action_dim

        # mix image_embedding and action

        # self.stem = nn.Sequential(
        #     nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
        #     nn.LayerNorm(feat_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feat_dim, feat_dim, bias=False),
        #     nn.LayerNorm(feat_dim)
        # )

        self.stem = [
            nn.Linear(stoch_dim + action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            Tensor.relu(),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
        ]

        self.position_encoding = PositionalEncoding1D(
            max_length=max_length, embed_dim=feat_dim
        )
        # self.layer_stack = nn.ModuleList([
        #     AttentionBlock(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        # ])
        self.layer_stack = [
            AttentionBlock(
                feat_dim=feat_dim,
                hidden_dim=feat_dim * 2,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        self.layer_norm = nn.LayerNorm(
            feat_dim, eps=1e-6
        )  # TODO: check if this is necessary

        self.head = nn.Linear(feat_dim, stoch_dim)

    def forward(self, samples, action: Tensor, mask):
        # action = F.one_hot(action.long(), self.action_dim).float()
        action = action.cast(dtypes.long).one_hot(self.action_dim).float()
        # feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = Tensor.cat([samples, action], dim=-1).sequential(self.stem)
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for enc_layer in self.layer_stack:
            feats, attn = enc_layer(feats, mask)

        feat = self.head(feats)
        return feat


class StochasticTransformerKVCache:
    def __init__(
        self,
        stoch_dim,
        action_dim,
        feat_dim,
        num_layers,
        num_heads,
        max_length,
        dropout,
    ):
        # super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim

        # mix image_embedding and action
        # self.stem = nn.Sequential(
        #     nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
        #     nn.LayerNorm(feat_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feat_dim, feat_dim, bias=False),
        #     nn.LayerNorm(feat_dim)
        # )

        # print('Stoch_STORM_TRANSF',type(feat_dim), feat_dim, stoch_dim+action_dim, type(stoch_dim+action_dim))

        self.stem = [
            nn.Linear(int(stoch_dim + action_dim), feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            Tensor.relu,
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
        ]

        self.position_encoding = PositionalEncoding1D(
            max_length=max_length, embed_dim=feat_dim
        )
        # self.layer_stack = nn.ModuleList([
        #     AttentionBlockKVCache(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        # ])
        self.layer_stack = [
            AttentionBlockKVCache(
                feat_dim=feat_dim,
                hidden_dim=feat_dim * 2,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        self.layer_norm = nn.LayerNorm(
            feat_dim, eps=1e-6
        )  # TODO: check if this is necessary

    def forward(self, samples, action: Tensor, mask):
        """
        Normal forward pass
        """
        # action = F.one_hot(action.long(), self.action_dim).float()
        # action = action.cast(dtypes.long).one_hot(self.action_dim).float()
        # print(action.numpy())
        # print(action.shape,action.dtype)

        # print(self.action_dim)
        # action = action.realize()
        action = action.one_hot(self.action_dim).float()
        # print('action one hot:', action.shape,action.dtype)
        # for s in action.shape:
        #     print(s, type(s))
        # feats = self.stem(torch.cat([samples, action], dim=-1))
        # feats = self.stem(Tensor.cat([samples, action], dim=-1))
        feats = Tensor.cat(*[samples, action], dim=-1).sequential(self.stem)
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)

        return feats

    def __call__(self, samples, action: Tensor, mask):
        return self.forward(samples, action, mask)

    def reset_kv_cache_list(self, batch_size, dtype):
        """
        Reset self.kv_cache_list
        """
        self.kv_cache_list = []
        for layer in self.layer_stack:
            # self.kv_cache_list.append(torch.zeros(size=(batch_size, 0, self.feat_dim), dtype=dtype, device="cuda"))
            self.kv_cache_list.append(
                Tensor.zeros((batch_size, 0, self.feat_dim), dtype=dtype)
            )

    def forward_with_kv_cache(self, samples, action):
        """
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        """
        assert samples.shape[1] == 1
        mask = get_vector_mask(self.kv_cache_list[0].shape[1] + 1, samples.device)

        # action = F.one_hot(action.long(), self.action_dim).float()
        action = action.cast(dtypes.long).one_hot(self.action_dim).float()
        # feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = Tensor.cat(*[samples, action], dim=-1).sequential(self.stem)
        feats = self.position_encoding.forward_with_position(
            feats, position=self.kv_cache_list[0].shape[1]
        )
        feats = self.layer_norm(feats)

        for idx, layer in enumerate(self.layer_stack):
            # self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], feats], dim=1)
            self.kv_cache_list[idx] = Tensor.cat(
                *[self.kv_cache_list[idx], feats], dim=1
            )
            feats, attn = layer(
                feats, self.kv_cache_list[idx], self.kv_cache_list[idx], mask
            )

        return feats