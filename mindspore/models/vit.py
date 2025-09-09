# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class PatchEmbed(nn.Cell):
    """ 2D Image to Patch Embedding """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def construct(self, x):
        # x shape: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = ops.flatten(x, start_dim=2)  # (B, embed_dim, H'*W')
        x = ops.transpose(x, (0, 2, 1))  # (B, H'*W', embed_dim)
        return x


class Mlp(nn.Cell):
    """ MLP as used in Vision Transformer """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)
        
    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    """ Multi-head Self Attention """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        
        self.softmax = nn.Softmax(axis=-1)
        self.matmul = ops.BatchMatMul()
        
    def construct(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = ops.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, N, head_dim)
        
        # Attention scores
        attn = self.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale  # (B, num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = self.matmul(attn, v)  # (B, num_heads, N, head_dim)
        x = ops.transpose(x, (0, 2, 1, 3))  # (B, N, num_heads, head_dim)
        x = ops.reshape(x, (B, N, C))
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):
    """ Transformer Block """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)
        
    def construct(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Cell):
    """ Vision Transformer with support for global average pooling """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., norm_layer=None, global_pool=False,
                 **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.get_features = False
        
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                     in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embeddings
        self.cls_token = ms.Parameter(ops.zeros((1, 1, embed_dim), ms.float32))
        self.pos_embed = ms.Parameter(ops.zeros((1, num_patches + 1, embed_dim), ms.float32))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Initialize position embeddings and class token
        self._init_weights()
        
        # Transformer blocks
        self.blocks = nn.SequentialCell([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                 qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                 norm_layer=norm_layer)
            for _ in range(depth)
        ])
        
        # Normalization layer
        if self.global_pool:
            self.fc_norm = norm_layer((embed_dim,))
        else:
            self.norm = norm_layer((embed_dim,))
        
        # Classification head
        self.head = nn.Dense(embed_dim, num_classes)
        
    def _init_weights(self):
        # Initialize class token with truncated normal distribution
        cls_token_np = np.random.normal(0, 0.02, self.cls_token.shape).astype(np.float32)
        self.cls_token.set_data(ms.Tensor(cls_token_np))
        
        # Initialize position embeddings with truncated normal distribution
        pos_embed_np = np.random.normal(0, 0.02, self.pos_embed.shape).astype(np.float32)
        self.pos_embed.set_data(ms.Tensor(pos_embed_np))
        
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Expand class token for batch
        cls_tokens = ops.broadcast_to(self.cls_token, (B, -1, -1))
        x = ops.concat((cls_tokens, x), axis=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Apply normalization and extract features
        if self.global_pool:
            x = x[:, 1:, :].mean(axis=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]  # extract cls token
        
        return outcome
    
    def construct(self, x):
        x = self.forward_features(x)
        features = [x, x]  # keep consistent with other arch
        x = self.head(x)
        if self.get_features:
            return features, x
        return x


def vit_base_patch16(**kwargs):
    """ViT-Base with patch size 16"""
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    """ViT-Large with patch size 16"""
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    """ViT-Huge with patch size 14"""
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model