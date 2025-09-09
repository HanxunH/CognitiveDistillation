# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch


class PatchEmbed(layers.Layer):
    """Image to Patch Embedding"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)
        
    def call(self, x):
        B, H, W, C = x.shape
        x = self.proj(x)  # (B, H', W', embed_dim)
        x = tf.reshape(x, [B, -1, x.shape[-1]])  # (B, num_patches, embed_dim)
        return x


class Attention(layers.Layer):
    """Multi-head self-attention"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)
        
    def call(self, x, training=False):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


class Mlp(layers.Layer):
    """MLP block"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=layers.Activation('gelu'), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features)
        self.act = act_layer
        self.fc2 = layers.Dense(out_features)
        self.drop = layers.Dropout(drop)
        
    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


class Block(layers.Layer):
    """Transformer block"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=layers.Activation('gelu'), norm_layer=layers.LayerNormalization):
        super().__init__()
        self.norm1 = norm_layer(epsilon=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # Note: drop path for stochastic depth, not implemented here for simplicity
        self.norm2 = norm_layer(epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def call(self, x, training=False):
        x = x + self.attn(self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x


class VisionTransformer(keras.Model):
    """Vision Transformer with support for global average pooling"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 representation_size=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, global_pool=False, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.get_features = False
        norm_layer = norm_layer or partial(layers.LayerNormalization, epsilon=1e-6)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token and positional embedding
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, embed_dim),
            initializer='zeros',
            trainable=True
        )
        self.pos_embed = self.add_weight(
            name='pos_embed',
            shape=(1, num_patches + 1, embed_dim),
            initializer='zeros',
            trainable=True
        )
        self.pos_drop = layers.Dropout(drop_rate)
        
        # Transformer blocks
        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = [
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ]
        
        # Final norm and head
        if self.global_pool:
            self.fc_norm = norm_layer(epsilon=1e-6)
        else:
            self.norm = norm_layer(epsilon=1e-6)
        
        # Classification head
        self.head = layers.Dense(num_classes) if num_classes > 0 else tf.identity
        
        # Try to load pretrained weights
        try:
            state_dict = torch.load('checkpoints/mae_pretrain_vit_base.pth')['model']
            print("Loaded PyTorch checkpoint: checkpoints/mae_pretrain_vit_base.pth")
            # Note: Weight conversion from PyTorch to TensorFlow would require manual mapping
            # For now, we'll skip the actual weight loading
            print("Warning: Weight conversion from PyTorch to TensorFlow not implemented")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
    
    def forward_features(self, x, training=False):
        B = tf.shape(x)[0]
        x = self.patch_embed(x)
        
        cls_tokens = tf.broadcast_to(self.cls_token, [B, 1, self.embed_dim])
        x = tf.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x, training=training)
        
        for blk in self.blocks:
            x = blk(x, training=training)
        
        if self.global_pool:
            x = tf.reduce_mean(x[:, 1:, :], axis=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        
        return outcome
    
    def call(self, x, training=False):
        # Convert input to channels-last format if needed (TensorFlow uses NHWC by default)
        # Input is expected to be (B, C, H, W) from PyTorch, convert to (B, H, W, C)
        if len(x.shape) == 4 and x.shape[1] in [1, 3]:  # Likely channels-first
            x = tf.transpose(x, [0, 2, 3, 1])
        
        x = self.forward_features(x, training=training)
        features = [x, x]  # keep consistent with other arch
        x = self.head(x)
        
        if self.get_features:
            return features, x
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), **kwargs)
    return model