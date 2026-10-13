import torch
import torch.nn as nn

if __package__:
    from .common import PatchEmbeddingLayer, SelfAttentionBlock, CrossAttentionBlock
else:
    from common import PatchEmbeddingLayer, SelfAttentionBlock, CrossAttentionBlock

DEPTH = 12

class SelfExpert(nn.Module):
    def __init__(self, interaction_size, patch_size, in_c, embed_dim, num_heads, mlp_ratio, drop, attn_drop, is_cls_token):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.is_cls_token = is_cls_token
        self.depth = DEPTH

        self.patch_embed = PatchEmbeddingLayer(in_c, embed_dim, interaction_size, patch_size, self.is_cls_token)
        self.blocks = nn.Sequential(*[
            SelfAttentionBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(self.depth)
        ])

        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)

        return x

class CrossExpert(nn.Module):
    def __init__(self, interaction_size, patch_size, in_c, embed_dim, num_heads, mlp_ratio, drop, attn_drop, is_cls_token):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.is_cls_token = is_cls_token
        self.depth = DEPTH

        self.patch_embed = PatchEmbeddingLayer(in_c, embed_dim, interaction_size, patch_size, self.is_cls_token)
        self.blocks = nn.Sequential(*[
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(self.depth)
        ])

        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)

    def forward(self, x, auxiliary):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, auxiliary)
        x = self.norm(x)

        return x

class Hybrid(nn.Module):
    def __init__(self, interaction_size, patch_size, in_c, embed_dim, num_heads, mlp_ratio, drop, attn_drop, is_cls_token):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.is_cls_token = is_cls_token
        self.depth = DEPTH

        self.patch_embed = PatchEmbeddingLayer(in_c, embed_dim, interaction_size, patch_size, self.is_cls_token)
        self.blocks = nn.Sequential(*[
            SelfAttentionBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop) if i % 2 == 0 else
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for i in range(self.depth)
        ])

        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)

    def forward(self, x, auxiliary):
        x = self.patch_embed(x)
        for i in range(self.depth):
            if i % 2 == 0:
                x = self.blocks[i](x)
            else:
                x = self.blocks[i](x, auxiliary)
        x = self.norm(x)

        return x
