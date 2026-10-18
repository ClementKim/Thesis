import torch.nn as nn

if __package__:
    from .common import PatchEmbeddingLayer, SelfAttentionBlock, CrossAttentionBlock
else:
    from common import PatchEmbeddingLayer, SelfAttentionBlock, CrossAttentionBlock

DEPTH = 12

class SelfExpert(nn.Module):
    def __init__(self, patch_size, in_c, embed_dim, num_heads, mlp_ratio, drop, attn_drop, is_cls_token):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.is_cls_token = is_cls_token
        self.depth = DEPTH

        self.patch_embed = PatchEmbeddingLayer(in_c, embed_dim, patch_size, self.is_cls_token)
        self.blocks = nn.Sequential(*[
            SelfAttentionBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(self.depth)
        ])

        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)

    def forward(self, x, mask = None):
        x, mask = self.patch_embed(x, mask)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        x = x.masked_fill(~mask[:, :, None], 0)

        return x, mask

class CrossExpert(nn.Module):
    def __init__(self, patch_size, in_c, embed_dim, num_heads, mlp_ratio, drop, attn_drop, is_cls_token):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.is_cls_token = is_cls_token
        self.depth = DEPTH

        self.patch_embed = PatchEmbeddingLayer(in_c, embed_dim, patch_size, self.is_cls_token)
        self.blocks = nn.Sequential(*[
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(self.depth)
        ])

        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)

    def forward(self, x, auxiliary, mask = None):
        x, mask = self.patch_embed(x, mask)
        auxiliary = auxiliary.expand(x.size(0), -1, -1)

        for block in self.blocks:
            x = block(x, auxiliary, mask)

        x = self.norm(x)
        x = x.masked_fill(~mask[:, :, None], 0)

        return x, mask

class HybridExpert(nn.Module):
    def __init__(self, patch_size, in_c, embed_dim, num_heads, mlp_ratio, drop, attn_drop, is_cls_token):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.is_cls_token = is_cls_token
        self.depth = DEPTH

        self.patch_embed = PatchEmbeddingLayer(in_c, embed_dim, patch_size, self.is_cls_token)

        self.blocks = nn.Sequential(*[
            SelfAttentionBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop) if i % 2 == 0 else
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for i in range(self.depth)
        ])

        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)

    def forward(self, x, auxiliary, mask = None):
        x, mask = self.patch_embed(x, mask)
        auxiliary = auxiliary.expand(x.size(0), -1, -1)

        for i in range(self.depth):
            if i % 2 == 0:
                x = self.blocks[i](x, mask)
        
            else:
                x = self.blocks[i](x, auxiliary, mask)
        
        x = self.norm(x)
        x = x.masked_fill(~mask[:, :, None], 0)

        return x, mask
