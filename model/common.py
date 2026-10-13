import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbeddingLayer(nn.Module):
    def __init__(self, in_channels, embed_dim, img_size, patch_size, is_cls_token = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.is_cls_token = is_cls_token

        self.patchify = nn.Conv2d(in_channels, embed_dim, kernel_size = patch_size, stride = patch_size)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn((1, 1, embed_dim)))

        if is_cls_token:
            self.positional_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))

        else:
            self.positional_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.patchify(x)
        x = torch.flatten(x, 2).transpose(1, 2)

        x = self.linear(x)

        if self.is_cls_token:
            cls = self.cls_token.expand(B, 1, self.embed_dim)
            x = torch.cat((cls, x), dim = 1)

        x = x + self.positional_embedding

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim = 192, num_heads = 12, drop = 0.0):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        B, N, D = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        A = F.softmax(q @ k.transpose(2, 3) * self.scale, dim = -1)

        SA = (A @ v).transpose(1, 2).reshape(B, N, D)

        out = self.dropout(self.o(SA))

        return out

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim = 192, num_heads = 12, drop = 0.0):
        super(MultiHeadCrossAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(drop)

    def forward(self, x, context):
        B, N, D = x.shape
        M = context.size(1)

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(2, 3) * self.scale

        A = F.softmax(scores, dim = -1)

        CA = (A @ v).transpose(1, 2).reshape(B, N, D)

        out = self.dropout(self.o(CA))

        return out

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop = 0.0):
        super().__init__()

        out_features = in_features

        self.linear1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return self.dropout(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 2.0, drop = 0.0, attn_drop = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 2.0, drop = 0.0, attn_drop = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MultiHeadCrossAttention(dim, num_heads, attn_drop)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x, auxiliary):
        x = x + self.attn(self.norm1(x), auxiliary)
        x = x + self.mlp(self.norm2(x))

        return x