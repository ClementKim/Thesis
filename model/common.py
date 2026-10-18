import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbeddingLayer(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, is_cls_token = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.is_cls_token = is_cls_token
        self.patch_size = patch_size

        self.patchify = nn.Conv2d(in_channels, embed_dim, kernel_size = patch_size, stride = patch_size)
        self.linear = nn.Linear(embed_dim, embed_dim)
        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn((1, 1, embed_dim)))

    def forward(self, x, mask = None):
        B, C, H, W = x.shape

        if mask is None:
            mask = torch.ones((B, H, W), dtype = torch.bool, device = x.device)
        mask = mask.to(device = x.device, dtype = torch.bool)
        x = x.masked_fill(~mask[:, None], 0)

        # Keep boundary cells when the rectangular map is not divisible by the patch size.
        pad_h = (-H) % self.patch_size
        pad_w = (-W) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h))
        mask = F.pad(mask, (0, pad_w, 0, pad_h), value = False)
        mask = F.max_pool2d(mask[:, None].float(), self.patch_size, self.patch_size).flatten(1).bool()

        x = self.patchify(x)
        H, W = x.shape[2:]
        x = torch.flatten(x, 2).transpose(1, 2)

        x = self.linear(x)

        # Absolute row/column positions do not change when another sample adds padding.
        positions = []
        for length, D in ((H, self.embed_dim // 2), (W, self.embed_dim - self.embed_dim // 2)):
            coordinates = torch.arange(length, device = x.device, dtype = torch.float32)[:, None]
            frequencies = torch.exp(-math.log(10000.0) * torch.arange(0, D, 2, device = x.device).float() / max(D, 1))
            angles = coordinates * frequencies[None]
            position = torch.zeros((length, D), device = x.device)
            position[:, 0::2] = angles.sin()
            position[:, 1::2] = angles[:, :D // 2].cos()
            positions.append(position)

        positional_embedding = torch.cat((positions[0][:, None].expand(-1, W, -1), positions[1][None].expand(H, -1, -1)), dim = -1)

        x = x + positional_embedding.reshape(1, H * W, self.embed_dim).to(x.dtype)
        x = x.masked_fill(~mask[:, :, None], 0)

        if self.is_cls_token:
            cls = self.cls_token.expand(B, 1, self.embed_dim)
            x = torch.cat((cls, x), dim = 1)
            mask = torch.cat((torch.ones((B, 1), dtype = torch.bool, device = x.device), mask), dim = 1)

        return x, mask

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

    def forward(self, x, mask = None):
        B, N, D = x.shape

        if mask is not None:
            mask = mask.to(device = x.device, dtype = torch.bool)
            x = x.masked_fill(~mask[:, :, None], 0)

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(2, 3) * self.scale

        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)

        A = F.softmax(scores, dim = -1)

        SA = (A @ v).transpose(1, 2).reshape(B, N, D)

        out = self.dropout(self.o(SA))

        if mask is not None:
            out = out.masked_fill(~mask[:, :, None], 0)

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

    def forward(self, x, context, mask = None):
        B, N, D = x.shape
        M = context.size(1)

        if mask is not None:
            mask = mask.to(device = context.device, dtype = torch.bool)
            context = context.masked_fill(~mask[:, :, None], 0)

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(2, 3) * self.scale

        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)

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

    def forward(self, x, mask = None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        
        if mask is not None:
            x = x.masked_fill(~mask.to(device = x.device, dtype = torch.bool)[:, :, None], 0)

        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 2.0, drop = 0.0, attn_drop = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MultiHeadCrossAttention(dim, num_heads, attn_drop)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x, auxiliary, mask = None):
        x = x + self.attn(self.norm1(x), auxiliary)
        x = x + self.mlp(self.norm2(x))

        if mask is not None:
            mask = mask.to(device = x.device, dtype = torch.bool)
            x = x.masked_fill(~mask[:, :, None], 0)

        return x
