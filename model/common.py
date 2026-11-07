import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbeddingLayer(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patchify = nn.Conv2d(in_channels, embed_dim, kernel_size = patch_size, stride = patch_size)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask = None, column_indices = None):
        B, _, H, W = x.shape

        if column_indices is not None and tuple(column_indices.shape) != (B, W):
            raise ValueError('Selected image indices must match the map batch and columns')

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

        # Retain original map-block coordinates after selection; ignore padded indices.
        positions = []
        for length, D, indices in ((H, self.embed_dim // 2, None), (W, self.embed_dim - self.embed_dim // 2, column_indices)):
            if indices is None:
                coordinates = torch.arange(length, device = x.device, dtype = torch.float32)[None].expand(B, -1)
            else:
                indices = indices.to(device = x.device)
                indices = F.pad(indices, (0, length * self.patch_size - indices.size(1)), value = -1)
                indices = indices.reshape(B, length, self.patch_size)
                valid = indices >= 0
                coordinates = torch.div(indices.clamp_min(0), self.patch_size, rounding_mode = 'floor').float()
                coordinates = coordinates.sum(dim = -1) / valid.sum(dim = -1).clamp_min(1)
            frequencies = torch.exp(-math.log(10000.0) * torch.arange(0, D, 2, device = x.device).float() / max(D, 1))
            angles = coordinates[:, :, None] * frequencies[None, None]
            position = torch.zeros((B, length, D), device = x.device)
            position[:, :, 0::2] = angles.sin()
            position[:, :, 1::2] = angles[:, :, :D // 2].cos()
            positions.append(position)

        positional_embedding = torch.cat((positions[0][:, :, None].expand(-1, -1, W, -1), positions[1][:, None].expand(-1, H, -1, -1)), dim = -1)

        x = x + positional_embedding.reshape(B, H * W, self.embed_dim).to(x.dtype)
        x = x.masked_fill(~mask[:, :, None], 0)

        return x, mask
