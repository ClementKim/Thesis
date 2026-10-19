import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicRouter(nn.Module):
    """Give every expert a learned weight for each input feature map."""

    def __init__(self, in_c, num_experts):
        super().__init__()

        self.grid_size = 24
        self.embed_dim = 48

        self.linear = nn.Linear(in_c, self.embed_dim)
        self.gelu = nn.GELU()
        self.attention = nn.Linear(self.embed_dim, 1, bias = False)
        self.gate = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, num_experts)
        )

    def forward(self, x, mask = None):
        B, C, L, _ = x.shape
        H = W = self.grid_size

        # Keep one feature for every text-token and visual-patch pair.
        x = x.permute(0, 2, 3, 1).reshape(B, L, H, W, C).flatten(1, 3)
        if mask is not None:
            mask = mask.to(device = x.device, dtype = torch.bool).reshape(B, L * H * W)
            x = x.masked_fill(~mask[:, :, None], 0)

        # Absolute text, image-row, and image-column positions are independent of padding.
        D = self.embed_dim // 3
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(0, D, 2, device = x.device).float() / D)
        positions = []
        for length in (L, H, W):
            coordinates = torch.arange(length, device = x.device, dtype = torch.float32)[:, None]
            angles = coordinates * frequencies[None]
            position = torch.zeros((length, D), device = x.device)
            position[:, 0::2] = angles.sin()
            position[:, 1::2] = angles.cos()
            positions.append(position)
        position = torch.cat((positions[0][:, None, None].expand(-1, H, W, -1),
                              positions[1][None, :, None].expand(L, -1, W, -1),
                              positions[2][None, None].expand(L, H, -1, -1)), dim = -1)

        x = self.linear(x)
        x = self.gelu(x + position.reshape(1, L * H * W, self.embed_dim).to(x.dtype))
        scores = self.attention(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -torch.inf)

        # Learn which cells to aggregate, accumulating in float32 for half precision.
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        attention = F.softmax(scores.to(values.dtype), dim = -1)
        features = (attention[:, :, None] * values).sum(dim = 1)
        logits = self.gate(features.to(x.dtype))
        weights = F.softmax(logits, dim = -1)

        return weights

class MoE(nn.Module):
    """Keep each expert's output with its own weight for each input."""

    def __init__(self, in_c, self_expert, cross_expert, hybrid_expert, router_mode = 'learned'):
        super().__init__()

        if router_mode not in ('learned', 'uniform', 'static'):
            raise ValueError(f'Unsupported router mode: {router_mode!r}')

        # Construct the original router in every arm to preserve subsequent initialization.
        self.router = DynamicRouter(in_c, num_experts = 3)
        self.self_expert = self_expert
        self.cross_expert = cross_expert
        self.hybrid_expert = hybrid_expert
        self.router_mode = router_mode

        if router_mode != 'learned':
            self.router.requires_grad_(False)

        if router_mode == 'static':
            self.router_logits = nn.Parameter(torch.zeros(3))
        else:
            self.register_parameter('router_logits', None)

    def forward(self, x, auxiliary, mask = None, router_x = None):
        """Return (B, 3N, D) tokens, (B, 3N) masks, and (B, 3) weights."""
        router_x = x if router_x is None else router_x

        if self.router_mode == 'uniform':
            weights = x.new_full((x.size(0), 3), 1.0 / 3.0)
        elif self.router_mode == 'static':
            weights = F.softmax(self.router_logits, dim = -1).unsqueeze(0).expand(x.size(0), -1)
        else:
            weights = self.router(router_x, mask)

        self_output, self_mask = self.self_expert(x, mask)
        cross_output, cross_mask = self.cross_expert(x, auxiliary, mask)
        hybrid_output, hybrid_mask = self.hybrid_expert(x, auxiliary, mask)

        # All experts must produce the same (B, N, D) shape.
        expert_outputs = torch.stack([self_output, cross_output, hybrid_output], dim = 1)
        expert_masks = torch.stack([self_mask, cross_mask, hybrid_mask], dim = 1)

        # Apply each sample's expert weights to all its tokens and features.
        out = expert_outputs * weights[:, :, None, None]
        # Keep the weighted outputs as separate tokens for the LLM.
        out = out.flatten(1, 2)
        out_mask = expert_masks.flatten(1, 2)

        return out, out_mask, weights
