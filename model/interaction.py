import torch
import torch.nn as nn

class Interaction(nn.Module):
    def __init__(self, interaction_type):
        super().__init__()

        self.interaction_type = interaction_type

    def forward(self, text, vision):
        if self.interaction_type == "bilinear":
            return torch.bmm(text, vision.transpose(1, 2))

        elif self.interaction_type == "additive":
            return text + vision