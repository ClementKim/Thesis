import torch
import random

import numpy as np
import torch.nn as nn

from sys import os

# Dataset
from preprocessing import train_dataset

# Proposed Method
from model.encoder import CLIPEncoder, TextEncoder, VisionEncoder
from model.interaction import Interaction
from model.experts import SelfExpert, CrossExpert
from model.router import Router
from model.llm import LlamaLLM

class IPMoE(nn.Module):
    def __init__(self):
        super().__init__()

        CLIP = CLIPEncoder()
        

def main(args):
    ## Random Seed
    seed = args.seed

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only = True)

    # Allow TF32 on Ampere+ GPUs for faster matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ## Device Setting
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    ## Dataset and Dataloader
    train, val = train_dataset(0.01, seed)

if __name__ == "__main__":
    main()