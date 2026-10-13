import torch
import random

import numpy as np
import torch.nn as nn

from sys import os

# Dataset
from preprocessing import train_dataset

# Proposed Method
from model.encoder import encoding
from model.interaction import Interaction
from model.experts import SelfExpert, CrossExpert, HybridExpert
from model.router import Router
from model.llm import LlamaLLM, Qwen25VLLLM

def reproduce(seed):
    try:
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

    except Exception as e:
        raise RuntimeError(f"Failed to set random seed: {e}")

def build_model(device, args):
    ## CLIP
    processor, text_encoder, vision_encoder = encoding(args.drop, device, args.embed_dim)

    ## Interaction Matrix Constructor
    interaction = Interaction(args.interaction_type)

    ## Experts
    self_experts   = nn.ModuleList([SelfExpert(args.interaction_size, args.patch_size, args.in_c, args.embed_dim, args.num_heads, args.mlp_ratio, args.drop, args.attn_drop, args.is_cls_token) for _ in range(args.num_experts)])
    cross_experts  = nn.ModuleList([CrossExpert(args.interaction_size, args.patch_size, args.in_c, args.embed_dim, args.num_heads, args.mlp_ratio, args.drop, args.attn_drop, args.is_cls_token) for _ in range(args.num_experts)])
    hybrid_experts = nn.ModuleList([HybridExpert(args.interaction_size, args.patch_size, args.in_c, args.embed_dim, args.num_heads, args.mlp_ratio, args.drop, args.attn_drop, args.is_cls_token) for _ in range(args.num_experts)])

    ## Router
    router = Router(args.embed_dim, args.num_experts)

    ## LLM
    if args.llm_type == "llama":
        llm = LlamaLLM(args.llm_size, args.lora_rank, args.attn_implementation, args.gradient_checkpointing)

    elif args.llm_type == "qwen":
        llm = Qwen25VLLLM(args.llm_size, args.lora_rank, args.attn_implementation, args.gradient_checkpointing)

    return processor, text_encoder, vision_encoder, interaction, self_experts, cross_experts, hybrid_experts, router, llm

def main(args):
    ## Random Seed
    seed = args.seed
    reproduce(seed)

    ## Device Setting
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    ## Dataset and Dataloader
    train, val = train_dataset(0.01, seed)

if __name__ == "__main__":
    main()