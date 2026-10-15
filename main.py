import os
import argparse
import torch
import random

import numpy as np
from torch.utils.data import DataLoader

# Dataset
from preprocessing import train_dataset, make_collate_fn, PREPROCESSING_PROTOCOL, SOURCES

# Proposed Method
from model.encoder import encoding
from model.interaction import Interaction
from model.experts import SelfExpert, CrossExpert, HybridExpert
from model.router import MoE
from model.llm import LlamaLLM, Qwen25VLLLM, NoValidTargetsError

from model.ipmoe import IPMoE

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
    processor, text_encoder, vision_encoder = encoding(args.drop, device, args.embed_dim, revision = args.clip_revision)

    ## Interaction Matrix Constructor
    interaction = Interaction(args.interaction_type)

    ## Experts
    self_expert = SelfExpert(None, args.patch_size, 1, args.embed_dim, args.num_heads, args.mlp_ratio, args.drop, args.attn_drop, True)
    cross_expert = CrossExpert(None, args.patch_size, 1, args.embed_dim, args.num_heads, args.mlp_ratio, args.drop, args.attn_drop, False)
    hybrid_expert = HybridExpert(None, args.patch_size, 1, args.embed_dim, args.num_heads, args.mlp_ratio, args.drop, args.attn_drop, False)

    ## Router
    moe = MoE(1, self_expert, cross_expert, hybrid_expert)

    ## LLM
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    if args.llm_type == "llama":
        llm = LlamaLLM(args.llm_size, args.lora_rank, args.attn_implementation, args.gradient_checkpointing, dtype, args.llm_revision)

    elif args.llm_type == "qwen":
        llm = Qwen25VLLLM(args.llm_size, args.lora_rank, args.attn_implementation, args.gradient_checkpointing, dtype, args.llm_revision)

    args.clip_revision = getattr(text_encoder.clip.config, '_commit_hash', None) or args.clip_revision
    args.llm_revision = getattr(llm.model.config, '_commit_hash', None) or args.llm_revision

    return processor, text_encoder, vision_encoder, interaction, moe, llm

def batch_loss(model, batch, device, context_limit, return_stats = False):
    _, question, text_inputs, image_inputs, answer = batch
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}

    return model(question, text_inputs, image_inputs, answer, context_limit, return_stats = return_stats)

def train(model, batch, device, context_limit):
    model.train()

    return batch_loss(model, batch, device, context_limit)

def val(model, batch, device, context_limit):
    model.eval()

    with torch.no_grad():
        return batch_loss(model, batch, device, context_limit, return_stats = True)

def main(args):
    ## Random Seed
    seed = args.seed
    reproduce(seed)

    ## Device Setting
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    ## Dataset and Dataloader
    train_set, val_set = train_dataset(seed = args.data_seed, max_train_samples = args.max_train_samples)

    ## Model Building
    processor, text_encoder, vision_encoder, interaction, moe, llm = build_model(device, args)
    ipmoe = IPMoE(processor, text_encoder, vision_encoder, interaction, moe, llm, args.embed_dim, args.patch_size).to(device)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = True,
                              collate_fn = make_collate_fn(processor), generator = generator)
    val_loader = DataLoader(val_set, batch_size = args.batch_size, shuffle = False,
                            collate_fn = make_collate_fn(processor), generator = torch.Generator().manual_seed(seed))

    ## Optimizer
    optimizer = torch.optim.AdamW((parameter for parameter in ipmoe.parameters() if parameter.requires_grad), lr = args.learning_rate)
    step = 0

    ## Training and Validation
    os.makedirs(args.output_dir, exist_ok = True)
    for epoch in range(args.epochs):
        skipped = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none = True)
            try:
                loss, weights = train(ipmoe, batch, device, args.context_limit)
            except NoValidTargetsError:
                skipped += len(batch[1])
                continue

            loss.backward()
            optimizer.step()
            step += 1
            skipped += len(batch[1]) - weights.size(0)
            print(f'Epoch {epoch + 1}, step {step}, loss {loss.item():.4f}')

        print(f'Skipped training examples: {skipped}')

        ## Validation
        val_loss = 0.0
        val_tokens = 0
        val_skipped = 0
        for batch in val_loader:
            try:
                loss, weights, tokens = val(ipmoe, batch, device, args.context_limit)
            except NoValidTargetsError:
                val_skipped += len(batch[1])
                continue

            val_loss += loss.item() * tokens
            val_tokens += tokens
            val_skipped += len(batch[1]) - weights.size(0)

        mean_loss = val_loss / val_tokens if val_tokens else None
        print(f'Validation token-mean loss: {mean_loss}, skipped validation examples: {val_skipped}')

        ## Save at the End of Each Epoch
        torch.save({
            'model': {name: parameter.detach().cpu() for name, parameter in ipmoe.named_parameters() if parameter.requires_grad},
            'args': vars(args), 'epoch': epoch + 1, 'step': step,
            'protocol': PREPROCESSING_PROTOCOL, 'source': SOURCES,
        }, os.path.join(args.output_dir, 'checkpoint.pt.tmp'))
        os.replace(os.path.join(args.output_dir, 'checkpoint.pt.tmp'), os.path.join(args.output_dir, 'checkpoint.pt'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--data-seed', type = int, default = 42)

    parser.add_argument('--interaction-type', choices = ['bilinear'], default = 'bilinear')

    parser.add_argument('--patch-size', type = int, default = 16)
    parser.add_argument('--embed-dim', type = int, default = 1024)
    parser.add_argument('--num-heads', type = int, default = 16)
    parser.add_argument('--mlp-ratio', type = float, default = 2.0)
    parser.add_argument('--drop', type = float, default = 0.0)
    parser.add_argument('--attn-drop', type = float, default = 0.0)
    parser.add_argument('--llm-type', default = 'llama')
    parser.add_argument('--llm-size', type = int, default = 3)
    parser.add_argument('--lora-rank', type = int, default = 64)
    parser.add_argument('--clip-revision', default = None)
    parser.add_argument('--llm-revision', default = None)
    parser.add_argument('--attn-implementation', default = 'sdpa')

    parser.add_argument('--gradient-checkpointing', action = argparse.BooleanOptionalAction, default = True)
    parser.add_argument('--context-limit', type = int, default = 4096)
    parser.add_argument('--batch-size', type = int, default = 1)
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--learning-rate', type = float, default = 1e-4)
    parser.add_argument('--max-train-samples', type = int, default = None)
    parser.add_argument('--output-dir', default = 'checkpoint')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main(parse_args())
