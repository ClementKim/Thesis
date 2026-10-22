import os
import torch
import random
import argparse

import numpy as np
import torch.distributed as dist

from tqdm import tqdm
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import get_cosine_schedule_with_warmup

from ablation import ABLATIONS, configure_ablation

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
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True)

        # Allow TF32 on Ampere+ GPUs for faster matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    except Exception as e:
        raise RuntimeError(f"Failed to set random seed: {e}")

def build_model(device, args):
    ablation = configure_ablation(args)

    if args.embed_dim <= 0 or args.num_heads <= 0 or args.embed_dim % args.num_heads:
        raise ValueError('embed_dim must be positive and divisible by positive num_heads')
    
    if args.patch_size <= 0:
        raise ValueError('patch_size must be positive')

    ## CLIP
    processor, text_encoder, vision_encoder = encoding(args.drop, device, args.embed_dim)

    vision = text_encoder.clip.config.vision_config
    if vision.image_size % vision.patch_size or vision.image_size // vision.patch_size != 24:
        raise ValueError('The router requires a 24 by 24 CLIP patch grid')

    ## Interaction Matrix Constructor
    interaction = Interaction(args.interaction_type)

    ## Experts
    self_expert = SelfExpert(args.patch_size, 1, args.embed_dim, args.num_heads, args.mlp_ratio, args.drop, args.attn_drop, False)
    cross_expert = CrossExpert(args.patch_size, 1, args.embed_dim, args.num_heads, args.mlp_ratio, args.drop, args.attn_drop, False)
    hybrid_expert = HybridExpert(args.patch_size, 1, args.embed_dim, args.num_heads, args.mlp_ratio, args.drop, args.attn_drop, False)

    ## Router
    moe = MoE(1, self_expert, cross_expert, hybrid_expert, router_mode = ablation['router_mode'])

    ## LLM
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    if args.llm_type == "llama":
        llm = LlamaLLM(args.llm_size, args.lora_rank, args.attn_implementation, args.gradient_checkpointing, dtype)

    elif args.llm_type == "qwen":
        llm = Qwen25VLLLM(args.llm_size, args.lora_rank, args.attn_implementation, args.gradient_checkpointing, dtype)

    return processor, text_encoder, vision_encoder, interaction, moe, llm

def batch_loss(model, batch, device, context_limit, return_stats = False, allow_empty = False):
    _, question, text_inputs, image_inputs, answer = batch
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}

    return model(question, text_inputs, image_inputs, answer, context_limit, return_stats = return_stats, allow_empty = allow_empty)

def train(model, batch, device, context_limit, return_stats = False, allow_empty = False):
    model.train()

    return batch_loss(model, batch, device, context_limit, return_stats, allow_empty)

def val(model, batch, device, context_limit):
    model.eval()

    with torch.no_grad():
        return batch_loss(model, batch, device, context_limit, return_stats = True)

def setup_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)

    else:
        device = torch.device('mps' if world_size == 1 and torch.backends.mps.is_available() else 'cpu')

    if world_size > 1:
        # Allow time for rank 0 to prepare the shared dataset cache on the first run.
        dist.init_process_group(backend = 'nccl' if device.type == 'cuda' else 'gloo',
                                device_id = device if device.type == 'cuda' else None, timeout = timedelta(hours = 2))

    return device, rank, world_size

def reduce_totals(values, device):
    if not dist.is_initialized():
        return values
    
    totals = torch.tensor(values, dtype = torch.float64, device = device)
    dist.all_reduce(totals, op = dist.ReduceOp.SUM)

    return totals.tolist()

def train_step(model, batch, optimizer, scheduler, device, context_limit, max_grad_norm):
    optimizer.zero_grad(set_to_none = True)
    distributed = isinstance(model, DDP)
    try:
        loss, weights, tokens = train(model, batch, device, context_limit, return_stats = True, allow_empty = distributed)

    except NoValidTargetsError:
        return None, 0, None

    totals = [loss.detach().item() * tokens, tokens]
    if distributed:
        totals = reduce_totals(totals, device)

    loss_sum, total_tokens = totals

    if total_tokens == 0:
        # Complete DDP's backward bookkeeping without advancing AdamW or weight decay.
        loss.backward()

        return None, 0, None

    world_size = dist.get_world_size() if distributed else 1

    # DDP averages gradients across ranks; compensate for unequal token counts.
    (loss * (world_size * tokens / total_tokens)).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()

    return loss_sum / total_tokens, weights.size(0), grad_norm.item()

def evaluate(model, loader, device, context_limit):
    """Evaluate disjoint shards without per-forward DDP communication."""
    if isinstance(model, DDP):
        model = model.module

    loss_sum, tokens, skipped, seen = 0.0, 0, 0, 0
    for batch in tqdm(loader):
        seen += len(batch[1])

        try:
            loss, weights, batch_tokens = val(model, batch, device, context_limit)

        except NoValidTargetsError:
            skipped += len(batch[1])
            continue

        loss_sum += loss.item() * batch_tokens
        tokens += batch_tokens

        skipped += len(batch[1]) - weights.size(0)

    loss_sum, tokens, skipped, seen = reduce_totals([loss_sum, tokens, skipped, seen], device)
    return loss_sum / tokens if tokens else None, int(skipped), int(seen)

def run_training(args, device, rank, world_size):
    seed = args.seed

    distributed = world_size > 1
    if rank == 0:
        print(f'Using device: {device}, processes: {world_size}, global batch size: {args.batch_size * world_size}')

    ## Dataset and Dataloader
    # Only rank 0 builds the cache; the other ranks then read the completed files.
    if distributed and rank != 0:
        dist.barrier()

    train_set, val_set = train_dataset(seed = args.data_seed, max_train_samples = args.max_train_samples)

    if distributed and rank == 0:
        dist.barrier()

    ## Model Building
    processor, text_encoder, vision_encoder, interaction, moe, llm = build_model(device, args)
    ipmoe = IPMoE(processor, text_encoder, vision_encoder, interaction, moe, llm,
                  args.embed_dim, args.patch_size, args.num_aux_tokens, ablation = args.ablation_config).to(device)
    model = DDP(ipmoe, device_ids = [device.index] if device.type == 'cuda' else None, broadcast_buffers = False) if distributed else ipmoe
    
    # Start from identical weights, then use independent dropout streams per rank.
    torch.manual_seed(seed + rank)

    generator = torch.Generator().manual_seed(seed)
    train_sampler = DistributedSampler(train_set, num_replicas = world_size, rank = rank, seed = seed) if distributed else None
    # Workers decode images and run the CLIP processor while the GPU trains on the previous batch.
    loader_options = {'num_workers': args.num_workers, 'persistent_workers': args.num_workers > 0}
    train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle = train_sampler is None, sampler = train_sampler, collate_fn = make_collate_fn(processor), generator = generator, **loader_options)
    
    # No duplicated validation examples, even when the split is smaller than world_size.
    val_shard = Subset(val_set, range(rank, len(val_set), world_size))
    val_loader = DataLoader(val_shard, batch_size = args.batch_size, shuffle = False, collate_fn = make_collate_fn(processor), generator = torch.Generator().manual_seed(seed), **loader_options)

    ## Optimizer
    step = 0
    optimizer = torch.optim.AdamW((parameter for parameter in ipmoe.parameters() if parameter.requires_grad), lr = args.learning_rate)

    # Linear warmup, then cosine decay to zero over the planned optimizer steps.
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(total_steps * args.warmup_ratio), total_steps)

    ## Training and Validation
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok = True)
    
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        seen = 0
        skipped = 0
        epoch_start_step = step
        for batch in tqdm(train_loader):
            loss, retained, grad_norm = train_step(model, batch, optimizer, scheduler, device, args.context_limit, args.max_grad_norm)
            seen += len(batch[1])
            skipped += len(batch[1]) - retained
            if loss is None:
                continue

            step += 1
            if rank == 0 and not (step % 1000):
                print(f'Epoch {epoch + 1}, step {step}, loss {loss:.4f}, lr {scheduler.get_last_lr()[0]:.2e}, grad norm {grad_norm:.2f}')

        epoch_steps = step - epoch_start_step
        if epoch_steps == 0:
            raise RuntimeError('No optimizer steps this epoch; refusing to save a new checkpoint')

        skipped, seen = (int(value) for value in reduce_totals([skipped, seen], device))
        if rank == 0:
            print(f'Skipped training examples: {skipped}, sampler padding: {seen - len(train_set)}')

        ## Validation
        mean_loss, val_skipped, val_seen = evaluate(ipmoe, val_loader, device, args.context_limit)
        if rank == 0:
            print(f'Validation token-mean loss: {mean_loss}, skipped validation examples: {val_skipped}')

        ## Save at the End of Each Epoch
        if rank == 0:
            torch.save({
                'model': {name: parameter.detach().cpu() for name, parameter in ipmoe.named_parameters() if parameter.requires_grad},
                'args': vars(args), 'epoch': epoch + 1, 'step': step,
                'protocol': PREPROCESSING_PROTOCOL, 'source': SOURCES,
                'world_size': world_size, 'global_batch_size': args.batch_size * world_size,
                'optimizer_steps_this_epoch': epoch_steps,
                'data_counts': {
                    'train_split': len(train_set), 'val_split': len(val_set),
                    'train_seen_this_epoch': seen, 'train_sampler_padding': seen - len(train_set),
                    'train_budget_skipped': skipped, 'val_budget_skipped': val_skipped,
                    'train_retained_this_epoch': seen - skipped,
                    'val_retained': val_seen - val_skipped,
                },
                'val_loss': mean_loss,
            }, os.path.join(args.output_dir, 'checkpoint.pt.tmp'))
            os.replace(os.path.join(args.output_dir, 'checkpoint.pt.tmp'), os.path.join(args.output_dir, 'checkpoint.pt'))
    
        if distributed:
            dist.barrier()

def main(args):
    # Configure determinism before CUDA or the process group is initialized.
    reproduce(args.seed)
    device, rank, world_size = setup_distributed()
    run_training(args, device, rank, world_size)

    # Tear down only after a clean run. On a failing rank, destroy_process_group() blocks while the other
    # ranks wait inside a collective, which hides the traceback until the NCCL timeout. Exiting instead
    # lets torchrun stop the job and report the error immediately.
    if dist.is_initialized():
        dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--data-seed', type = int, default = 42)

    parser.add_argument('--interaction-type', choices = ['bilinear'], default = 'bilinear')
    parser.add_argument('--ablation', choices = list(ABLATIONS), default = 'full')

    parser.add_argument('--patch-size', type = int, default = 16)
    parser.add_argument('--embed-dim', type = int, default = 1024)
    parser.add_argument('--num-heads', type = int, default = 16)
    parser.add_argument('--num-aux-tokens', type = int, default = 32)
    parser.add_argument('--mlp-ratio', type = float, default = 2.0)
    parser.add_argument('--drop', type = float, default = 0.0)
    parser.add_argument('--attn-drop', type = float, default = 0.0)
    parser.add_argument('--llm-type', choices = ['llama', 'qwen'], default = 'llama')
    parser.add_argument('--llm-size', type = int, default = 3)
    parser.add_argument('--lora-rank', type = int, default = 64)
    parser.add_argument('--attn-implementation', default = 'flash_attention_2')

    parser.add_argument('--gradient-checkpointing', action = argparse.BooleanOptionalAction, default = True)
    parser.add_argument('--context-limit', type = int, default = 4096)
    parser.add_argument('--batch-size', type = int, default = 1)
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--learning-rate', type = float, default = 1e-4)
    parser.add_argument('--warmup-ratio', type = float, default = 0.03)
    parser.add_argument('--max-grad-norm', type = float, default = 1.0)
    parser.add_argument('--max-train-samples', type = int, default = None)
    parser.add_argument('--num-workers', type = int, default = 4)
    parser.add_argument('--output-dir', default = None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = 'checkpoint' if args.ablation == 'full' else os.path.join('checkpoint', 'ablations', args.ablation, f'seed_{args.seed}')

    return args

if __name__ == "__main__":
    main(parse_args())
