#!/usr/bin/bash

for SEED in 42 43 44; do
    torchrun --standalone --nproc-per-node=gpu main.py \
        --seed $SEED \
        --attn-implementation flash_attention_2 \
        --output-dir checkpoint/seed_$SEED
done
