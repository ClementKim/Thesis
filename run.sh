#!/usr/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1

for SEED in 42 43 44; do
    PYTHONHASHSEED="$SEED" torchrun --standalone --nproc-per-node=gpu main.py \
        --seed $SEED \
        --attn-implementation flash_attention_2 \
        --output-dir checkpoint/seed_$SEED \
        --batch-size 4
done
