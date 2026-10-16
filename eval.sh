#!/usr/bin/bash

# To use this sh file
# CUDA_VISIBLE_DEVICES=0 ./eval.sh     # GPU 0 only
# CUDA_VISIBLE_DEVICES=0,1 ./eval.sh   # GPUs 0 and 1

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
export PYTHONPATH="$PWD/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}"

for SEED in 42 43 44; do
    PYTHONHASHSEED="$SEED" torchrun --standalone --nproc-per-node=gpu eval.py \
        --model IPMoE \
        --data MME \
        --work-dir evaluation/seed_$SEED \
        --judge exact_matching \
        "$@" || exit $?
done
