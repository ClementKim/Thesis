#!/usr/bin/bash
# Train the model for each seed. Choose a subset with environment variables, e.g. SEEDS=42 ./run.sh

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
# DataLoader workers fork after the tokenizer is loaded; keep its threads in the main process only.
export TOKENIZERS_PARALLELISM=false

SEEDS=${SEEDS:-"42 43 44"}
LLM_TYPE=${LLM_TYPE:-llama}
# LLM_TYPE=${LLM_TYPE:-qwen}

FAILED=()
for SEED in $SEEDS; do
    # eval.py loads checkpoint/seed_SEED by default; other locations need IPMOE_CHECKPOINT.
    OUTPUT_DIR=checkpoint/seed_$SEED
    mkdir -p "$OUTPUT_DIR"

    echo "=== $(date '+%F %T') seed=$SEED output=$OUTPUT_DIR"
    PYTHONHASHSEED="$SEED" torchrun --standalone --nproc-per-node=gpu main.py \
        --seed $SEED \
        --attn-implementation flash_attention_2 \
        --llm-type "$LLM_TYPE" \
        --output-dir $OUTPUT_DIR \
        --batch-size 6 2>&1 | tee "$OUTPUT_DIR/train.log"

    # Keep going after a failed run; the log keeps its traceback.
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        FAILED+=("seed_$SEED")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed runs: ${FAILED[*]}" >&2
    exit 1
fi
