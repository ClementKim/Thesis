#!/usr/bin/bash
# Train the full model for each seed into checkpoint/LLM_TYPE/seed_N (checkpoint.pt and train.log), the reference that
# eval.sh evaluates and test.sh compares the other arms with. Choose a subset with environment variables,
# e.g. SEEDS=42 LLM_TYPE=qwen ./run.sh
# A seed that already has a checkpoint is skipped; RETRAIN=1 trains it again in place. ARM=full STEPS=train ./test.sh
# trains the same model but keeps the current run until the new one has finished.
# Every train.log begins with a line naming the configuration, followed by main.py's full argument list.

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
# DataLoader workers fork after the tokenizer is loaded; keep its threads in the main process only.
export TOKENIZERS_PARALLELISM=false

SEEDS=${SEEDS:-"42 43 44"}
# LLM_TYPE=${LLM_TYPE:-llama}
LLM_TYPE=${LLM_TYPE:-qwen}
RETRAIN=${RETRAIN:-0}
# Per GPU.
BATCH_SIZE=6

# As in test.sh: untracked Python or shell files are code too.
CODE=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] || [ -n "$(git ls-files --others --exclude-standard -- '*.py' '*.sh' 2>/dev/null)" ]; then
    CODE="$CODE+uncommitted"
fi

FAILED=()
for SEED in $SEEDS; do
    # eval.sh and test.sh read this layout; eval.py run directly needs IPMOE_CHECKPOINT for it.
    OUTPUT_DIR=checkpoint/$LLM_TYPE/seed_$SEED
    if [ -f "$OUTPUT_DIR/checkpoint.pt" ] && [ "$RETRAIN" != 1 ]; then
        echo "=== Skipping seed=$SEED: $OUTPUT_DIR/checkpoint.pt exists (RETRAIN=1 trains it again)"
        continue
    fi
    mkdir -p "$OUTPUT_DIR"

    echo "=== $(date '+%F %T') train arm=full llm=$LLM_TYPE code=$CODE gpus=${CUDA_VISIBLE_DEVICES:-all} seed=$SEED" \
         "batch_size=$BATCH_SIZE/GPU output=$OUTPUT_DIR" | tee "$OUTPUT_DIR/train.log"
    PYTHONHASHSEED="$SEED" torchrun --standalone --nproc-per-node=gpu main.py \
        --seed $SEED \
        --attn-implementation flash_attention_2 \
        --llm-type "$LLM_TYPE" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" 2>&1 | tee -a "$OUTPUT_DIR/train.log"

    # Keep going after a failed run; the log keeps its traceback.
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        FAILED+=("seed_$SEED")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed runs: ${FAILED[*]}" >&2
    exit 1
fi
