#!/usr/bin/bash
# Train the selected prefix variant once per seed; full remains the default.
# Example: SEEDS="42 43 44" LLM_TYPE=llama ./run.sh
# Example: ABLATION=spectral_only SEEDS="42" ./run.sh
# --ablation also selects a separate default checkpoint directory.
# CHECKPOINT_DIR separates these checkpoints from historical architectures.

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
# DataLoader workers fork after the tokenizer is loaded; keep its threads in the main process only.
export TOKENIZERS_PARALLELISM=false

trap 'exit 130' INT
trap 'exit 143' TERM

SEEDS=${SEEDS:-"42 43 44"}
LLM_TYPE=${LLM_TYPE:-llama}
ABLATION=${ABLATION:-full}
ARGS=("$@")
for ((INDEX=0; INDEX<${#ARGS[@]}; INDEX++)); do
    case "${ARGS[INDEX]}" in
        --ablation=*) ABLATION=${ARGS[INDEX]#*=} ;;
        --ablation)
            if ((INDEX + 1 >= ${#ARGS[@]})); then
                echo "--ablation requires a value" >&2
                exit 1
            fi
            ABLATION=${ARGS[INDEX+1]} ;;
    esac
done
case "$ABLATION" in
    full|spectral_only|evidence_only|random_selection|direct_image) ;;
    *) echo "Unknown ablation: $ABLATION" >&2; exit 1 ;;
esac
DEFAULT_DIRECTORY=checkpoint/svd_evidence_prefix
if [ "$ABLATION" != full ]; then
    DEFAULT_DIRECTORY=$DEFAULT_DIRECTORY/$ABLATION
fi
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$DEFAULT_DIRECTORY}

FAILED=()
for SEED in $SEEDS; do
    OUTPUT_DIR=$CHECKPOINT_DIR/seed_$SEED
    mkdir -p "$OUTPUT_DIR"

    echo "=== $(date '+%F %T') architecture=svd_evidence_prefix ablation=$ABLATION seed=$SEED output=$OUTPUT_DIR"
    PYTHONHASHSEED="$SEED" torchrun --standalone --nproc-per-node=gpu main.py \
        --seed "$SEED" \
        --ablation "$ABLATION" \
        --attn-implementation flash_attention_2 \
        --llm-type "$LLM_TYPE" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 6 "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"

    STATUS=${PIPESTATUS[0]}
    if [ "$STATUS" -eq 130 ] || [ "$STATUS" -eq 143 ]; then
        exit "$STATUS"
    fi
    # Keep going after an ordinary failure; the log keeps its traceback.
    if [ "$STATUS" -ne 0 ]; then
        FAILED+=("seed_$SEED")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed runs: ${FAILED[*]}" >&2
    exit 1
fi
