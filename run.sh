#!/usr/bin/bash
# Train every ablation arm in ablation.py for each seed, finishing all arms of one seed before the next.
# Choose a subset with environment variables, e.g. SEEDS=42 ABLATIONS="full uniform" ./run.sh

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1

ALL_ABLATIONS=$(python -c 'from ablation import ABLATIONS; print(*ABLATIONS)') || exit 1
SEEDS=${SEEDS:-"42"}
ABLATIONS=${ABLATIONS:-$ALL_ABLATIONS}

# Reject typos now rather than after days of other runs.
for ABLATION in $ABLATIONS; do
    if [[ " $ALL_ABLATIONS " != *" $ABLATION "* ]]; then
        echo "Unknown ablation '$ABLATION'; choose from: $ALL_ABLATIONS" >&2
        exit 1
    fi
done

FAILED=()
for SEED in $SEEDS; do
    for ABLATION in $ABLATIONS; do
        # eval.py loads checkpoint/seed_SEED by default; other arms need IPMOE_CHECKPOINT.
        if [ "$ABLATION" = full ]; then
            OUTPUT_DIR=checkpoint/seed_$SEED
        else
            OUTPUT_DIR=checkpoint/ablations/$ABLATION/seed_$SEED
        fi
        mkdir -p "$OUTPUT_DIR"

        echo "=== $(date '+%F %T') ablation=$ABLATION seed=$SEED output=$OUTPUT_DIR"
        PYTHONHASHSEED="$SEED" torchrun --standalone --nproc-per-node=gpu main.py \
            --seed $SEED \
            --ablation $ABLATION \
            --attn-implementation flash_attention_2 \
            --output-dir $OUTPUT_DIR \
            --batch-size 6 2>&1 | tee "$OUTPUT_DIR/train.log"

        # Keep going after a failed run; the log keeps its traceback.
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            FAILED+=("$ABLATION/seed_$SEED")
        fi
    done
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed runs: ${FAILED[*]}" >&2
    exit 1
fi
