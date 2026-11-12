#!/usr/bin/bash
# RQ2: record the expert weights the router assigns to every benchmark question (see router_weights.py).
# Each run writes one CSV row per question, and a log next to it that ends with the per-dataset summary.
# Choose a subset with environment variables, e.g. SEEDS="42 43 44" ./router_weights.sh
# FULL_DIR holds the model seeds (checkpoint/llama by default; checkpoint/qwen for the Qwen runs).
# Extra arguments go to every run, e.g. ./router_weights.sh --data MMStar POPE
#
# The whole model is loaded on one device; choose the GPU with CUDA_VISIBLE_DEVICES.

export PYTHONPATH="$PWD/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}"

trap 'exit 130' INT

SEEDS=${SEEDS:-"42"}
FULL_DIR=${FULL_DIR:-checkpoint/llama}
OUT=${OUT:-evaluation/router}

# Each run sets its own checkpoint and output.
for ARG in "$@"; do
    case "$ARG" in
        --checkpoint|--checkpoint=*|--output|--output=*)
            echo "Set SEEDS, FULL_DIR or OUT instead of passing $ARG" >&2
            exit 1 ;;
    esac
done

FAILED=()
for SEED in $SEEDS; do
    # Outputs mirror the checkpoint layout, e.g. evaluation/router/llama/seed_42.csv.
    CHECKPOINT=$FULL_DIR/seed_$SEED/checkpoint.pt
    OUTPUT=$OUT/$(basename "$FULL_DIR")/seed_$SEED.csv
    LOG=${OUTPUT%.csv}.log

    # Seeds still training have no checkpoint yet.
    if [ ! -f "$CHECKPOINT" ]; then
        echo "=== Skipping seed=$SEED: $CHECKPOINT not found" >&2
        FAILED+=("seed_$SEED (no checkpoint)")
        continue
    fi

    # Remove the previous results first so a failure cannot leave stale ones behind.
    mkdir -p "$(dirname "$OUTPUT")"
    rm -f "$OUTPUT"
    echo "=== $(date '+%F %T') seed=$SEED checkpoint=$CHECKPOINT output=$OUTPUT"
    python router_weights.py --checkpoint "$CHECKPOINT" --output "$OUTPUT" "$@" 2>&1 | tee "$LOG"

    # Keep going after a failed run; the log keeps its traceback.
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        FAILED+=("seed_$SEED (see $LOG)")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    printf 'Failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
