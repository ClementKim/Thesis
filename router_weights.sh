#!/usr/bin/bash
# RQ2: record the expert weights each learned-router arm assigns to every benchmark question (see router_weights.py).
# Each run writes one CSV row per question, and a log next to it that ends with the per-dataset summary.
# Choose a subset with environment variables, e.g. SEEDS="42 43 44" ARMS="full all_map_mean" ./router_weights.sh
# ARMS defaults to every arm in ablation.py with a learned router over all three experts; the others have fixed weights.
# FULL_DIR holds the full-model seeds (checkpoint/llama by default; checkpoint/qwen for the Qwen runs) and
# ABLATION_DIR the other arms, as in h1.sh.
# Extra arguments go to every run, e.g. ./router_weights.sh --data MMStar POPE
#
# The whole model is loaded on one device; choose the GPU with CUDA_VISIBLE_DEVICES.

export PYTHONPATH="$PWD/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}"

trap 'exit 130' INT

# image_only feeds no expert output to the LLM, so its router produces no weights either.
LEARNED_ARMS=$(python -c '
from ablation import ABLATIONS
print(*(name for name, config in ABLATIONS.items() if config["router_mode"] == "learned"
        and config.get("expert_set", "all") == "all" and config.get("prefix", "full") == "full"))
') || exit 1
SEEDS=${SEEDS:-"42"}
ARMS=${ARMS:-$LEARNED_ARMS}
FULL_DIR=${FULL_DIR:-checkpoint/llama}
ABLATION_DIR=${ABLATION_DIR:-checkpoint/ablations}
OUT=${OUT:-evaluation/router}

if [ -z "$ARMS" ]; then
    echo "ablation.py has no arm with a learned router over all three experts" >&2
    exit 1
fi

# Reject arms without input-dependent weights now rather than after loading the model.
for ARM in $ARMS; do
    if [[ " $LEARNED_ARMS " != *" $ARM "* ]]; then
        echo "Arm '$ARM' is not in ablation.py or has no learned router over all three experts; choose from: $LEARNED_ARMS" >&2
        exit 1
    fi
done

# Each run sets its own checkpoint and output.
for ARG in "$@"; do
    case "$ARG" in
        --checkpoint|--checkpoint=*|--output|--output=*)
            echo "Set SEEDS, ARMS, FULL_DIR, ABLATION_DIR or OUT instead of passing $ARG" >&2
            exit 1 ;;
    esac
done

FAILED=()
for SEED in $SEEDS; do
    for ARM in $ARMS; do
        # Outputs mirror the checkpoint layout, e.g. evaluation/router/llama/seed_42.csv for the full arm.
        if [ "$ARM" = full ]; then
            CHECKPOINT=$FULL_DIR/seed_$SEED/checkpoint.pt
            OUTPUT=$OUT/$(basename "$FULL_DIR")/seed_$SEED.csv
        else
            CHECKPOINT=$ABLATION_DIR/$ARM/seed_$SEED/checkpoint.pt
            OUTPUT=$OUT/$(basename "$ABLATION_DIR")/$ARM/seed_$SEED.csv
        fi
        LOG=${OUTPUT%.csv}.log

        # Arms still training have no checkpoint yet.
        if [ ! -f "$CHECKPOINT" ]; then
            echo "=== Skipping arm=$ARM seed=$SEED: $CHECKPOINT not found" >&2
            FAILED+=("$ARM/seed_$SEED (no checkpoint)")
            continue
        fi

        # Remove the previous results first so a failure cannot leave stale ones behind.
        mkdir -p "$(dirname "$OUTPUT")"
        rm -f "$OUTPUT"
        echo "=== $(date '+%F %T') arm=$ARM seed=$SEED checkpoint=$CHECKPOINT output=$OUTPUT"
        python router_weights.py --checkpoint "$CHECKPOINT" --output "$OUTPUT" "$@" 2>&1 | tee "$LOG"

        # Keep going after a failed run; the log keeps its traceback.
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            FAILED+=("$ARM/seed_$SEED (see $LOG)")
        fi
    done
done

if [ ${#FAILED[@]} -gt 0 ]; then
    printf 'Failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
