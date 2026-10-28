#!/usr/bin/bash
# H1: test whether each object word's interaction map points at that object's CV-Bench-3D box.
# Measures the frozen-CLIP reference and every arm for each seed, then pools each arm over the seeds measured
# in this invocation and compares it with the reference and with CONTROL, pairing runs by seed.
# The primary result is the full arm's pooled/summary.json; the h1.py docstring defines the test.
# CONTROL (all_map_mean by default) is trained like the full model, but its map is averaged over patches
# before use, so no spatial gradient reaches it; it must be in ARMS to be compared with.
# Choose a subset with environment variables, e.g. SEEDS=42 ARMS="all_map_mean full" ./h1.sh
# FULL_DIR holds the full-model seeds (checkpoint/llama by default; checkpoint/qwen for the Qwen runs) and
# ABLATION_DIR the other arms. run.sh writes the full arm to checkpoint/seed_N, so move it under FULL_DIR first.
# h1.py refuses to pool or compare runs whose language models differ.
# JUDGE selects which VLMEvalKit result file supplies per-question correctness.
# REFERENCE=0 reuses an earlier $OUT/clip instead of measuring it again.
# Extra arguments go to every measurement, e.g. ./h1.sh --limit 20
#
# Only the CLIP encoders are loaded (a few GB). Choose the GPU with CUDA_VISIBLE_DEVICES,
# or CUDA_VISIBLE_DEVICES= ./h1.sh to run on the CPU while training occupies the GPUs.

export PYTHONPATH="$PWD/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}"

trap 'exit 130' INT

SEEDS=${SEEDS:-"42"}
CONTROL=${CONTROL:-all_map_mean}
ARMS=${ARMS:-"$CONTROL full"}
FULL_DIR=${FULL_DIR:-checkpoint/llama}
ABLATION_DIR=${ABLATION_DIR:-checkpoint/ablations}
JUDGE=${JUDGE:-gpt-6-luna}
OUT=${OUT:-evaluation/h1}
REFERENCE=${REFERENCE:-1}

ARGS=("$@")
FAILED=()
# Output directory of each run that succeeded in this invocation, keyed by arm/seed.
declare -A RUNS

# Run h1.py into a directory, first removing its previous results so a failure cannot leave stale ones behind.
run_h1() {
    local dir=$1
    shift
    mkdir -p "$dir"
    rm -f "$dir/items.csv" "$dir/summary.json"
    python h1.py --output-dir "$dir" "$@" 2>&1 | tee "$dir/h1.log"

    return "${PIPESTATUS[0]}"
}

REFERENCE_RUN=()
if [ "$REFERENCE" = 1 ]; then
    echo "=== $(date '+%F %T') reference=clip output=$OUT/clip"
    if run_h1 "$OUT/clip" --reference clip "${ARGS[@]}"; then
        REFERENCE_RUN=(--compare-with "$OUT/clip")
    else
        FAILED+=("clip (see $OUT/clip/h1.log)")
    fi
elif [ -f "$OUT/clip/summary.json" ]; then
    echo "=== Reusing the CLIP reference in $OUT/clip" >&2
    REFERENCE_RUN=(--compare-with "$OUT/clip")
fi

for SEED in $SEEDS; do
    for ARM in $ARMS; do
        # The full arm's directories are named after FULL_DIR (e.g. evaluation/llama/seed_N); the others mirror ABLATION_DIR.
        if [ "$ARM" = full ]; then
            CHECKPOINT=$FULL_DIR/seed_$SEED/checkpoint.pt
            EVAL_DIR=evaluation/$(basename "$FULL_DIR")/seed_$SEED
            WORK_DIR=$OUT/$(basename "$FULL_DIR")/seed_$SEED
        else
            CHECKPOINT=$ABLATION_DIR/$ARM/seed_$SEED/checkpoint.pt
            EVAL_DIR=evaluation/$(basename "$ABLATION_DIR")/$ARM/seed_$SEED
            WORK_DIR=$OUT/$(basename "$ABLATION_DIR")/$ARM/seed_$SEED
        fi

        # Arms still training have no checkpoint yet.
        if [ ! -f "$CHECKPOINT" ]; then
            echo "=== Skipping arm=$ARM seed=$SEED: $CHECKPOINT not found" >&2
            FAILED+=("$ARM/seed_$SEED (no checkpoint)")
            continue
        fi

        EXTRA=()
        PREDICTIONS=$EVAL_DIR/IPMoE/IPMoE_CV-Bench-3D_${JUDGE}_result.xlsx
        if [ -f "$PREDICTIONS" ]; then
            EXTRA+=(--predictions "$PREDICTIONS")
        elif [ -L "$PREDICTIONS" ]; then
            echo "=== $PREDICTIONS is a broken link; skipping the correctness join for arm=$ARM seed=$SEED" >&2
        else
            echo "=== No $JUDGE predictions for arm=$ARM seed=$SEED; skipping the correctness join" >&2
        fi

        echo "=== $(date '+%F %T') arm=$ARM seed=$SEED checkpoint=$CHECKPOINT output=$WORK_DIR"
        # Keep going after a failed run; the log keeps its traceback.
        if run_h1 "$WORK_DIR" --checkpoint "$CHECKPOINT" "${EXTRA[@]}" "${ARGS[@]}"; then
            RUNS[$ARM/$SEED]=$WORK_DIR
        else
            FAILED+=("$ARM/seed_$SEED (see $WORK_DIR/h1.log)")
        fi
    done
done

CONTROL_RUNS=()
for SEED in $SEEDS; do
    if [ -n "${RUNS[$CONTROL/$SEED]}" ]; then
        CONTROL_RUNS+=("${RUNS[$CONTROL/$SEED]}")
    fi
done

# Pool each arm over its seeds; h1.py pairs the control's runs with this arm's by seed.
for ARM in $ARMS; do
    POOL=()
    for SEED in $SEEDS; do
        if [ -n "${RUNS[$ARM/$SEED]}" ]; then
            POOL+=("${RUNS[$ARM/$SEED]}")
        fi
    done
    if [ ${#POOL[@]} -eq 0 ]; then
        continue
    fi

    COMPARE=("${REFERENCE_RUN[@]}")
    if [ "$ARM" != "$CONTROL" ] && [ ${#CONTROL_RUNS[@]} -gt 0 ]; then
        COMPARE+=(--compare-with "${CONTROL_RUNS[@]}")
    fi

    POOLED=$(dirname "${POOL[0]}")/pooled
    echo "=== $(date '+%F %T') arm=$ARM pooled over ${#POOL[@]} seed(s) output=$POOLED"
    if ! run_h1 "$POOLED" --pool "${POOL[@]}" "${COMPARE[@]}"; then
        FAILED+=("$ARM pooled (see $POOLED/h1.log)")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    printf 'Failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
