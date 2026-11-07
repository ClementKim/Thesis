#!/usr/bin/env bash
# Run the full control and four ablations sequentially on the same training setup.
# Defaults: one seed, batch size 1, separate checkpoint/ablations/<variant>/seed_<seed> directories.
# Examples:
#   CUDA_VISIBLE_DEVICES=0 ./ablation.sh --max-train-samples 10000
#   ABLATIONS="spectral_only evidence_only" SEEDS="42 43" ./ablation.sh
#   MODE=eval DATASETS="POPE MMStar" ./ablation.sh
# Extra arguments go to run.sh (train) or eval.sh (eval). Use the same training arguments for every variant.

if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    cat <<'EOF'
Usage: ./ablation.sh [arguments for main.py]
       MODE=eval ./ablation.sh [arguments for eval.py]

ABLATIONS       Variants to run (default: full spectral_only evidence_only random_selection direct_image)
SEEDS           Shared seeds (default: 42)
MODE            train or eval (default: train)
BATCH_SIZE      Training batch size (default: 1; --batch-size can override it)
CHECKPOINT_ROOT Checkpoint root (default: checkpoint/ablations)
OUT_ROOT        Evaluation root (default: evaluation/ablations)
DATASETS        Evaluation datasets (default: POPE)
JUDGE           Evaluation judge (default: exact_matching)

Each variant uses <root>/<variant>/seed_<seed>. Training and evaluation must use
the same CHECKPOINT_ROOT and SEEDS. LLM_TYPE and CUDA_VISIBLE_DEVICES pass through.
EOF
    exit 0
fi

# Stop the sweep on interruption, including when torchrun exits normally on Ctrl-C.
trap 'exit 130' INT
trap 'exit 143' TERM
set -f

MODE=${MODE:-train}
ABLATIONS=${ABLATIONS-"full spectral_only evidence_only random_selection direct_image"}
SEEDS=${SEEDS-"42"}
BATCH_SIZE=${BATCH_SIZE:-1}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-checkpoint/ablations}
OUT_ROOT=${OUT_ROOT:-evaluation/ablations}
DATASETS=${DATASETS:-POPE}
JUDGE=${JUDGE:-exact_matching}

case "$MODE" in
    train|eval) ;;
    *) echo "MODE must be train or eval" >&2; exit 1 ;;
esac
if [ -n "${CHECKPOINT_DIR:-}" ] || [ -n "${OUT:-}" ]; then
    echo "Set CHECKPOINT_ROOT and OUT_ROOT instead of CHECKPOINT_DIR or OUT" >&2
    exit 1
fi
for ARG in "$@"; do
    OPTION=${ARG%%=*}
    if [[ "$OPTION" == --?* ]]; then
        # argparse also accepts unambiguous prefixes, such as --ablat or --out.
        for OWNED in --ablation --seed --output-dir --work-dir; do
            if [[ "$OWNED" == "$OPTION"* ]]; then
                echo "Use ABLATIONS, SEEDS, CHECKPOINT_ROOT or OUT_ROOT instead of $ARG" >&2
                exit 1
            fi
        done
        if [ "$MODE" = eval ] && [[ --model == "$OPTION"* ]]; then
            echo "Evaluation uses the IPMoE checkpoint; do not pass $ARG" >&2
            exit 1
        fi
    fi
done

# Validate the entire sweep before starting the first expensive run.
VARIANTS=()
for VARIANT in $ABLATIONS; do
    case "$VARIANT" in
        full|spectral_only|evidence_only|random_selection|direct_image) ;;
        *) echo "Unknown ablation: $VARIANT" >&2; exit 1 ;;
    esac
    for PREVIOUS in "${VARIANTS[@]}"; do
        if [ "$PREVIOUS" = "$VARIANT" ]; then
            echo "Duplicate ablation: $VARIANT" >&2
            exit 1
        fi
    done
    VARIANTS+=("$VARIANT")
done
if [ ${#VARIANTS[@]} -eq 0 ] || [[ ! "$SEEDS" =~ [^[:space:]] ]]; then
    echo "ABLATIONS and SEEDS must not be empty" >&2
    exit 1
fi
for SEED in $SEEDS; do
    if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
        echo "SEEDS must contain nonnegative integers: $SEED" >&2
        exit 1
    fi
done

SCRIPT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
cd "$SCRIPT_DIRECTORY" || exit 1

FAILED=()
for VARIANT in "${VARIANTS[@]}"; do
    # One seed per child keeps interruption control in this wrapper.
    for SEED in $SEEDS; do
        echo "=== $(date '+%F %T') mode=$MODE ablation=$VARIANT seed=$SEED"
        if [ "$MODE" = train ]; then
            ABLATION="$VARIANT" SEEDS="$SEED" CHECKPOINT_DIR="$CHECKPOINT_ROOT/$VARIANT" \
                bash "$SCRIPT_DIRECTORY/run.sh" --batch-size "$BATCH_SIZE" "$@"
        else
            ABLATION="$VARIANT" SEEDS="$SEED" CHECKPOINT_DIR="$CHECKPOINT_ROOT/$VARIANT" \
                OUT="$OUT_ROOT/$VARIANT" DATASETS="$DATASETS" JUDGE="$JUDGE" \
                bash "$SCRIPT_DIRECTORY/eval.sh" "$@"
        fi
        STATUS=$?
        if [ "$STATUS" -eq 130 ] || [ "$STATUS" -eq 143 ]; then
            exit "$STATUS"
        fi
        if [ "$STATUS" -ne 0 ]; then
            FAILED+=("$VARIANT seed=$SEED ($MODE)")
        fi
    done
done

if [ ${#FAILED[@]} -gt 0 ]; then
    printf 'Failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
