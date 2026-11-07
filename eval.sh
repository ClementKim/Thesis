#!/usr/bin/bash
# Evaluate one SVD evidence-prefix checkpoint per seed.
# Example: SEEDS="42 43 44" DATASETS="MMStar POPE" ./eval.sh
# Example: ABLATION=spectral_only SEEDS="42" ./eval.sh
# ABLATION selects default paths; the checkpoint determines the actual model variant.
# CHECKPOINT_DIR and OUT default to the svd_evidence_prefix training/results directories.
# JUDGE=exact_matching avoids an LLM judge; other judges need their API credentials.
# Extra arguments go to eval.py, e.g. ./eval.sh --judge-api-nproc 8
# Inference uses the decoder's context capacity, independent of the training cap.
# Set IPMOE_CONTEXT_LIMIT=16384 to impose a smaller total token budget if needed.
#
# Choose GPUs with CUDA_VISIBLE_DEVICES:
# CUDA_VISIBLE_DEVICES=0 ./eval.sh     # GPU 0 only
# CUDA_VISIBLE_DEVICES=0,1 ./eval.sh   # GPUs 0 and 1

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
export PYTHONPATH="$PWD/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}"

# torchrun exits normally on Ctrl-C, so without this the loop would move on to the next run.
trap 'exit 130' INT
trap 'exit 143' TERM

SEEDS=${SEEDS:-"42"}
DATASETS=${DATASETS:-"NaturalBenchDataset CV-Bench-2D CV-Bench-3D MMBench_DEV_EN_V11 MMStar POPE SEEDBench_IMG"}
JUDGE=${JUDGE:-gpt-6-luna}
ABLATION=${ABLATION:-full}
case "$ABLATION" in
    full|spectral_only|evidence_only|random_selection|direct_image) ;;
    *) echo "Unknown ablation: $ABLATION" >&2; exit 1 ;;
esac
DIRECTORY=svd_evidence_prefix
if [ "$ABLATION" != full ]; then
    DIRECTORY=$DIRECTORY/$ABLATION
fi
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoint/$DIRECTORY}
OUT=${OUT:-evaluation/$DIRECTORY}

# Each scoring group below sets its own datasets and judge.
for ARG in "$@"; do
    case "$ARG" in
        --data|--data=*|--judge|--judge=*)
            echo "Set DATASETS or JUDGE instead of passing $ARG" >&2
            exit 1 ;;
        --ablation|--ablation=*)
            echo "Set ABLATION instead of passing $ARG; evaluation loads the checkpoint's variant" >&2
            exit 1 ;;
    esac
done

# As in IP-MoE's evaluation/run.py: these score with their own answer extraction and never use a judge,
# so they run as a separate exact_matching group instead of being recorded as judge-scored.
JUDGE_IGNORING="NaturalBenchDataset VisuLogic"
JUDGE_DATASETS=()
EXACT_DATASETS=()
for DATASET in $DATASETS; do
    if [ "$JUDGE" != exact_matching ] && [[ " $JUDGE_IGNORING " == *" $DATASET "* ]]; then
        EXACT_DATASETS+=("$DATASET")
    else
        JUDGE_DATASETS+=("$DATASET")
    fi
done

# One eval.py run per group, since VLMEvalKit takes a single --judge. Entries: <judge> <dataset>...
PROTOCOLS=()
if [ ${#JUDGE_DATASETS[@]} -gt 0 ]; then
    PROTOCOLS+=("$JUDGE ${JUDGE_DATASETS[*]}")
fi
if [ ${#EXACT_DATASETS[@]} -gt 0 ]; then
    PROTOCOLS+=("exact_matching ${EXACT_DATASETS[*]}")
fi
if [ ${#PROTOCOLS[@]} -eq 0 ]; then
    echo "DATASETS is empty" >&2
    exit 1
fi

FAILED=()
for SEED in $SEEDS; do
    CHECKPOINT=$CHECKPOINT_DIR/seed_$SEED/checkpoint.pt
    WORK_DIR=$OUT/seed_$SEED
    if [ ! -f "$CHECKPOINT" ]; then
        echo "=== Skipping seed=$SEED: $CHECKPOINT not found" >&2
        FAILED+=("seed_$SEED (no checkpoint)")
        continue
    fi
    mkdir -p "$WORK_DIR"
    : > "$WORK_DIR/eval.log"

    for PROTOCOL in "${PROTOCOLS[@]}"; do
        read -r RUN_JUDGE RUN_DATASETS <<< "$PROTOCOL"
        echo "=== $(date '+%F %T') architecture=svd_evidence_prefix ablation=$ABLATION seed=$SEED judge=$RUN_JUDGE data=[$RUN_DATASETS] checkpoint=$CHECKPOINT" \
            | tee -a "$WORK_DIR/eval.log"
        PYTHONHASHSEED="$SEED" IPMOE_CHECKPOINT="$CHECKPOINT" torchrun --standalone --nproc-per-node=gpu eval.py \
            --model IPMoE \
            --data $RUN_DATASETS \
            --work-dir "$WORK_DIR" \
            --judge "$RUN_JUDGE" \
            "$@" 2>&1 | tee -a "$WORK_DIR/eval.log"

        STATUS=${PIPESTATUS[0]}
        if [ "$STATUS" -eq 130 ] || [ "$STATUS" -eq 143 ]; then
            exit "$STATUS"
        fi
        if [ "$STATUS" -ne 0 ]; then
            FAILED+=("seed_$SEED judge=$RUN_JUDGE (see $WORK_DIR/eval.log)")
        fi
    done
done

if [ ${#FAILED[@]} -gt 0 ]; then
    printf 'Failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
