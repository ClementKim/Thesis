#!/usr/bin/bash
# Evaluate the external baselines in vlmeval_baselines.py with the same datasets, judge and failure checks as eval.sh.
# Choose a subset with environment variables, e.g. MODELS="SpatialLadder-3B Spatial-SSRL-3B" DATASETS="MMStar POPE" ./eval_baselines.sh
# Baselines decode greedily, so one run per model; SEED only fixes PYTHONHASHSEED.
# JUDGE extracts answers that rule-based matching cannot parse; gpt-6-luna reads OPENAI_API_KEY from VLMEvalKit/.env.
# JUDGE=exact_matching scores every benchmark without an LLM.
# Extra arguments go to eval.py, e.g. ./eval_baselines.sh --judge-api-nproc 8
#
# Choose GPUs with CUDA_VISIBLE_DEVICES:
# CUDA_VISIBLE_DEVICES=0 ./eval_baselines.sh     # GPU 0 only
# CUDA_VISIBLE_DEVICES=0,1 ./eval_baselines.sh   # GPUs 0 and 1

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
export PYTHONPATH="$PWD/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}"

# torchrun exits normally on Ctrl-C, so without this the loop would move on to the next run.
trap 'exit 130' INT

ALL_MODELS=$(python -c 'from vlmeval_baselines import BASELINES; print(*BASELINES)') || exit 1
SEED=${SEED:-42}
MODELS=${MODELS:-$ALL_MODELS}
DATASETS=${DATASETS:-"NaturalBenchDataset CV-Bench-2D CV-Bench-3D MMBench_DEV_EN_V11 MMStar POPE SEEDBench_IMG"}
JUDGE=${JUDGE:-gpt-6-luna}

# Reject typos now rather than after hours of other evaluations.
for MODEL in $MODELS; do
    if [[ " $ALL_MODELS " != *" $MODEL "* ]]; then
        echo "Unknown model '$MODEL'; choose from: $ALL_MODELS" >&2
        exit 1
    fi
done

# Each scoring group below sets its own datasets and judge.
for ARG in "$@"; do
    case "$ARG" in
        --data|--data=*|--judge|--judge=*)
            echo "Set DATASETS or JUDGE instead of passing $ARG" >&2
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
for MODEL in $MODELS; do
    WORK_DIR=evaluation/baselines/$MODEL
    mkdir -p "$WORK_DIR"
    : > "$WORK_DIR/eval.log"

    for PROTOCOL in "${PROTOCOLS[@]}"; do
        read -r RUN_JUDGE RUN_DATASETS <<< "$PROTOCOL"
        echo "=== $(date '+%F %T') model=$MODEL judge=$RUN_JUDGE data=[$RUN_DATASETS]" | tee -a "$WORK_DIR/eval.log"
        PYTHONHASHSEED="$SEED" torchrun --standalone --nproc-per-node=gpu eval.py \
            --model "$MODEL" \
            --data $RUN_DATASETS \
            --work-dir "$WORK_DIR" \
            --judge "$RUN_JUDGE" \
            "$@" 2>&1 | tee -a "$WORK_DIR/eval.log"

        # Keep going after a failed evaluation; the log keeps its traceback.
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            FAILED+=("$MODEL judge=$RUN_JUDGE (see $WORK_DIR/eval.log)")
        fi
    done
done

if [ ${#FAILED[@]} -gt 0 ]; then
    printf 'Failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
