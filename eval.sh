#!/usr/bin/bash
# Evaluate the trained model for each seed.
# Choose a subset with environment variables, e.g. SEEDS=42 DATASETS="MMStar POPE" ./eval.sh
# JUDGE extracts answers that rule-based matching cannot parse; gpt-6-luna reads OPENAI_API_KEY from VLMEvalKit/.env.
# JUDGE=exact_matching scores every benchmark without an LLM.
# Extra arguments go to eval.py, e.g. ./eval.sh --judge-api-nproc 8
# FULL_DIR holds the model seeds and OUT the results: checkpoint/LLM_TYPE, where run.sh writes them, and
# evaluation/LLM_TYPE by default, e.g. LLM_TYPE=qwen ./eval.sh. ARM=full ./test.sh evaluates into the same OUT; it
# refuses a directory that holds eval.sh's runs, so evaluate each reference with one of the two.
# Every eval.log entry begins with a line naming the configuration it ran with.
# IPMOE_INTERVENTION applies a test-time intervention (see vlmeval_adapter.py); give it its own OUT.
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

SEEDS=${SEEDS:-"42"}
DATASETS=${DATASETS:-"NaturalBenchDataset CV-Bench-2D CV-Bench-3D MMBench_DEV_EN_V11 MMStar POPE SEEDBench_IMG"}
JUDGE=${JUDGE:-gpt-6-luna}
LLM_TYPE=${LLM_TYPE:-llama}
FULL_DIR=${FULL_DIR:-checkpoint/$LLM_TYPE}
OUT=${OUT:-evaluation/$LLM_TYPE}

# As in test.sh: untracked Python or shell files are code too.
CODE=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] || [ -n "$(git ls-files --others --exclude-standard -- '*.py' '*.sh' 2>/dev/null)" ]; then
    CODE="$CODE+uncommitted"
fi

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
for SEED in $SEEDS; do
    # Load the checkpoint run.sh wrote and mirror its layout under OUT.
    CHECKPOINT=$FULL_DIR/seed_$SEED/checkpoint.pt
    WORK_DIR=$OUT/seed_$SEED

    # Seeds still training, or whose training failed, have no checkpoint yet.
    if [ ! -f "$CHECKPOINT" ]; then
        echo "=== Skipping seed=$SEED: $CHECKPOINT not found" >&2
        FAILED+=("seed_$SEED (no checkpoint)")
        continue
    fi
    mkdir -p "$WORK_DIR"
    : > "$WORK_DIR/eval.log"

    for PROTOCOL in "${PROTOCOLS[@]}"; do
        read -r RUN_JUDGE RUN_DATASETS <<< "$PROTOCOL"
        echo "=== $(date '+%F %T') eval arm=full llm=$LLM_TYPE code=$CODE gpus=${CUDA_VISIBLE_DEVICES:-all} seed=$SEED" \
             "judge=$RUN_JUDGE data=[$RUN_DATASETS] checkpoint=$CHECKPOINT intervention=${IPMOE_INTERVENTION:-none}" \
             "context_limit=${IPMOE_CONTEXT_LIMIT:-decoder} extra=[$*]" | tee -a "$WORK_DIR/eval.log"
        PYTHONHASHSEED="$SEED" IPMOE_CHECKPOINT="$CHECKPOINT" torchrun --standalone --nproc-per-node=gpu eval.py \
            --model IPMoE \
            --data $RUN_DATASETS \
            --work-dir "$WORK_DIR" \
            --judge "$RUN_JUDGE" \
            "$@" 2>&1 | tee -a "$WORK_DIR/eval.log"

        # Keep going after a failed evaluation; the log keeps its traceback.
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            FAILED+=("seed_$SEED judge=$RUN_JUDGE (see $WORK_DIR/eval.log)")
        fi
    done
done

if [ ${#FAILED[@]} -gt 0 ]; then
    printf 'Failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
