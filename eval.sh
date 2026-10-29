#!/usr/bin/bash
# Evaluate every ablation arm in ablation.py for each seed, finishing all arms of one seed before the next.
# Choose a subset with environment variables, e.g. SEEDS=42 ABLATIONS="full uniform" DATASETS="MMStar POPE" ./eval.sh
# JUDGE extracts answers that rule-based matching cannot parse; gpt-6-luna reads OPENAI_API_KEY from VLMEvalKit/.env.
# JUDGE=exact_matching scores every benchmark without an LLM.
# Extra arguments go to eval.py, e.g. ./eval.sh --judge-api-nproc 8
# FULL_DIR holds the full-model seeds (checkpoint by default, where run.sh writes them; checkpoint/llama once moved)
# and OUT the results, e.g. FULL_DIR=checkpoint/llama OUT=evaluation/llama ABLATIONS=full ./eval.sh
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

ALL_ABLATIONS=$(python -c 'from ablation import ABLATIONS; print(*ABLATIONS)') || exit 1
SEEDS=${SEEDS:-"42"}
ABLATIONS=${ABLATIONS:-$ALL_ABLATIONS}
DATASETS=${DATASETS:-"NaturalBenchDataset CV-Bench-2D CV-Bench-3D MMBench_DEV_EN_V11 MMStar POPE SEEDBench_IMG"}
JUDGE=${JUDGE:-gpt-6-luna}
FULL_DIR=${FULL_DIR:-checkpoint}
OUT=${OUT:-evaluation}

# Reject typos now rather than after hours of other evaluations.
for ABLATION in $ABLATIONS; do
    if [[ " $ALL_ABLATIONS " != *" $ABLATION "* ]]; then
        echo "Unknown ablation '$ABLATION'; choose from: $ALL_ABLATIONS" >&2
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
for SEED in $SEEDS; do
    for ABLATION in $ABLATIONS; do
        # Load the checkpoint run.sh wrote and mirror its layout under OUT.
        if [ "$ABLATION" = full ]; then
            CHECKPOINT=$FULL_DIR/seed_$SEED/checkpoint.pt
            WORK_DIR=$OUT/seed_$SEED
        else
            CHECKPOINT=checkpoint/ablations/$ABLATION/seed_$SEED/checkpoint.pt
            WORK_DIR=$OUT/ablations/$ABLATION/seed_$SEED
        fi

        # Arms still training, or whose training failed, have no checkpoint yet.
        if [ ! -f "$CHECKPOINT" ]; then
            echo "=== Skipping ablation=$ABLATION seed=$SEED: $CHECKPOINT not found" >&2
            FAILED+=("$ABLATION/seed_$SEED (no checkpoint)")
            continue
        fi
        mkdir -p "$WORK_DIR"
        : > "$WORK_DIR/eval.log"

        for PROTOCOL in "${PROTOCOLS[@]}"; do
            read -r RUN_JUDGE RUN_DATASETS <<< "$PROTOCOL"
            echo "=== $(date '+%F %T') ablation=$ABLATION seed=$SEED judge=$RUN_JUDGE data=[$RUN_DATASETS] checkpoint=$CHECKPOINT intervention=${IPMOE_INTERVENTION:-none}" \
                | tee -a "$WORK_DIR/eval.log"
            PYTHONHASHSEED="$SEED" IPMOE_CHECKPOINT="$CHECKPOINT" torchrun --standalone --nproc-per-node=gpu eval.py \
                --model IPMoE \
                --data $RUN_DATASETS \
                --work-dir "$WORK_DIR" \
                --judge "$RUN_JUDGE" \
                "$@" 2>&1 | tee -a "$WORK_DIR/eval.log"

            # Keep going after a failed evaluation; the log keeps its traceback.
            if [ "${PIPESTATUS[0]}" -ne 0 ]; then
                FAILED+=("$ABLATION/seed_$SEED judge=$RUN_JUDGE (see $WORK_DIR/eval.log)")
            fi
        done
    done
done

if [ ${#FAILED[@]} -gt 0 ]; then
    printf 'Failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
