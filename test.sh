#!/usr/bin/bash
# Ablation: IP-MoE without the router and the expert pool (see test.py). The language model's prefix keeps the image
# and interaction tokens; the experts, the router, their auxiliary tokens and expert_head are removed.
# For each seed, train the arm as run.sh trains the full model and evaluate it as eval.sh does; then compare it with the
# full model seed by seed. Choose with environment variables, e.g. SEEDS=42 STEPS="train eval" ./test.sh
#   train    CKPT_DIR/seed_N (checkpoint/no_moe/llama/seed_N): checkpoint.pt and train.log. A finished run is kept;
#            RETRAIN=1 trains it again. Each attempt trains in seed_N/attempt-<time> (follow its train.log) and replaces
#            the current run only once it has finished; the replaced run moves to seed_N/previous-<time>.
#   eval     EVAL_DIR/seed_N (evaluation/no_moe/llama/seed_N) and its eval.log: one test.py eval per benchmark without a
#            completed evaluation of the current checkpoint (REEVALUATE=1 evaluates them all again), with eval.sh's
#            judges (NaturalBench by exact matching) and checks. A failed benchmark is evaluated again by the next run.
#   compare  EVAL_DIR/comparison/seeds_<SEEDS>_<JUDGE>[_<DATASETS>]: scores, validation loss and training counters beside
#            the full model's, with the run behind each score (sources.csv). It is replaced only by a comparison without
#            problems; one with problems is kept as <that directory>.failed-<time>.
# FULL_CKPT_DIR and FULL_EVAL_DIR hold the full model's seeds (checkpoint/LLM_TYPE and evaluation/LLM_TYPE by default,
# where they were moved after run.sh and eval.sh). With MATCH=1 (default) a seed is trained and evaluated only when its
# full-model checkpoint exists and the arm's settings, GPU count and data equal that run's; MATCH=0 skips the check.
# JUDGE as in eval.sh: gpt-6-luna reads OPENAI_API_KEY from VLMEvalKit/.env; JUDGE=exact_matching needs no LLM.
# Extra arguments go to every evaluation, e.g. ./test.sh --judge-api-nproc 8
#
# --batch-size 6 is per GPU, so train on as many GPUs as the full model was; choose them with CUDA_VISIBLE_DEVICES.
# Run from an environment with the training and VLMEvalKit dependencies, or run each step in its own environment.
# One test.sh at a time works on a seed (CKPT_DIR/seed_N.lock). Commit test.py and test.sh first: VLMEvalKit's
# status.json records only the commit.

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
# DataLoader workers fork after the tokenizer is loaded; keep its threads in the main process only.
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}"
# The full model was evaluated without a test-time intervention.
unset IPMOE_INTERVENTION

# torchrun exits normally on Ctrl-C, so without this the loop would move on to the next run.
trap 'exit 130' INT

ALL_STEPS="train eval compare"
ALL_DATASETS="NaturalBenchDataset CV-Bench-2D CV-Bench-3D MMBench_DEV_EN_V11 MMStar POPE SEEDBench_IMG"
STEPS=${STEPS:-$ALL_STEPS}
SEEDS=${SEEDS:-"42 43 44"}
LLM_TYPE=${LLM_TYPE:-llama}
DATASETS=${DATASETS:-$ALL_DATASETS}
JUDGE=${JUDGE:-gpt-6-luna}
RETRAIN=${RETRAIN:-0}
REEVALUATE=${REEVALUATE:-0}
MATCH=${MATCH:-1}
ARM=no_moe
CKPT_DIR=${CKPT_DIR:-checkpoint/$ARM/$LLM_TYPE}
EVAL_DIR=${EVAL_DIR:-evaluation/$ARM/$LLM_TYPE}
FULL_CKPT_DIR=${FULL_CKPT_DIR:-checkpoint/$LLM_TYPE}
FULL_EVAL_DIR=${FULL_EVAL_DIR:-evaluation/$LLM_TYPE}

# Reject mistakes now rather than after days of training.
if [ -n "${FORCE:-}" ] && [ "$FORCE" != 0 ]; then
    echo "FORCE is not used: RETRAIN=1 trains finished seeds again and REEVALUATE=1 evaluates them again" >&2
    exit 1
fi
# Split a list on any whitespace and drop repeats, keeping the order.
words() {
    local list
    read -rd '' -a list <<< "$1"
    printf '%s\n' "${list[@]}" | awk 'NF && !seen[$0]++'
}
mapfile -t STEP_LIST < <(words "$STEPS")
mapfile -t SEED_LIST < <(words "$SEEDS")
mapfile -t DATASET_LIST < <(words "$DATASETS")
for STEP in "${STEP_LIST[@]}"; do
    if [[ " $ALL_STEPS " != *" $STEP "* ]]; then
        echo "Unknown step '$STEP'; choose from: $ALL_STEPS" >&2
        exit 1
    fi
done
for SEED in "${SEED_LIST[@]}"; do
    if ! [[ $SEED =~ ^[0-9]+$ ]]; then
        echo "Seeds must be integers separated by spaces, not '$SEED'" >&2
        exit 1
    fi
done
for DATASET in "${DATASET_LIST[@]}"; do
    if [[ " $ALL_DATASETS " != *" $DATASET "* ]]; then
        echo "Unknown dataset '$DATASET'; choose from the full model's benchmarks: $ALL_DATASETS" >&2
        exit 1
    fi
done
if [ ${#STEP_LIST[@]} -eq 0 ] || [ ${#SEED_LIST[@]} -eq 0 ] || [ ${#DATASET_LIST[@]} -eq 0 ]; then
    echo "STEPS, SEEDS and DATASETS must not be empty" >&2
    exit 1
fi
for SWITCH in RETRAIN REEVALUATE MATCH; do
    case "${!SWITCH}" in 0|1) ;; *) echo "$SWITCH must be 0 or 1" >&2; exit 1 ;; esac
done

# The arm's seed directories must never be the full model's, nor where run.sh and eval.sh write by default.
for ARM_DIR in "$CKPT_DIR" "$EVAL_DIR"; do
    for OTHER in "$FULL_CKPT_DIR" "$FULL_EVAL_DIR" checkpoint evaluation; do
        if [ "$(realpath -m "$ARM_DIR")" = "$(realpath -m "$OTHER")" ]; then
            echo "$ARM_DIR is $OTHER; CKPT_DIR and EVAL_DIR must keep the arm's seeds apart from the full model's runs" >&2
            exit 1
        fi
    done
done

# Each evaluation sets its own model, datasets, judge and output directory, and predicts afresh. VLMEvalKit accepts
# abbreviated options, so a prefix of a blocked option is blocked too.
for ARG in "$@"; do
    NAME=${ARG%%=*}
    for BLOCKED in --model --data --judge --work-dir --config --reuse --reuse-aux --mode; do
        if [[ $NAME == --?* && $BLOCKED == "$NAME"* ]]; then
            echo "Extra arguments cannot set $BLOCKED (from $ARG); use DATASETS, JUDGE or EVAL_DIR, and see test.py eval" >&2
            exit 1
        fi
    done
done
EXTRA=("$@")

want() {
    [[ " ${STEP_LIST[*]} " == *" $1 "* ]]
}

# Fail now, not after the first seed has trained, in an environment without a step's dependencies.
check_python() {
    local error
    if ! error=$(python -c "$2" 2>&1); then
        echo "$1: ${error##*$'\n'}" >&2
        exit 1
    fi
}
if want train || want eval; then
    for COMMAND in torchrun flock; do
        if ! command -v "$COMMAND" > /dev/null; then
            echo "$COMMAND is not on PATH" >&2
            exit 1
        fi
    done
fi
if want train; then
    check_python "train needs the training environment" 'import main'
fi
if want eval; then
    if [ ! -f VLMEvalKit/run.py ]; then
        echo "eval needs VLMEvalKit cloned into ./VLMEvalKit" >&2
        exit 1
    fi
    check_python "eval needs the VLMEvalKit environment" 'import main, pandas, vlmeval'
fi
if want compare; then
    check_python "compare needs the training environment and pandas" 'import main, pandas'
fi

# As in rebuttal.sh: untracked Python or shell files are code too; other untracked files (archives, logs) are not.
CODE=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] || [ -n "$(git ls-files --others --exclude-standard -- '*.py' '*.sh' 2>/dev/null)" ]; then
    CODE="$CODE+uncommitted"
    echo "=== code=$CODE: commit test.py and test.sh so that the results name the code that produced them" >&2
fi

FAILED=()

# With MATCH=1 a seed needs the full model's checkpoint to check the arm against. $1 is the seed, $2 the step.
full_checkpoint() {
    local full=$FULL_CKPT_DIR/seed_$1/checkpoint.pt
    if [ "$MATCH" = 1 ] && [ ! -f "$full" ]; then
        FAILED+=("$2 seed_$1 ($full not found; set FULL_CKPT_DIR, or MATCH=0 to skip comparing settings)")
        return 1
    fi
}

# Print the state of the arm checkpoint $1 of seed $2 (see test.py state), or fail if test.py cannot tell.
arm_state() {
    local output
    local match=()
    if [ "$MATCH" = 1 ]; then
        match=(--match "$FULL_CKPT_DIR/seed_$2/checkpoint.pt")
    fi
    output=$(python test.py state "$1" --llm-type "$LLM_TYPE" --seed "$2" "${match[@]}") || return 1
    # Only the last line: libraries may print while they load.
    echo "${output##*$'\n'}"
}

train() {
    local seed=$1 dir=$CKPT_DIR/seed_$1 state staged previous file
    local match=() earlier=()

    full_checkpoint "$seed" train || return 1
    if [ "$MATCH" = 1 ]; then
        match=(--match "$FULL_CKPT_DIR/seed_$seed/checkpoint.pt")
    fi
    if ! state=$(arm_state "$dir/checkpoint.pt" "$seed"); then
        FAILED+=("train seed_$seed (test.py state failed on $dir/checkpoint.pt or the full-model checkpoint; see the error above)")
        return 1
    fi
    case "$state" in
        finished)
            if [ "$RETRAIN" = 0 ]; then
                echo "=== Keeping seed=$seed: $dir/checkpoint.pt finished training (RETRAIN=1 trains it again)"
                return 0
            fi ;;
        mismatched)
            if [ "$RETRAIN" = 0 ]; then
                FAILED+=("train seed_$seed ($dir/checkpoint.pt has other settings than the full model's; RETRAIN=1 trains it again)")
                return 1
            fi ;;
        absent|unfinished) ;;
        *)
            # Never replace another run, e.g. the full model's, whatever RETRAIN says.
            FAILED+=("train seed_$seed ($dir/checkpoint.pt belongs to another run; choose another CKPT_DIR)")
            return 1 ;;
    esac

    # The seed's current run stays in place until the new attempt has finished.
    if ! staged=$(mkdir -p "$dir" && mktemp -d "$dir/attempt-$(date '+%Y%m%d-%H%M%S')-XXXX"); then
        FAILED+=("train seed_$seed (cannot create an attempt directory in $dir)")
        return 1
    fi
    echo "=== $(date '+%F %T') code=$CODE train seed=$seed llm=$LLM_TYPE output=$staged" | tee "$staged/train.log"
    PYTHONHASHSEED="$seed" torchrun --standalone --nproc-per-node=gpu test.py train \
        "${match[@]}" \
        --seed "$seed" \
        --attn-implementation flash_attention_2 \
        --llm-type "$LLM_TYPE" \
        --output-dir "$staged" \
        --batch-size 6 2>&1 | tee -a "$staged/train.log"

    # Keep going after a failed run; the attempt's log keeps its traceback.
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        FAILED+=("train seed_$seed (see $staged/train.log)")
        return 1
    fi
    if ! state=$(arm_state "$staged/checkpoint.pt" "$seed") || [ "$state" != finished ]; then
        FAILED+=("train seed_$seed ($staged/checkpoint.pt is ${state:-unreadable} after training)")
        return 1
    fi

    # Move the replaced run aside whole, then the new one into place.
    for file in checkpoint.pt train.log; do
        if [ -e "$dir/$file" ]; then
            earlier+=("$dir/$file")
        fi
    done
    if [ ${#earlier[@]} -gt 0 ]; then
        if ! previous=$(mktemp -d "$dir/previous-$(date '+%Y%m%d-%H%M%S')-XXXX") || ! mv -- "${earlier[@]}" "$previous/"; then
            FAILED+=("train seed_$seed (cannot move the replaced run aside; the new one is in $staged)")
            return 1
        fi
        echo "=== Moved the replaced run of seed=$seed to $previous"
    fi
    if ! mv -- "$staged/checkpoint.pt" "$staged/train.log" "$dir/" || ! rmdir "$staged"; then
        FAILED+=("train seed_$seed (cannot move $staged into $dir)")
        return 1
    fi
}

evaluate() {
    local seed=$1 checkpoint=$CKPT_DIR/seed_$1/checkpoint.pt work=$EVAL_DIR/seed_$1 state output dataset run_judge
    local todo=()

    full_checkpoint "$seed" eval || return 1
    # Seeds still training, or whose training failed, have nothing to evaluate yet.
    if ! state=$(arm_state "$checkpoint" "$seed"); then
        FAILED+=("eval seed_$seed (test.py state failed on $checkpoint or the full-model checkpoint; see the error above)")
        return 1
    fi
    if [ "$state" != finished ]; then
        FAILED+=("eval seed_$seed ($checkpoint is $state, not a finished $ARM run with the full model's settings)")
        return 1
    fi

    # The benchmarks without a completed evaluation of this checkpoint. It refuses a directory with other models' runs.
    if ! output=$(python test.py pending "$checkpoint" "$work" --judge "$JUDGE" --data "${DATASET_LIST[@]}"); then
        FAILED+=("eval seed_$seed (cannot evaluate into $work; see the error above)")
        return 1
    fi
    read -ra todo <<< "$(sed -n 's/^pending://p' <<< "$output" | tail -n 1)"
    if [ "$REEVALUATE" = 1 ]; then
        todo=("${DATASET_LIST[@]}")
    fi
    if [ ${#todo[@]} -eq 0 ]; then
        echo "=== Keeping seed=$seed: $work has completed evaluations of this checkpoint (REEVALUATE=1 evaluates it again)"
        return 0
    fi

    mkdir -p "$work"
    # One run per benchmark, so that a failure sends only that benchmark back. As in eval.sh, NaturalBench scores with
    # its own answer extraction and never uses a judge, so it runs as exact_matching.
    for dataset in "${todo[@]}"; do
        run_judge=$JUDGE
        if [ "$dataset" = NaturalBenchDataset ]; then
            run_judge=exact_matching
        fi
        echo "=== $(date '+%F %T') code=$CODE eval seed=$seed judge=$run_judge data=[$dataset] checkpoint=$checkpoint" | tee -a "$work/eval.log"
        PYTHONHASHSEED="$seed" IPMOE_CHECKPOINT="$checkpoint" torchrun --standalone --nproc-per-node=gpu test.py eval \
            --model IPMoE \
            --data "$dataset" \
            --work-dir "$work" \
            --judge "$run_judge" \
            "${EXTRA[@]}" 2>&1 | tee -a "$work/eval.log"

        # Keep going after a failed evaluation; the log keeps its traceback.
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            FAILED+=("eval seed_$seed $dataset (see $work/eval.log)")
        fi
    done
}

# Each seed is evaluated as soon as it is trained.
for SEED in "${SEED_LIST[@]}"; do
    # A run still training has no checkpoint yet; a second test.sh would take it for an abandoned attempt.
    if want train || want eval; then
        mkdir -p "$CKPT_DIR"
        exec 9> "$CKPT_DIR/seed_$SEED.lock"
        if ! flock -n 9; then
            FAILED+=("seed_$SEED (another test.sh is training or evaluating it)")
            continue
        fi
    fi
    if want train && ! train "$SEED"; then
        continue
    fi
    if want eval; then
        evaluate "$SEED"
    fi
done
exec 9>&-

if want compare; then
    # One directory per selection of seeds, judge and benchmarks.
    NAME=seeds_$(printf '%s\n' "${SEED_LIST[@]}" | sort -n | paste -sd _ -)_$JUDGE
    if [ "${DATASET_LIST[*]}" != "$ALL_DATASETS" ]; then
        NAME+=_$(IFS=+; echo "${DATASET_LIST[*]}")
    fi
    COMPARISON=$EVAL_DIR/comparison/$NAME
    mkdir -p "$EVAL_DIR/comparison"
    STAGED=$(mktemp -d "$COMPARISON.tmp-XXXX")
    echo "=== $(date '+%F %T') code=$CODE compare seeds=[${SEED_LIST[*]}] judge=$JUDGE data=[${DATASET_LIST[*]}] full=$FULL_CKPT_DIR,$FULL_EVAL_DIR arm=$CKPT_DIR,$EVAL_DIR" \
        | tee "$STAGED/compare.log"
    python test.py compare \
        --seeds "${SEED_LIST[@]}" \
        --data "${DATASET_LIST[@]}" \
        --judge "$JUDGE" \
        --full-checkpoints "$FULL_CKPT_DIR" \
        --full-results "$FULL_EVAL_DIR" \
        --arm-checkpoints "$CKPT_DIR" \
        --arm-results "$EVAL_DIR" \
        --output "$STAGED" 2>&1 | tee -a "$STAGED/compare.log"

    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        if [ -e "$COMPARISON" ]; then
            mv -- "$COMPARISON" "$COMPARISON.previous-$(date '+%Y%m%d-%H%M%S')"
        fi
        mv -- "$STAGED" "$COMPARISON"
        echo "=== Moved to $COMPARISON" | tee -a "$COMPARISON/compare.log"
    else
        KEPT=$COMPARISON.failed-$(date '+%Y%m%d-%H%M%S')
        mv -- "$STAGED" "$KEPT"
        FAILED+=("compare (see $KEPT/compare.log)")
    fi
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    printf 'Failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
