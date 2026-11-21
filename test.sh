#!/usr/bin/bash
# Run one of the eleven IP-MoE classes in test.py, selected by ARM (default: no_moe).
# Arms: full, image_only, no_moe, self_only, cross_only, hybrid_only, without_self, without_cross, without_hybrid,
# additive_map (the full model with the bilinear map t_i·v_j replaced by a·t_i + b·v_j), additive_map_no_moe (no_moe
# with the same replacement).
# ARM=all runs all eleven in that order, the full model (the reference) first; an arm that fails does not stop the rest.
# For each seed, train the arm as run.sh trains the full model and evaluate it as eval.sh does; then compare it with the
# full model seed by seed. For example: ARM=without_self LLM_TYPE=qwen SEEDS=42 STEPS="train eval" ./test.sh
#   train    CKPT_DIR/seed_N (checkpoint/ARM/LLM_TYPE/seed_N; checkpoint/LLM_TYPE/seed_N for ARM=full): checkpoint.pt
#            and train.log. A finished run is kept;
#            RETRAIN=1 trains it again. Each attempt trains in seed_N/attempt-<time> (follow its train.log) and replaces
#            the current run only once it has finished; the replaced run moves to seed_N/previous-<time>.
#   eval     EVAL_DIR/seed_N (evaluation/ARM/LLM_TYPE/seed_N; evaluation/LLM_TYPE/seed_N for ARM=full) and its
#            eval.log: one test.py eval per benchmark without a
#            completed evaluation of the current checkpoint (REEVALUATE=1 evaluates them all again), with eval.sh's
#            judge (EXACT_DATASETS below by exact matching) and checks. A failed benchmark is evaluated again by the
#            next run.
#   compare  EVAL_DIR/comparison/seeds_<SEEDS>_<JUDGE>[_<DATASETS>]: scores, validation loss and training counters beside
#            the full model's, with the run behind each score (sources.csv). It is replaced only by a comparison without
#            problems; one with problems is kept as <that directory>.failed-<time>.
# DATASETS chooses the benchmarks to evaluate and compare (default: the fifteen of ALL_DATASETS below), e.g.
# DATASETS="CV-Bench-2D CV-Bench-3D POPE" for the first three alone. The comparison needs every chosen benchmark of both
# models, so evaluate the full model on them first (ARM=full STEPS=eval ./test.sh).
# FULL_CKPT_DIR and FULL_EVAL_DIR hold the full model's seeds (checkpoint/LLM_TYPE and evaluation/LLM_TYPE by default).
# With MATCH=1 (default) a seed is trained and evaluated only when its full-model checkpoint exists and the arm's
# settings, GPU count and data equal that run's; MATCH=0 skips the check.
# ARM=full trains and evaluates that reference itself, in FULL_CKPT_DIR and FULL_EVAL_DIR. It has nothing to match or
# compare with, so MATCH does not apply and STEPS defaults to "train eval". To retrain the full model as a check of the
# reference instead, set CKPT_DIR and EVAL_DIR elsewhere, e.g. checkpoint/full/LLM_TYPE and evaluation/full/LLM_TYPE.
# Every train.log, eval.log and compare.log entry begins with a line naming the configuration it ran with.
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
# The three benchmarks of the first comparison, then the twelve further ones (test.py's EXTRA_DATASETS, in its order).
ALL_DATASETS="CV-Bench-2D CV-Bench-3D POPE VSR-zeroshot EmbSpatialBench 3DSRBench TopViewRS OmniSpatialBench RealWorldQA MMVP AMBER MME GQA_TestDev_Balanced HallusionBench CountBenchQA"
# Scored without a judge whatever JUDGE is (test.py's EXACT_DATASETS, which must name the same benchmarks). NaturalBench,
# GQA and CountBench never use one; the two EASI benchmarks would have a judge grade every answer, so they use their
# rule-based option matching instead.
EXACT_DATASETS="NaturalBenchDataset EmbSpatialBench OmniSpatialBench GQA_TestDev_Balanced CountBenchQA"
SEEDS=${SEEDS:-"42 43 44"}
LLM_TYPE=${LLM_TYPE:-qwen}
DATASETS=${DATASETS:-$ALL_DATASETS}
JUDGE=${JUDGE:-gpt-6-luna}
RETRAIN=${RETRAIN:-0}
REEVALUATE=${REEVALUATE:-0}
MATCH=${MATCH:-1}
ARM=${ARM:-no_moe}
ALL_ARMS="full image_only no_moe self_only cross_only hybrid_only without_self without_cross without_hybrid additive_map additive_map_no_moe"

# ARM=all runs this script for every arm in turn, the full model first: it is the reference the others are checked
# against while they train and compared with afterwards.
if [ "$ARM" = all ]; then
    if [ -n "${CKPT_DIR:-}" ] || [ -n "${EVAL_DIR:-}" ]; then
        echo "ARM=all keeps every arm in its own default directories; unset CKPT_DIR and EVAL_DIR" >&2
        exit 1
    fi
    FAILED_ARMS=()
    for EACH in $ALL_ARMS; do
        EACH_STEPS=${STEPS:-}
        if [ "$EACH" = full ] && [ -n "$EACH_STEPS" ]; then
            # The reference has nothing to be compared with.
            read -ra PARTS <<< "$EACH_STEPS"
            EACH_STEPS=
            for PART in "${PARTS[@]}"; do
                [ "$PART" != compare ] && EACH_STEPS+="$PART "
            done
            if [ -z "$EACH_STEPS" ]; then
                continue
            fi
        fi
        echo "=== $(date '+%F %T') ARM=all: running ARM=$EACH"
        ARM=$EACH STEPS=$EACH_STEPS bash "${BASH_SOURCE[0]}" "$@"
        STATUS=$?
        if [ "$STATUS" -eq 130 ]; then
            exit 130
        fi
        if [ "$STATUS" -ne 0 ]; then
            FAILED_ARMS+=("$EACH")
        fi
    done
    if [ ${#FAILED_ARMS[@]} -gt 0 ]; then
        echo "ARM=all: these arms failed (see their Failed: lines above): ${FAILED_ARMS[*]}" >&2
        exit 1
    fi
    exit 0
fi

FULL_CKPT_DIR=${FULL_CKPT_DIR:-checkpoint/$LLM_TYPE}
FULL_EVAL_DIR=${FULL_EVAL_DIR:-evaluation/$LLM_TYPE}
# The full model's runs are the reference the other arms are compared with, so ARM=full writes them there.
if [ "$ARM" = full ]; then
    CKPT_DIR=${CKPT_DIR:-$FULL_CKPT_DIR}
    EVAL_DIR=${EVAL_DIR:-$FULL_EVAL_DIR}
else
    CKPT_DIR=${CKPT_DIR:-checkpoint/$ARM/$LLM_TYPE}
    EVAL_DIR=${EVAL_DIR:-evaluation/$ARM/$LLM_TYPE}
fi
# Per GPU, as run.sh.
BATCH_SIZE=${BATCH_SIZE:-10}
same_dir() {
    [ "$(realpath -m "$1")" = "$(realpath -m "$2")" ]
}
# REFERENCE=1 when this run is the full model's reference itself, which has nothing to match or be compared with.
REFERENCE=0
if [ "$ARM" = full ] && same_dir "$CKPT_DIR" "$FULL_CKPT_DIR"; then
    REFERENCE=1
    STEPS=${STEPS:-train eval}
fi
STEPS=${STEPS:-$ALL_STEPS}

# Reject mistakes now rather than after days of training.
case "$ARM" in
    full|image_only|no_moe|self_only|cross_only|hybrid_only|without_self|without_cross|without_hybrid|additive_map|additive_map_no_moe) ;;
    *)
        echo "Unknown ARM '$ARM'; choose full, image_only, no_moe, self_only, cross_only, hybrid_only, without_self, without_cross, without_hybrid, additive_map, additive_map_no_moe or all" >&2
        exit 1 ;;
esac
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
if [ "$REFERENCE" = 1 ]; then
    MATCH=0
    if [[ " ${STEP_LIST[*]} " == *" compare "* ]]; then
        echo "ARM=full trains the reference itself in $CKPT_DIR, so there is nothing to compare it with; compare the other arms" \
             "with it, or set CKPT_DIR and EVAL_DIR elsewhere to compare a retrained full model with the reference" >&2
        exit 1
    fi
    if ! same_dir "$EVAL_DIR" "$FULL_EVAL_DIR"; then
        echo "ARM=full trains the reference in FULL_CKPT_DIR ($FULL_CKPT_DIR), so it evaluates into FULL_EVAL_DIR" \
             "($FULL_EVAL_DIR); set FULL_EVAL_DIR rather than EVAL_DIR" >&2
        exit 1
    fi
fi

# The arm's seed directories must not be where run.sh and eval.sh write by default, nor the full model's unless this run
# is that reference.
for ARM_DIR in "$CKPT_DIR" "$EVAL_DIR"; do
    for OTHER in checkpoint evaluation; do
        if same_dir "$ARM_DIR" "$OTHER"; then
            echo "$ARM_DIR is where run.sh and eval.sh write by default; choose another CKPT_DIR, EVAL_DIR or FULL_*_DIR" >&2
            exit 1
        fi
    done
    for OTHER in "$FULL_CKPT_DIR" "$FULL_EVAL_DIR"; do
        if [ "$REFERENCE" = 0 ] && same_dir "$ARM_DIR" "$OTHER"; then
            echo "$ARM_DIR is $OTHER; CKPT_DIR and EVAL_DIR must keep the arm's seeds apart from the full model's runs" >&2
            exit 1
        fi
    done
done

# Each evaluation sets its own model, datasets, judge and output directory, and predicts afresh. VLMEvalKit accepts
# abbreviated options, so a prefix of a blocked option is blocked too.
for ARG in "$@"; do
    NAME=${ARG%%=*}
    for BLOCKED in --arm --model --data --judge --work-dir --config --reuse --reuse-aux --mode; do
        if [[ $NAME == --?* && $BLOCKED == "$NAME"* ]]; then
            echo "Extra arguments cannot set $BLOCKED (from $ARG); use ARM, DATASETS, JUDGE or EVAL_DIR, and see test.py eval" >&2
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
# test.py eval refuses an evaluation directory holding runs it did not make for this arm (VLMEvalKit would relink their
# results), e.g. eval.sh's runs of the full model. Find them now rather than after days of training.
if want eval; then
    for SEED in "${SEED_LIST[@]}"; do
        for RUN in "$EVAL_DIR/seed_$SEED"/IPMoE/T*/; do
            if [ -f "$RUN/status.json" ] && [ ! -f "$RUN/$ARM.json" ]; then
                echo "$EVAL_DIR/seed_$SEED holds evaluations that test.py eval did not make for $ARM (${RUN%/});" \
                     "move that directory aside, as checkpoint/previous, or choose another EVAL_DIR" >&2
                exit 1
            fi
        done
    done
fi
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

# The configuration every log entry begins with; the arguments add the step's own settings as key=value.
config() {
    local role=ablation
    if [ "$ARM" = full ]; then
        role=$([ "$REFERENCE" = 1 ] && echo reference || echo retrained-reference)
    fi
    echo "arm=$ARM ($role) llm=$LLM_TYPE code=$CODE gpus=${CUDA_VISIBLE_DEVICES:-all} $*"
}

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
    output=$(python test.py state "$1" --arm "$ARM" --llm-type "$LLM_TYPE" --seed "$2" "${match[@]}") || return 1
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
    echo "=== $(date '+%F %T') train $(config "seed=$seed" "batch_size=$BATCH_SIZE/GPU" "match=${match[1]:-none}" "output=$staged")" \
        | tee "$staged/train.log"
    PYTHONHASHSEED="$seed" torchrun --standalone --nproc-per-node=gpu test.py train \
        --arm "$ARM" \
        "${match[@]}" \
        --seed "$seed" \
        --attn-implementation flash_attention_2 \
        --llm-type "$LLM_TYPE" \
        --output-dir "$staged" \
        --batch-size "$BATCH_SIZE" 2>&1 | tee -a "$staged/train.log"

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
    if ! output=$(python test.py pending "$checkpoint" "$work" --arm "$ARM" --judge "$JUDGE" --data "${DATASET_LIST[@]}"); then
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
    # One run per benchmark, so that a failure sends only that benchmark back. As eval.sh does for NaturalBench, the
    # benchmarks in EXACT_DATASETS run as exact_matching, so that their runs record how they were scored.
    for dataset in "${todo[@]}"; do
        run_judge=$JUDGE
        if [[ " $EXACT_DATASETS " == *" $dataset "* ]]; then
            run_judge=exact_matching
        fi
        echo "=== $(date '+%F %T') eval $(config "seed=$seed" "judge=$run_judge" "data=[$dataset]" "checkpoint=$checkpoint" \
            "context_limit=${IPMOE_CONTEXT_LIMIT:-decoder}" "extra=[${EXTRA[*]}]")" | tee -a "$work/eval.log"
        PYTHONHASHSEED="$seed" IPMOE_CHECKPOINT="$checkpoint" torchrun --standalone --nproc-per-node=gpu test.py eval \
            --arm "$ARM" \
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
    echo "=== $(date '+%F %T') compare $(config "seeds=[${SEED_LIST[*]}]" "judge=$JUDGE" "data=[${DATASET_LIST[*]}]" \
        "full=$FULL_CKPT_DIR,$FULL_EVAL_DIR" "runs=$CKPT_DIR,$EVAL_DIR")" \
        | tee "$STAGED/compare.log"
    python test.py compare \
        --arm "$ARM" \
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
