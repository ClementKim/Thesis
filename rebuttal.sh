#!/usr/bin/bash
# Evidence for IP_MoE_Final_Defense_Report.md, in this order; choose steps with STEPS, e.g. STEPS="h1 stats" ./rebuttal.sh.
#   provenance     code revision, environment, the VLMEvalKit fork as an archive, file inventory for authorship (R2, R3)
#   counts         checkpoint counters, split hashes and wall time from train.log, active parameters, prefix tokens (R4, R8)
#   overlap        benchmark images among the training images, by SHA-256 and perceptual hash; CPU, in the background
#   center         H1 of token-centred maps (exploratory; not part of the report's revisions)
#   init           H1 of the interaction MLPs each seed started training from (R5 route A)
#   cosine         H1 of the trained features compared by cosine, as frozen CLIP's are (R5 route A)
#   dot            H1 of frozen CLIP compared by raw dot product, as the trained map's are (R5 route A)
#   h1             H1 recomputed from every items.csv: tie-repaired sign, signed peaks, same box, pooled seeds (R1)
#   router         router weights of every full seed (H2)
#   cambrian       LLaVA-NeXT-8B on CV-Bench with Cambrian-1's prompt (R10)
#   interventions  swap, mean and uniform on every full seed, and each expert's weight set to 0 on ZERO_SEEDS (R5 route A)
#   records        result links, the commit behind each result, and prediction agreement between reruns (R2, R4)
#   stats          validated scores and seed SD, paired intervals against full seed 42, Tables 4.5-4.11 (R2, R6, R7)
#   baselines      every evaluated baseline, unreported ones included, with protocol, published values, judge use (R10)
# Failed steps are listed at the end; the others still run.
#
# Measurements and evaluations that already have complete outputs are kept; FORCE=1 redoes them, moving old H1 outputs
# aside. Analyses always rerun, since they only read saved results. Everything they write goes to OUT; no step changes an
# original result or repoints a link, so canonical links stay a decision recorded by hand after reading records_links.csv.
# Commit the code before running: VLMEvalKit's status.json records the commit but not uncommitted changes. Every log
# banner records the commit and whether the working tree differed from it.
#
# Run from the ipmoe-vlmeval environment. Evaluations use every GPU in CUDA_VISIBLE_DEVICES; H1 and the router use
# SMALL_GPU (the first visible GPU by default). JUDGE (gpt-6-luna) needs OPENAI_API_KEY in VLMEvalKit/.env, as for eval.sh.
# NB_METRIC=Acc is the NaturalBench score in the thesis's Avg.

export PYTHONPATH="$PWD/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}"
trap 'exit 130' INT

ALL_STEPS="provenance counts overlap center init cosine dot h1 router cambrian interventions records stats baselines"
STEPS=${STEPS:-$ALL_STEPS}
SEEDS=${SEEDS:-"42 43 44"}
ZERO_SEEDS=${ZERO_SEEDS:-42}
JUDGE=${JUDGE:-gpt-6-luna}
NB_METRIC=${NB_METRIC:-Acc}
FORCE=${FORCE:-0}
REPLICATES=${REPLICATES:-2000}
OUT=${OUT:-evaluation/rebuttal}
FULL_DIR=${FULL_DIR:-checkpoint/llama}
# Every arm was compared with full seed 42, which shares its initialization and data order.
REFERENCE=${REFERENCE:-evaluation/llama/seed_42}
PACKAGE=${PACKAGE:-thesis.tar}
# SHA-256 of thesis(5).tar, the archive IP_MoE_Final_Defense_Report.md reviewed.
REVIEWED_SHA256=32d8c9aa542e8bbb8553ea650c893cede42e6ae11a643200774959172793a5f8
DATASETS="NaturalBenchDataset CV-Bench-2D CV-Bench-3D MMBench_DEV_EN_V11 MMStar POPE SEEDBench_IMG"
SMALL_GPU=${SMALL_GPU:-$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)}

case "$FORCE" in 0|1) ;; *) echo "FORCE must be 0 or 1" >&2; exit 1 ;; esac
for STEP in $STEPS; do
    if [[ " $ALL_STEPS " != *" $STEP "* ]]; then
        echo "Unknown step '$STEP'; choose from: $ALL_STEPS" >&2
        exit 1
    fi
done
python -c 'import transformers, vlmeval' 2>/dev/null || { echo 'Activate the ipmoe-vlmeval environment first' >&2; exit 1; }

FORCE_ARGS=()
[ "$FORCE" = 1 ] && FORCE_ARGS=(--force)
mkdir -p "$OUT"
FAILED=()

# Untracked Python or shell files are code too; other untracked files (archives, the report) are not.
CODE=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] || [ -n "$(git ls-files --others --exclude-standard -- '*.py' '*.sh' 2>/dev/null)" ]; then
    CODE="$CODE+uncommitted"
fi

want() {
    [[ " $STEPS " == *" $1 "* ]]
}

banner() {
    echo "=== $(date '+%F %T') code=$CODE $*"
}

# Run a command, logging to $OUT/<name>.log, and record a failure without stopping.
run() {
    local name=$1
    shift
    banner "$name: $*" | tee -a "$OUT/$name.log"
    "$@" 2>&1 | tee -a "$OUT/$name.log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        FAILED+=("$name (see $OUT/$name.log)")
        return 1
    fi
}

# The per-question file each benchmark's analysis reads; a dangling link counts as missing.
result_file() {
    local run=$1 dataset=$2 model=${3:-IPMoE}
    case "$dataset" in
        NaturalBenchDataset) echo "$run/$model/${model}_$dataset.xlsx" ;;
        POPE) echo "$run/$model/${model}_${dataset}_auxmatch.xlsx" ;;
        *) echo "$run/$model/${model}_${dataset}_${JUDGE}_result.xlsx" ;;
    esac
}

# Benchmarks of RUN (among $3, all seven by default) without a result; FORCE=1 counts every one as missing.
missing_datasets() {
    local run=$1 model=$2 dataset missing=()
    for dataset in ${3:-$DATASETS}; do
        if [ "$FORCE" = 1 ] || [ ! -f "$(result_file "$run" "$dataset" "$model")" ]; then
            missing+=("$dataset")
        fi
    done
    echo "${missing[*]}"
}

complete() {
    [ -z "$(FORCE=0 missing_datasets "$1" "${2:-IPMoE}" "$3")" ]
}

# Measure H1 into DIR unless it holds a complete measurement; a new one is staged and replaces DIR only when complete.
measure_h1() {
    local name=$1 dir=$2 staged
    shift 2
    if [ "$FORCE" = 0 ] && [ -s "$dir/items.csv" ] && [ -s "$dir/summary.json" ]; then
        banner "$name: keeping $dir"
        return 0
    fi
    mkdir -p "$(dirname "$dir")"
    staged=$(mktemp -d "$(dirname "$dir")/.h1.XXXXXX") || { FAILED+=("$name (cannot stage H1 output)"); return 1; }
    if ! run "$name" env CUDA_VISIBLE_DEVICES="$SMALL_GPU" python h1.py --output-dir "$staged" "$@"; then
        rm -rf -- "$staged"
        return 1
    fi
    if [ ! -s "$staged/items.csv" ] || [ ! -s "$staged/summary.json" ]; then
        rm -rf -- "$staged"
        FAILED+=("$name (H1 returned without complete outputs)")
        return 1
    fi
    cp -- "$OUT/$name.log" "$staged/h1.log"
    [ ! -e "$dir" ] || mv -- "$dir" "$dir.previous-$(date +%Y%m%d-%H%M%S)"
    mv -- "$staged" "$dir"
}

# CV-Bench-3D correctness for H1's regression, when the checkpoint's evaluation has it.
predictions() {
    local file
    file=$(result_file "$1" CV-Bench-3D)
    [ -f "$file" ] && echo "--predictions $file"
}

ARMS=$(python -c 'from ablation import ABLATIONS; print(*(name for name in ABLATIONS if name != "full"))') || exit 1
FULL_RUNS=()
for SEED in $SEEDS; do
    FULL_RUNS+=("evaluation/llama/seed_$SEED")
done
QWEN_RUNS=()
for RUN in evaluation/qwen/seed_*; do
    [ -d "$RUN" ] && QWEN_RUNS+=("$RUN")
done
ARM_RUNS=()
for ARM in $ARMS; do
    for RUN in evaluation/ablations/$ARM/seed_*; do
        [ -d "$RUN" ] && ARM_RUNS+=("$RUN")
    done
done
# Reported baselines, then those the thesis left out of its results.
BASELINE_RUNS=()
for RUN in evaluation/baselines/* "evaluation/Not used"/*; do
    [ -d "$RUN" ] && BASELINE_RUNS+=("$RUN")
done
INTERVENTIONS="swap mean uniform zero_self zero_cross zero_hybrid"
intervention_seeds() {
    case "$1" in zero_*) echo "$ZERO_SEEDS" ;; *) echo "$SEEDS" ;; esac
}

## R2, R3: Code, Environment, Fork and File Inventory
if want provenance; then
    PACKAGE_ARGS=()
    [ -f "$PACKAGE" ] && PACKAGE_ARGS=(--package "$PACKAGE" --reviewed-sha256 "$REVIEWED_SHA256")
    run provenance python rebuttal.py --output "$OUT" provenance "${PACKAGE_ARGS[@]}"
fi

## R4, R8: Checkpoint Counters, Training Logs and Prefix Tokens
if want counts; then
    run counts python rebuttal.py --output "$OUT" counts --prompts "$REFERENCE" --judge "$JUDGE"
fi

## Minor Corrections: Benchmark Images in the Training Data, in the Background
OVERLAP_PID=
if want overlap; then
    banner "overlap: started in the background, log $OUT/overlap.log" | tee -a "$OUT/overlap.log"
    python rebuttal.py --output "$OUT" overlap "${FORCE_ARGS[@]}" >> "$OUT/overlap.log" 2>&1 &
    OVERLAP_PID=$!
fi

## Exploratory: Token-Centred H1
if want center; then
    measure_h1 center_clip evaluation/h1_centered/clip --reference clip --center
    for SEED in $SEEDS; do
        measure_h1 "center_$SEED" "evaluation/h1_centered/llama/seed_$SEED" --checkpoint "$FULL_DIR/seed_$SEED/checkpoint.pt" \
            --center $(predictions "evaluation/llama/seed_$SEED")
    done
    measure_h1 center_all_map_mean evaluation/h1_centered/ablations/all_map_mean/seed_42 \
        --checkpoint checkpoint/ablations/all_map_mean/seed_42/checkpoint.pt --center $(predictions evaluation/ablations/all_map_mean/seed_42)
fi

## R5 Route A: H1 at Initialization
if want init; then
    for SEED in $SEEDS; do
        CHECKPOINT=checkpoint/initial/llama/seed_$SEED/checkpoint.pt
        DIR=evaluation/h1/initial/seed_$SEED
        if [ "$FORCE" = 0 ] && [ -s "$DIR/items.csv" ] && [ -s "$DIR/summary.json" ]; then
            banner "init_$SEED: keeping $DIR"
            continue
        fi
        # The reconstruction is checked against the trained weights but is not proven byte-identical to training's start.
        if [ "$FORCE" = 1 ] || [ ! -f "$CHECKPOINT" ]; then
            run "init_$SEED" env CUDA_VISIBLE_DEVICES= python rebuttal.py init --checkpoint "$FULL_DIR/seed_$SEED/checkpoint.pt" \
                --output "$CHECKPOINT" || continue
        fi
        measure_h1 "h1_init_$SEED" "$DIR" --checkpoint "$CHECKPOINT"
    done
fi

## R5 Route A: Normalization Controls
# Frozen CLIP's map is a cosine and the trained map a raw dot product; swapping them separates normalization from
# projection and training. Neither control by itself explains what training did.
if want cosine; then
    for SEED in $SEEDS; do
        measure_h1 "cosine_$SEED" "evaluation/h1_cosine/llama/seed_$SEED" --checkpoint "$FULL_DIR/seed_$SEED/checkpoint.pt" \
            --similarity cosine $(predictions "evaluation/llama/seed_$SEED")
    done
fi
if want dot; then
    measure_h1 dot_clip evaluation/h1_dot/clip --reference clip --similarity dot
fi

## R1: H1 With the Tie-Repaired Sign, From the Saved Measurements
if want h1; then
    H1_ARGS=()
    # Groups of complete measurements; h1 compares only groups that exist.
    add_group() {
        local name=$1 dir dirs=()
        shift
        for dir in "$@"; do
            [ -s "$dir/items.csv" ] && dirs+=("$dir")
        done
        [ ${#dirs[@]} -eq 0 ] || H1_ARGS+=(--group "$name" "${dirs[@]}")
    }
    seeded() {
        local pattern=$1 seed
        for seed in $SEEDS; do
            echo "${pattern//SEED/$seed}"
        done
    }
    add_group full $(seeded evaluation/h1/llama/seed_SEED)
    add_group all_map_mean evaluation/h1/ablations/all_map_mean/seed_42
    add_group clip evaluation/h1/clip
    add_group initial $(seeded evaluation/h1/initial/seed_SEED)
    add_group full_cosine $(seeded evaluation/h1_cosine/llama/seed_SEED)
    add_group clip_dot evaluation/h1_dot/clip
    add_group full_centered $(seeded evaluation/h1_centered/llama/seed_SEED)
    add_group all_map_mean_centered evaluation/h1_centered/ablations/all_map_mean/seed_42
    add_group clip_centered evaluation/h1_centered/clip
    run h1 python rebuttal.py --output "$OUT" h1 --judge "$JUDGE" --primary full "${H1_ARGS[@]}" \
        --compare full all_map_mean --compare full clip --compare full initial \
        --compare full_cosine full --compare clip_dot clip --compare full_cosine clip --compare full clip_dot \
        --compare full_centered all_map_mean_centered --compare full_centered clip_centered
fi

## H2: Router Weights of Every Full Seed
if want router; then
    NEW_SEEDS=$(for SEED in $SEEDS; do [ "$FORCE" = 0 ] && [ -f "evaluation/router/llama/seed_$SEED.csv" ] || echo -n "$SEED "; done)
    if [ -n "$NEW_SEEDS" ]; then
        run router_weights env CUDA_VISIBLE_DEVICES="$SMALL_GPU" SEEDS="$NEW_SEEDS" ARMS=full FULL_DIR=$FULL_DIR ./router_weights.sh
    fi
    CSVS=()
    for SEED in $SEEDS; do
        [ -f "evaluation/router/llama/seed_$SEED.csv" ] && CSVS+=("evaluation/router/llama/seed_$SEED.csv")
    done
    [ ${#CSVS[@]} -eq 0 ] || run router python rebuttal.py --output "$OUT" router --csv "${CSVS[@]}"
fi

## R10: LLaVA-NeXT-8B with Cambrian-1's Prompt
if want cambrian; then
    MISSING=$(missing_datasets evaluation/baselines/LLaVA-NeXT-8B-Cambrian LLaVA-NeXT-8B-Cambrian "CV-Bench-2D CV-Bench-3D")
    if [ -n "$MISSING" ]; then
        run cambrian env MODELS=LLaVA-NeXT-8B-Cambrian DATASETS="$MISSING" JUDGE="$JUDGE" ./eval_baselines.sh
    else
        banner "cambrian: keeping complete results"
    fi
fi

## R5 Route A: Test-Time Interventions on the Full Model
# Image tokens stay fixed. Zeroing an expert's weight leaves the bias of expert_head, so its tokens become constant
# rather than removed; mean keeps each token's global match; no intervention removes the whole interaction branch.
if want interventions; then
    for MODE in $INTERVENTIONS; do
        for SEED in $(intervention_seeds "$MODE"); do
            RUN=evaluation/interventions/$MODE/llama/seed_$SEED
            MISSING=$(missing_datasets "$RUN" IPMoE)
            if [ -n "$MISSING" ]; then
                run "intervention_${MODE}_$SEED" env IPMOE_INTERVENTION="$MODE" SEEDS="$SEED" ABLATIONS=full FULL_DIR=$FULL_DIR \
                    OUT="evaluation/interventions/$MODE/llama" DATASETS="$MISSING" JUDGE="$JUDGE" ./eval.sh
            else
                banner "intervention_${MODE}_$SEED: keeping complete results"
            fi
        done
    done
fi

## R2, R4: Result Links, Commits and Reruns
# Results are labelled 493f723b, 1d96b902 or later, and inference changed in between (the context limit). Each result's
# own log shows the context actually used, and a run directory holding a benchmark twice shows whether a rerun of the
# same checkpoint reproduced its predictions; full seed 43 was rerun on five benchmarks on 2026-10-29.
if want records; then
    RECORD_RUNS=()
    for RUN in evaluation/llama/seed_* evaluation/qwen/seed_* evaluation/ablations/*/seed_* evaluation/interventions/*/llama/seed_* \
               "${BASELINE_RUNS[@]}"; do
        [ -d "$RUN" ] && RECORD_RUNS+=("$RUN")
    done
    run records python rebuttal.py --output "$OUT" records --judge "$JUDGE" --reference "$REFERENCE" --runs "${RECORD_RUNS[@]}"
fi

## R2, R6, R7: Scores, Checkpoint-Level Intervals and the Thesis Tables
if want stats; then
    STATS_ARGS=(--judge "$JUDGE" --naturalbench-metric "$NB_METRIC")
    COMPLETE_FULL=()
    for RUN in "${FULL_RUNS[@]}"; do
        if complete "$RUN"; then
            COMPLETE_FULL+=("$RUN")
        else
            FAILED+=("stats input $RUN (missing results; see records_results.csv)")
        fi
    done
    COMPLETE_ARMS=()
    for RUN in "${ARM_RUNS[@]}"; do
        complete "$RUN" && COMPLETE_ARMS+=("$RUN")
    done
    COMPLETE_QWEN=()
    for RUN in "${QWEN_RUNS[@]}"; do
        complete "$RUN" && COMPLETE_QWEN+=("$RUN")
    done
    COMPLETE_INTERVENTIONS=()
    for MODE in $INTERVENTIONS; do
        for SEED in $(intervention_seeds "$MODE"); do
            complete "evaluation/interventions/$MODE/llama/seed_$SEED" && COMPLETE_INTERVENTIONS+=("evaluation/interventions/$MODE/llama/seed_$SEED")
        done
    done
    # Baselines with every benchmark the tables use.
    TABLE_BASELINES=()
    for RUN in "${BASELINE_RUNS[@]}"; do
        complete "$RUN" "$(basename "$RUN")" "CV-Bench-2D CV-Bench-3D MMBench_DEV_EN_V11 MMStar SEEDBench_IMG" && TABLE_BASELINES+=("$RUN")
    done

    if [[ " ${COMPLETE_FULL[*]} " != *" $REFERENCE "* ]]; then
        FAILED+=("stats (reference $REFERENCE is incomplete)")
    elif run scores python rebuttal.py --output "$OUT" scores "${STATS_ARGS[@]}" --full "${COMPLETE_FULL[@]}" \
            --runs "${COMPLETE_ARMS[@]}" "${COMPLETE_INTERVENTIONS[@]}" "${COMPLETE_QWEN[@]}"; then
        # Every arm and the other full seeds against full seed 42; the seeds show what training randomness alone gives.
        OTHERS=()
        for RUN in "${COMPLETE_FULL[@]}" "${COMPLETE_ARMS[@]}"; do
            [ "$RUN" = "$REFERENCE" ] || OTHERS+=("$RUN")
        done
        [ ${#OTHERS[@]} -eq 0 ] || run bootstrap_arms python rebuttal.py --output "$OUT" bootstrap "${STATS_ARGS[@]}" \
            --name bootstrap_arms --reference "$REFERENCE" --runs "${OTHERS[@]}" --replicates "$REPLICATES"
        # Each intervention against the unperturbed checkpoint it was applied to.
        for RUN in "${COMPLETE_FULL[@]}"; do
            SEED=${RUN##*seed_}
            PERTURBED=()
            for INTERVENTION in "${COMPLETE_INTERVENTIONS[@]}"; do
                [ "${INTERVENTION##*seed_}" = "$SEED" ] && PERTURBED+=("$INTERVENTION")
            done
            [ ${#PERTURBED[@]} -eq 0 ] || run "bootstrap_interventions_$SEED" python rebuttal.py --output "$OUT" bootstrap "${STATS_ARGS[@]}" \
                --name "bootstrap_interventions_seed_$SEED" --reference "$RUN" --runs "${PERTURBED[@]}" --replicates "$REPLICATES"
        done
        QWEN_GROUP=()
        [ ${#COMPLETE_QWEN[@]} -eq 0 ] || QWEN_GROUP=(--group qwen "${COMPLETE_QWEN[@]}")
        run tables python rebuttal.py --output "$OUT" tables "${STATS_ARGS[@]}" --reference "$REFERENCE" --cue evaluation/h1/clip/items.csv \
            --group full "${COMPLETE_FULL[@]}" "${QWEN_GROUP[@]}" --runs "${COMPLETE_ARMS[@]}" "${TABLE_BASELINES[@]}"
    else
        echo "=== stats: score validation failed; no interval or table was computed" >&2
    fi
fi

## R10: Every Evaluated Baseline
if want baselines; then
    IPMOE_GROUPS=(--ipmoe "IP-MoE (learned), Llama" "${FULL_RUNS[@]}")
    [ ${#QWEN_RUNS[@]} -eq 0 ] || IPMOE_GROUPS+=(--ipmoe "IP-MoE (learned), Qwen" "${QWEN_RUNS[@]}")
    [ -d evaluation/ablations/uniform/seed_42 ] && IPMOE_GROUPS+=(--ipmoe "IP-MoE (uniform), Llama" evaluation/ablations/uniform/seed_42)
    [ ${#BASELINE_RUNS[@]} -eq 0 ] || run baselines python rebuttal.py --output "$OUT" baselines --judge "$JUDGE" \
        --naturalbench-metric "$NB_METRIC" --runs "${BASELINE_RUNS[@]}" "${IPMOE_GROUPS[@]}" \
        --unreported evaluation/baselines/LLaVA-NeXT-8B-Cambrian
fi

if [ -n "$OVERLAP_PID" ]; then
    banner "overlap: waiting for the background job"
    wait "$OVERLAP_PID" || FAILED+=("overlap (see $OUT/overlap.log)")
    tail -n 20 "$OUT/overlap.log"
fi

banner "done"
if [ ${#FAILED[@]} -gt 0 ]; then
    printf 'Failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
