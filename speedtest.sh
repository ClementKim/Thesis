#!/usr/bin/bash
# Short training runs that measure (1) DataLoader workers and (2) gradient checkpointing vs smaller batches.
# Each run trains one epoch on SAMPLES samples; compare samples/s and peak GPU memory in the summary.
# Override with environment variables, e.g. SAMPLES=3000 ./speedtest.sh
# The first run also builds the capped cache for this protocol, which can take a while; it is not timed.

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
export TOKENIZERS_PARALLELISM=false

SAMPLES=${SAMPLES:-6000}
SEED=${SEED:-42}
ROOT=checkpoint/speedtest
mkdir -p "$ROOT"

# name, workers, batch size, gradient checkpointing flag
RUNS=(
    "workers0_batch6_ckpt 0 6 --gradient-checkpointing"
    "workers4_batch6_ckpt 4 6 --gradient-checkpointing"
    "workers4_batch3_nockpt 4 3 --no-gradient-checkpointing"
    "workers4_batch2_nockpt 4 2 --no-gradient-checkpointing"
)

for RUN in "${RUNS[@]}"; do
    read -r NAME WORKERS BATCH CKPT <<< "$RUN"
    OUTPUT_DIR=$ROOT/$NAME
    mkdir -p "$OUTPUT_DIR"

    # Record peak GPU memory every 2 seconds, only on the GPUs this job can see.
    nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i $CUDA_VISIBLE_DEVICES} --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$OUTPUT_DIR/memory.csv" &
    MONITOR=$!

    echo "=== $(date '+%F %T') $NAME"
    PYTHONHASHSEED="$SEED" torchrun --standalone --nproc-per-node=gpu main.py \
        --seed $SEED \
        --max-train-samples $SAMPLES \
        --num-workers $WORKERS \
        --batch-size $BATCH \
        $CKPT \
        --attn-implementation flash_attention_2 \
        --output-dir $OUTPUT_DIR 2>&1 | tee "$OUTPUT_DIR/train.log"
    echo "${PIPESTATUS[0]}" > "$OUTPUT_DIR/exit_code"

    kill $MONITOR 2> /dev/null
    wait $MONITOR 2> /dev/null
done

## Summary
python - "$ROOT" "${RUNS[@]}" <<'EOF'
import math, os, re, sys

import torch

root, runs = sys.argv[1], sys.argv[2:]
# Same count torchrun uses for --nproc-per-node=gpu, respecting CUDA_VISIBLE_DEVICES.
gpus = torch.cuda.device_count()

def seconds(text):
    value = 0
    for part in text.split(':'):
        value = value * 60 + int(part)
    return value

print(f'\n{"run":26s} {"exit":>4} {"steps":>6} {"s/step":>7} {"samples/s":>10} {"peak GiB":>9}')
for run in runs:
    name, _, batch, _ = run.split()
    directory = os.path.join(root, name)
    log = open(os.path.join(directory, 'train.log'), errors = 'replace').read().replace('\r', '\n')
    code = open(os.path.join(directory, 'exit_code')).read().strip()

    memory = [int(value) for value in open(os.path.join(directory, 'memory.csv')).read().split() if value.isdigit()]
    peak = f'{max(memory) / 1024:.1f}' if memory else '-'

    # The training bar is the one whose total matches this run's optimizer steps per epoch.
    split = re.search(r'split: train=([\d,]+)', log)
    if split is None:
        print(f'{name:26s} {code:>4} {"-":>6} {"-":>7} {"-":>10} {peak:>9}  (no split line; see train.log)')
        continue

    train = int(split.group(1).replace(',', ''))
    steps = math.ceil(math.ceil(train / gpus) / int(batch))
    bars = re.findall(rf'100%\|[^|]*\| {steps}/{steps} \[([\d:]+)<', log)
    if not bars:
        print(f'{name:26s} {code:>4} {steps:>6} {"-":>7} {"-":>10} {peak:>9}  (epoch did not finish; out of memory?)')
        continue

    elapsed = max(seconds(bar) for bar in bars)
    print(f'{name:26s} {code:>4} {steps:>6} {elapsed / steps:7.2f} {train / elapsed:10.1f} {peak:>9}')

print('\nCompare samples/s: rows 1-2 measure workers; rows 2-4 measure checkpointing vs batch size.')
print('A peak near 40 GiB means the setting may still run out of memory on long batches.')
EOF
