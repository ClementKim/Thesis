# Core interaction-map ablations

`ablation.py` defines the six experiments. Select one with `main.py --ablation NAME`.
Omitting the option selects `full`, preserving the current architecture and defaults.

| Name | Expert input | Router |
| --- | --- | --- |
| `full` | Original map | Learned from the original map |
| `uniform` | Original map | Fixed weights of 1/3 |
| `static` | Original map | Learned input-independent weights |
| `expert_mean` | Spatial mean | Learned from the original map |
| `router_mean` | Original map | Learned from the spatial mean |
| `both_mean` | Spatial mean | Learned from the spatial mean |

The spatial mean is computed over all 576 visual positions separately for each
text token, then broadcast back to the original shape. The complete interaction
matrix is computed first. The direct image and interaction prefixes are unchanged.
Every arm keeps the same padding masks and `576 + 4N` valid prefix tokens, where
`N = ceil(L / patch_size) * ceil(576 / patch_size)` and `L` is the retained CLIP
question-token count (at least one for the empty-question placeholder).

Uniform and static controls keep the original dynamic router object frozen and
unused. This preserves initialization of subsequent modules without leaving unused
trainable parameters. Static logits start at zero, so static routing initially
matches uniform routing. These controls have fewer active trainable router
parameters; they are not identical in trainable parameter count.

## Running and loading

Use the same data seed, patch size, backbone snapshots, environment, context limit,
effective batch size, and training budget across comparisons. The existing scripts
continue to run `full` unless changed explicitly. No multi-run launcher is invoked
by importing `ablation.py`.

For a future run, choose the arm and an explicit output directory, for example:

```bash
PYTHONHASHSEED=42 python main.py --ablation expert_mean --seed 42 \
    --output-dir checkpoint/ablations/expert_mean/seed_42
```

Without `--output-dir`, `full` retains the original `checkpoint` directory. Other
arms default to `checkpoint/ablations/NAME/seed_SEED` to avoid overwriting the
default full-model checkpoint.

Checkpoints save the ablation name, version, and resolved configuration in `args`.
The evaluation adapter validates this metadata before constructing the model and
retains its strict parameter checks. Current-architecture checkpoints with no
ablation metadata reconstruct as `full`; partially present or inconsistent metadata
is rejected. A checkpoint must still contain `num_aux_tokens`.

Select a checkpoint explicitly when evaluating an arm:

```bash
PYTHONHASHSEED=42 PYTHONPATH="$PWD/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}" \
IPMOE_CHECKPOINT="$PWD/checkpoint/ablations/expert_mean/seed_42/checkpoint.pt" \
python eval.py --model IPMoE --data CV-Bench-2D \
    --work-dir evaluation/ablations/expert_mean/seed_42 --judge exact_matching
```

If `IPMOE_CHECKPOINT` is unset, evaluation retains the existing
`checkpoint/seed_SEED/checkpoint.pt` fallback. Existing evaluation status and
prediction-completeness checks remain enabled.

## Verification

Run the offline tests in an environment with the project's model dependencies:

```bash
python -B -m unittest discover -s tests -p 'test_ablation.py' -v
```

These tests use small real connector modules and synthetic backbone outputs to
check baseline behavior, paired initialization, prefix lengths, masking, gradient
coverage, and checkpoint reconstruction through the actual evaluation adapter.
They do not download pretrained weights, evaluate benchmarks, or run full training.
Backbone revision pinning and a full training-resume checkpoint are outside this change.
