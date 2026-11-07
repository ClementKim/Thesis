# Core interaction-map ablations

`ablation.py` defines the training configurations. Select one with `main.py --ablation NAME`.
Omitting the option selects `full`, preserving the current architecture and defaults.

| Name | Interaction path | Experts and routing |
| --- | --- | --- |
| `full` | Original map throughout | All three, learned routing |
| `uniform` | Original map throughout | All three, fixed weights of 1/3 |
| `static` | Original map throughout | All three, learned input-independent weights |
| `expert_mean` | Spatial mean for experts only | All three, router receives original map |
| `router_mean` | Spatial mean for router only | All three, experts receive original map |
| `both_mean` | Spatial mean for experts and router; direct interaction prefix stays original | All three, learned routing |
| `image_only` | No interaction prefix or expert prefix | No experts or router |
| `all_map_mean` | Spatial mean before the adapter, shared by every interaction-map consumer | All three, learned routing |
| `self_only` | Original map throughout | One self-attention expert, weight 1 |
| `cross_only` | Original map throughout | One cross-attention expert, weight 1 |
| `hybrid_only` | Original map throughout | One hybrid expert, weight 1 |

The spatial mean is computed over all 576 visual positions separately for each
text token, then broadcast back to the original shape. `all_map_mean` applies this
to the complete interaction matrix before the interaction adapter, so the direct
interaction prefix, experts, and router all receive the averaged map. It keeps the
projected image prefix. In contrast, `both_mean` leaves the direct interaction
prefix unchanged. The optional `shared_map` setting defaults to `full`.

Let `N = ceil(L / patch_size) * ceil(576 / patch_size)`, where `L` is the retained
CLIP question-token count (at least one for the empty-question placeholder).
Full and mean-map configurations have `576 + 4N` valid prefix tokens. Single-expert
configurations have `576 + 2N`; `image_only` has 576. Padding masks follow these
lengths. Shorter prefixes may retain more training or validation examples under the
context limit, so compare the saved `data_counts` as well as the training settings.

Uniform and static controls keep the original dynamic router object frozen and
unused. This preserves initialization of subsequent modules without leaving unused
trainable parameters. Static logits start at zero, so static routing initially
matches uniform routing.

The optional `expert_set` defaults to `all`. Single-expert configurations use uniform
routing over their one active expert, giving it weight 1 without running the router.
All three expert modules remain registered and are constructed in the original
order; unused experts and the router are frozen. `self_only` also freezes the unused
auxiliary memory. This preserves shared initialization and module names, while
reducing active trainable parameters and prefix length. It does not reduce the
memory allocated for the registered modules. Compare these runs as separately
trained single-expert alternatives; they do not isolate architectural diversity at
equal capacity. Router diagnostics retain three columns, with a one-hot weight for
single-expert runs.

## Running and loading

Use the same data seed, patch size, backbone snapshots, environment, context limit,
effective batch size, and training budget across comparisons. Each new configuration
is trained from its initialization, not loaded as an inference-time intervention on
an existing full checkpoint. No multi-run launcher is invoked by importing
`ablation.py`.

To train only the four additional configurations with seed 42 and the same Qwen
backbone as the corresponding full run:

```bash
SEEDS=42 LLM_TYPE=qwen \
ABLATIONS="all_map_mean self_only cross_only hybrid_only" ./run.sh
```

Set `LLM_TYPE=llama` instead when comparing with a Llama full run. `run.sh` defaults
to Qwen, seeds 43 and 44, and every registered ablation, so select the subset
explicitly. The checkpoint paths do not include the backbone: do not write different
backbones into the same arm/seed directory.

Without `--output-dir`, `full` retains the original `checkpoint` directory. Other
arms default to `checkpoint/ablations/NAME/seed_SEED` to avoid overwriting the
default full-model checkpoint. The launcher explicitly uses `checkpoint/seed_SEED`
for `full`.

Checkpoints save the ablation name, version, and resolved configuration in `args`.
The metadata version remains 1, and existing configuration dictionaries are unchanged.
The evaluation adapter validates this metadata before constructing the model and
retains its strict parameter checks. Current-architecture checkpoints with no
ablation metadata reconstruct as `full`; partially present or inconsistent metadata
is rejected. A checkpoint must still contain `num_aux_tokens`.

Select a checkpoint explicitly when evaluating an arm:

```bash
PYTHONHASHSEED=42 PYTHONPATH="$PWD/VLMEvalKit${PYTHONPATH:+:$PYTHONPATH}" \
IPMOE_CHECKPOINT="$PWD/checkpoint/ablations/all_map_mean/seed_42/checkpoint.pt" \
python eval.py --model IPMoE --data CV-Bench-2D \
    --work-dir evaluation/ablations/all_map_mean/seed_42 --judge exact_matching
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
check baseline behavior, shared initialization, map averaging, single-expert
selection, prefix lengths, masking, gradient coverage, and checkpoint reconstruction.
They do not download pretrained weights, evaluate benchmarks, or run full training.
Backbone revision pinning and a full training-resume checkpoint are outside this change.
