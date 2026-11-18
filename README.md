# IP-MoE (Draft)

Code for the master's thesis *A Study of Spatial Token–Patch Interactions and Soft Dense Expert Weighting in
Vision–Language Models*.

IP-MoE tests whether an explicit map of the interactions between question tokens and image patches can usefully
condition a mixture of experts in a vision–language model. This repository holds the model, the training loop, the
benchmark evaluation through VLMEvalKit, the external baselines and the ablation runner used in the thesis.

## Model

1. **Encoders** ([model/encoder.py](model/encoder.py)). Frozen `openai/clip-vit-large-patch14-336` encodes the image
   (penultimate-layer patch tokens, a 24 × 24 grid of 576 patches) and the question (chunked to fit CLIP's text
   length). A trainable MLP on each side projects the features to `--embed-dim`.
2. **Interaction map** ([model/interaction.py](model/interaction.py)). The bilinear product of the text and patch
   features gives a `(question tokens × 576)` matrix that is *not* pooled over image positions. It is patchified into
   interaction tokens.
3. **Experts** ([model/experts.py](model/experts.py)). Three 8-layer transformers read the interaction map:
   a self-attention expert, a cross-attention expert that attends to learned auxiliary tokens, and a hybrid that
   alternates the two.
4. **Router** ([model/router.py](model/router.py)). An attention-pooled gate over every token–patch cell of the map
   gives a softmax weight per expert. Each expert's output is scaled by its weight and kept as separate tokens.
5. **Language model** ([model/llm.py](model/llm.py)). The image tokens, interaction tokens and weighted expert tokens
   are projected into the embedding space of a LoRA-adapted decoder and prepended to the question. Supported decoders:

   | `--llm-type` | `--llm-size` | Model |
   |---|---|---|
   | `llama` | 1, 3 | `meta-llama/Llama-3.2-{1,3}B-Instruct` |
   | `qwen` | 3, 7 | language model of `Qwen/Qwen2.5-VL-{3,7}B-Instruct` (its vision encoder is unused) |

[model/ipmoe.py](model/ipmoe.py) assembles these into `IPMoE`.

## Repository layout

| Path | Purpose |
|---|---|
| [main.py](main.py) | Training (single GPU or `torchrun` DDP); writes `checkpoint.pt` after each epoch |
| [preprocessing.py](preprocessing.py) | LLaVA-NeXT-Data download, filtering, caching and train/validation split |
| [model/](model/) | IP-MoE modules (see above) |
| [eval.py](eval.py) | VLMEvalKit entry point; registers IP-MoE and the baselines and fails on any incomplete benchmark |
| [vlmeval_adapter.py](vlmeval_adapter.py) | VLMEvalKit wrapper for an IP-MoE checkpoint, prompts, and test-time interventions |
| [vlmeval_baselines.py](vlmeval_baselines.py) | External baseline models evaluated with the same protocol |
| [router_weights.py](router_weights.py) | Records the router's expert weights for every benchmark question |
| [test.py](test.py) | Ablation arms: train, evaluate and compare each against the full model |
| [run.sh](run.sh), [eval.sh](eval.sh), [eval_baselines.sh](eval_baselines.sh), [test.sh](test.sh), [router_weights.sh](router_weights.sh), [speedtest.sh](speedtest.sh) | Drivers for the experiments below |

`dataset/`, `checkpoint/`, `evaluation/` and `VLMEvalKit/` are created locally and are not tracked.

## Setup

**Hardware.** A CUDA GPU with bfloat16 support. Training uses FlashAttention 2 and was sized for 40 GB GPUs
(see [speedtest.sh](speedtest.sh)); all scripts launch one process per visible GPU.

**Environment.**

```bash
# 1. Install a CUDA-enabled PyTorch (and torchvision for evaluation) that matches your CUDA toolkit.
# 2. Training only:
pip install -r requirements.txt
# 3. Training and evaluation: place VLMEvalKit at ./VLMEvalKit first, then
pip install -r vlmeval_requirements.txt
```

[eval.py](eval.py) hooks into internals of VLMEvalKit's `run.py`, so use the VLMEvalKit copy this code was developed
against; another version may not work.

**Model access.** The Llama 3.2 models are gated on Hugging Face; accept the license and run `huggingface-cli login`.

**LLM judge.** By default evaluation uses `gpt-6-luna` to extract answers that rule-based matching cannot parse.
Put `OPENAI_API_KEY=...` in `VLMEvalKit/.env`, or set `JUDGE=exact_matching` to score without an LLM.

## Data

Training uses the first user/assistant turn of each row of
[`lmms-lab/LLaVA-NeXT-Data`](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data), pinned to a fixed revision
(250 parquet shards, 779,289 rows). On first use [preprocessing.py](preprocessing.py) downloads the shards, writes the
images to `dataset/llava-next/images/` and a processed cache to `dataset/cache/`. The validation split is grouped by
identical image bytes (10%, at most 2,000 samples, fixed seed 42) so no image appears in both splits.

The first build processes the whole source and can take a long time. Under DDP only rank 0 builds it, but you can
also prepare it ahead of training:

```bash
python preprocessing.py
```

Changing the filtering rules requires bumping `PREPROCESSING_PROTOCOL`, which rebuilds the cache.

## Training

```bash
SEEDS="42 43 44" LLM_TYPE=qwen ./run.sh
```

[run.sh](run.sh) trains the full model for each seed with `torchrun` on all visible GPUs (batch size 6 per GPU) into
`checkpoint/<LLM_TYPE>/seed_<N>/` (`checkpoint.pt`, `train.log`). A seed that already has a checkpoint is skipped;
`RETRAIN=1` trains it again. Choose GPUs with `CUDA_VISIBLE_DEVICES`.

The checkpoint stores only trainable parameters (projections, experts, router, LoRA) together with the full argument
list, data counts and validation loss. The main options of [main.py](main.py):

| Option | Default | Meaning |
|---|---|---|
| `--llm-type`, `--llm-size` | `llama`, `3` | Decoder (see table above) |
| `--lora-rank` | `64` | LoRA rank on the decoder's attention and MLP projections |
| `--embed-dim`, `--num-heads` | `1024`, `16` | Width of the projected CLIP features and experts |
| `--patch-size` | `16` | Patch size used to tokenize the interaction map |
| `--num-aux-tokens` | `32` | Learned auxiliary tokens for the cross and hybrid experts |
| `--context-limit` | `4096` | Total token budget per training example (prefix + text) |
| `--batch-size` | `1` | Per-GPU batch size |
| `--epochs`, `--learning-rate`, `--warmup-ratio` | `1`, `1e-4`, `0.03` | AdamW with linear warmup and cosine decay |
| `--max-train-samples` | all | Cap on training samples (validation is unaffected) |
| `--seed`, `--data-seed` | `42`, `42` | Model/dropout seed and training-subset seed |

Runs are deterministic: [main.py](main.py) enables deterministic algorithms, and the scripts set
`CUBLAS_WORKSPACE_CONFIG`, `FLASH_ATTENTION_DETERMINISTIC` and `PYTHONHASHSEED`.

## Evaluation

```bash
LLM_TYPE=qwen SEEDS="42 43 44" ./eval.sh
```

[eval.sh](eval.sh) evaluates `checkpoint/<LLM_TYPE>/seed_<N>/checkpoint.pt` on seven benchmarks and writes results and
`eval.log` to `evaluation/<LLM_TYPE>/seed_<N>/`:

`NaturalBenchDataset`, `CV-Bench-2D`, `CV-Bench-3D`, `MMBench_DEV_EN_V11`, `MMStar`, `POPE`, `SEEDBench_IMG`

NaturalBench uses its own answer extraction, so it always runs as a separate exact-matching group. Variables:

| Variable | Default | Meaning |
|---|---|---|
| `SEEDS` | `42` | Seeds to evaluate |
| `DATASETS` | the seven above | Subset of benchmarks |
| `JUDGE` | `gpt-6-luna` | Answer-extraction judge, or `exact_matching` |
| `LLM_TYPE` | `llama` | Selects `FULL_DIR=checkpoint/<LLM_TYPE>` and `OUT=evaluation/<LLM_TYPE>` |
| `IPMOE_INTERVENTION` | none | Test-time intervention (below); give it its own `OUT` |
| `IPMOE_CONTEXT_LIMIT` | decoder capacity | Smaller total token budget at inference |

Extra arguments go to [eval.py](eval.py), e.g. `./eval.sh --judge-api-nproc 8`.

Test-time interventions ([vlmeval_adapter.py](vlmeval_adapter.py)) modify a trained checkpoint without retraining:

| `IPMOE_INTERVENTION` | Effect |
|---|---|
| `swap` | Build the interaction map from a different image of the same benchmark; the image tokens keep the true image |
| `mean` | Average the interaction map over image positions before the prefix, experts and router |
| `uniform` | Replace the router's weights with 1/3 each |
| `zero_self`, `zero_cross`, `zero_hybrid` | Set one expert's weight to 0 without renormalizing the others |

### Baselines

```bash
MODELS="SpatialLadder-3B Spatial-SSRL-3B" ./eval_baselines.sh
```

[eval_baselines.sh](eval_baselines.sh) runs the models in [vlmeval_baselines.py](vlmeval_baselines.py) with the same
benchmarks, judge and checks, writing to `evaluation/baselines/<model>/`: BLIP2-FlanT5-XL, InstructBLIP-Vicuna-7B,
Mono-InternVL-2B, Qwen2.5-VL-3B-Instruct, SpatialLadder-3B, Spatial-SSRL-3B, LLaVA-1.5-7B, LLaVA-NeXT-8B and
LLaVA-NeXT-8B with Cambrian-1's CV-Bench prompt. All models run by default.

## Ablations

[test.sh](test.sh) trains, evaluates and compares one ablation arm against the full model, seed by seed:

```bash
ARM=without_self LLM_TYPE=qwen SEEDS="42 43 44" ./test.sh       # train, eval, compare
ARM=all LLM_TYPE=qwen ./test.sh                                  # every arm, full model first
```

| `ARM` | Prefix given to the decoder |
|---|---|
| `full` | Image + interaction + three routed experts (the reference) |
| `image_only` | Image only; no interaction or experts |
| `no_moe` | Image + interaction; no router or experts |
| `self_only`, `cross_only`, `hybrid_only` | Image + interaction + one expert with weight 1 |
| `without_self`, `without_cross`, `without_hybrid` | Image + interaction + the other two experts with a two-way router |

Steps (`STEPS`, default `train eval compare`):

- **train** writes `checkpoint/<ARM>/<LLM_TYPE>/seed_<N>/`. A new attempt replaces the current run only once it has
  finished. With `MATCH=1` (default) an arm trains only if the full model's checkpoint for that seed exists and its
  settings, GPU count and data match, so keep the GPU count and global batch size the same as the full model's.
- **eval** writes `evaluation/<ARM>/<LLM_TYPE>/seed_<N>/` and evaluates only benchmarks without a completed result for
  the current checkpoint (`REEVALUATE=1` redoes them all). Default benchmarks: CV-Bench-2D, CV-Bench-3D, POPE.
- **compare** writes the arm's scores, validation loss and training counters beside the full model's, with a 95%
  paired t interval over seeds, to `evaluation/<ARM>/<LLM_TYPE>/comparison/`.

The full model is read from `FULL_CKPT_DIR`/`FULL_EVAL_DIR` (`checkpoint/<LLM_TYPE>` and `evaluation/<LLM_TYPE>`).
[test.py](test.py) can also be called directly (`train`, `eval`, `compare`, `state`, `pending`); its docstring
describes each command. Commit [test.py](test.py) and [test.sh](test.sh) before running, since evaluation records only
the commit.

## Router analysis

```bash
FULL_DIR=checkpoint/qwen SEEDS="42 43 44" ./router_weights.sh
```

[router_weights.sh](router_weights.sh) runs only the prefix encoder of a checkpoint on every benchmark question and
writes the router's `self`/`cross`/`hybrid` weights, one row per question with its benchmark category labels, to
`evaluation/router/<llm>/seed_<N>.csv`. The log ends with the mean, standard deviation, minimum and maximum per
benchmark; a router that ignores its input has a spread near zero. It loads the model on one GPU.

## Throughput test

[speedtest.sh](speedtest.sh) trains on `SAMPLES` examples (default 6,000) with different DataLoader worker counts,
batch sizes and gradient-checkpointing settings and prints samples per second and peak GPU memory for each.

## Output layout

```
checkpoint/<llm>/seed_<N>/checkpoint.pt, train.log         # run.sh (full model)
checkpoint/<arm>/<llm>/seed_<N>/                           # test.sh
evaluation/<llm>/seed_<N>/eval.log, <VLMEvalKit results>   # eval.sh
evaluation/<arm>/<llm>/seed_<N>/, comparison/              # test.sh
evaluation/baselines/<model>/                              # eval_baselines.sh
evaluation/router/<llm>/seed_<N>.csv, .log                 # router_weights.sh
```

Every `train.log` and `eval.log` entry starts with a line naming the configuration, code commit and GPUs it ran with.

## Findings

In the thesis, none of the four hypotheses about the interaction map and the router was supported: the trained map
does not localize the queried objects, the learned router performs no better than uniform weights, single experts are
at least as accurate as the mixture, and a model given only the image tokens reaches the full model's average
accuracy (one seed). See the thesis for the full results and analysis.
