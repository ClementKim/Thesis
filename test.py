"""Nine IP-MoE setups, each represented by its own class in ARM_CLASSES.

Select --arm full, image_only, no_moe, self_only, cross_only, hybrid_only,
without_self, without_cross, or without_hybrid. Training/state/compare default to no_moe;
eval/pending infer the setup from the checkpoint and validate an explicit --arm if supplied.
Each reduced setup is trained independently; without_* retains two experts with a two-way softmax.

train    main.py's training loop and checkpoint format, with the selected class in place of IPMoE. Every module is
         built in the full model's order before unused branches are dropped, so kept modules start from its initialization
         for the same seed. Each example gets the text budget the full model gives it under --context-limit, so the arm
         trains and validates on exactly the examples the full model keeps (the thesis's full runs skipped none at 4096
         tokens, so this equals budgeting with the arm's shorter prefix). --match FULL_CHECKPOINT refuses to start
         unless the reference finished training and its seed, settings, GPU count and data match. Recorded runtime
         versions are checked before construction, loaded backbone revisions before training, and actual optimizer
         steps after training. Missing historical provenance is explicitly reported as unverified.
eval     eval.py's VLMEvalKit evaluation of IPMOE_CHECKPOINT through the same adapter, so prompts, context limit,
         generation and checks are the full model's. It always predicts and scores afresh (no --reuse, --mode all).
         Each run directory it creates gets an <arm>.json naming the checkpoint and its SHA-256, marked completed only
         once eval.py's checks have passed. Give runs under IPMOE_INTERVENTION (swap or mean) their own --work-dir, as
         eval.sh does: pending and compare never read them.
compare  the arm's scores, validation loss and training counters beside the full model's, paired by seed, with a 95%
         paired t interval over seeds. The arm's scores come only from completed runs of its current checkpoint. The
         legacy full-model runs can be checked only against status.json (the run's status and judge), since eval.py
         keeps no verdict of its own. Seeds and completed training budgets are validated before statistics; a
         verification.json records which provenance fields could not be verified.
state    whether a checkpoint is a finished run of this arm, and with --match whether its settings, recorded identity
         and completed training budget match Full.
pending  the benchmarks an evaluation directory has no usable result of the arm's current checkpoint for.

test.sh runs these for every seed; for example ARM=without_self LLM_TYPE=qwen ./test.sh.
Run from the repository root. Existing no_moe checkpoints and markers remain supported.
Original main-branch non-full ablation checkpoints use a different format and are not loaded by this runner.
"""

import sys

# `test.py eval` runs under torchrun as eval.py does. Importing VLMEvalKit's run module assigns each rank its GPUs, which
# must happen before Torch or the model is imported. No other command may import it: main.py assigns the training GPUs.
if __name__ == '__main__' and sys.argv[1:2] == ['eval']:
    import VLMEvalKit.run  # noqa: F401

import argparse
import functools
import glob
import hashlib
import json
import logging
import math
import os
import subprocess
from importlib import metadata

import torch

import main
from model.ipmoe import IPMoE
from model.llm import NoValidTargetsError
from preprocessing import PREPROCESSING_PROTOCOL, SOURCES

DEFAULT_ARM = 'no_moe'

# The seven benchmarks of the thesis's Avg, in eval.sh's order, and the NaturalBench score it uses.
DATASETS = ('NaturalBenchDataset', 'CV-Bench-2D', 'CV-Bench-3D', 'MMBench_DEV_EN_V11', 'MMStar', 'POPE', 'SEEDBench_IMG')
NATURALBENCH_METRIC = 'Acc'
# Map interventions supported by every setup that uses the interaction path.
INTERVENTIONS = ('swap', 'mean')
# Two-sided 95% quantiles of Student's t by degrees of freedom, for intervals over a few seeds without scipy.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

# Saved arguments that may differ from the full model's run without changing what is learned: where and under which
# name a run is saved, how its batches are loaded and recomputed, and the removed ablation code's metadata.
# Provenance is compared separately: runtime/revisions must match when recorded; code IDs may differ for an ablation.
UNCOMPARED = {'output_dir', 'num_workers', 'gradient_checkpointing', 'arm', 'provenance', 'ablation', 'ablation_version', 'ablation_config'}
RUNTIME_FIELDS = ('python', 'cuda', 'torch', 'transformers', 'peft', 'accelerate', 'flash-attn')
SNAPSHOT_FIELDS = ('clip', 'llm')
# Checkpoint fields besides the arguments that must match too: the process layout and the training data.
COMPARED = ('world_size', 'global_batch_size', 'protocol', 'source')
# main.py's data_counts shown by compare.
COUNTS = ('train_split', 'train_retained_this_epoch', 'train_budget_skipped', 'val_split', 'val_retained', 'val_budget_skipped')
UNSET = '<unset>'

_EXPERT_NAMES = ('self', 'cross', 'hybrid')


class _SelectedExperts(torch.nn.Module):
    """Run one or two retained experts, preserving their original parameter names."""

    def __init__(self, moe, active_experts):
        super().__init__()
        self.active_experts = tuple(active_experts)
        self.indices = tuple(_EXPERT_NAMES.index(name) for name in self.active_experts)

        for name in _EXPERT_NAMES:
            expert = getattr(moe, f'{name}_expert')
            if name in self.active_experts:
                setattr(self, f'{name}_expert', expert)
            else:
                # main.run_training still holds the original MoE until training finishes.
                expert.to('meta')

        if len(self.active_experts) == 1:
            moe.router.to('meta')
        else:
            self.router = moe.router
            # Keep the selected rows of the full router's initialization. Its final
            # softmax now has exactly two logits, with no inactive output parameter.
            head = self.router.gate[-1]
            head.weight = torch.nn.Parameter(head.weight[list(self.indices)].detach().clone(),
                                             requires_grad = head.weight.requires_grad)
            if head.bias is not None:
                head.bias = torch.nn.Parameter(head.bias[list(self.indices)].detach().clone(),
                                               requires_grad = head.bias.requires_grad)
            head.out_features = len(self.active_experts)

    def forward(self, x, auxiliary, mask = None):
        if len(self.active_experts) == 1:
            selected_weights = x.new_ones((x.size(0), 1))
        else:
            selected_weights = self.router(x, mask)

        outputs, masks = [], []
        for name in self.active_experts:
            expert = getattr(self, f'{name}_expert')
            output, output_mask = expert(x, mask) if name == 'self' else expert(x, auxiliary, mask)
            outputs.append(output)
            masks.append(output_mask)

        outputs = torch.stack(outputs, dim = 1)
        masks = torch.stack(masks, dim = 1)
        outputs = (outputs * selected_weights[:, :, None, None]).flatten(1, 2)

        # Diagnostic columns always mean Self/Cross/Hybrid, even when some are absent.
        weights = selected_weights.new_zeros((x.size(0), 3))
        weights[:, list(self.indices)] = selected_weights
        return outputs, masks.flatten(1, 2), weights


class _AblationIPMoE(IPMoE):
    """Shared prefix assembly and full-model text budgeting for reduced setups."""

    active_experts = ()
    keep_interaction = True
    prefix_factor = 1

    def __init__(self, *args, **kwargs):
        # Construct the full model first so every retained module starts from the
        # corresponding full-model initialization for the same seed.
        super().__init__(*args, **kwargs)

        if self.active_experts:
            self.moe = _SelectedExperts(self.moe, self.active_experts)
            if self.active_experts == ('self',):
                del self.aux
        else:
            self.moe.to('meta')
            del self.moe, self.aux, self.expert_head

        if not self.keep_interaction:
            # Prefix-length prediction and provenance still read these objects.
            # In particular, text_encoder.clip is shared with vision_encoder.clip,
            # so moving the text encoder to meta would destroy the image backbone.
            self.text_encoder.requires_grad_(False)
            self.interaction.requires_grad_(False)
            self.interaction_adapter.requires_grad_(False)
            del self.interaction_head

    def encode_prefix(self, text_inputs, image_inputs, map_image_inputs = None):
        image_features = self.vision_encoder(**image_inputs)
        image_mask = torch.ones(image_features.shape[:2], dtype = torch.bool, device = image_features.device)
        image_prefix = self.image_head(image_features)
        weights = image_features.new_empty((image_features.size(0), 0))
        if not self.keep_interaction:
            return image_prefix, image_mask, weights

        text_features, text_mask = self.text_encoder(**text_inputs)
        map_features = image_features if map_image_inputs is None else self.vision_encoder(**map_image_inputs)
        similarity = self.interaction(text_features, map_features)
        interaction_tokens, interaction_map, map_mask, token_mask = self.interaction_adapter(similarity, text_mask)

        prefix = [image_prefix, self.interaction_head(interaction_tokens)]
        prefix_masks = [image_mask, token_mask]
        if self.active_experts:
            outputs, expert_mask, weights = self.moe(interaction_map, getattr(self, 'aux', None), map_mask)
            prefix.append(self.expert_head(outputs))
            prefix_masks.append(expert_mask)

        return torch.cat(prefix, dim = 1), torch.cat(prefix_masks, dim = 1), weights

    def _prefix_lengths(self, text_inputs, image_inputs):
        vision = self.vision_encoder.clip.config.vision_config
        image_tokens = (int(vision.image_size) // int(vision.patch_size)) ** 2
        full_lengths = IPMoE._prefix_lengths(self, text_inputs, image_inputs)
        return [image_tokens + self.prefix_factor * ((length - image_tokens) // 4) for length in full_lengths]

    def forward(self, questions, text_inputs, image_inputs, answers, context_limit, return_stats = False, allow_empty = False):
        """Retain exactly the examples that fit the full model's training budget."""
        prefix_lengths = self._prefix_lengths(text_inputs, image_inputs)
        text_budget = [context_limit - length for length in IPMoE._prefix_lengths(self, text_inputs, image_inputs)]

        try:
            token_rows, label_rows, kept_rows, positions = self.llm.tokenize_training_chats(questions, answers, max_total_tokens = text_budget)
        except NoValidTargetsError:
            if not allow_empty:
                raise

            # Empty ranks must participate in every DDP parameter reduction.
            loss = sum(parameter.reshape(-1)[0].float() * 0 for parameter in self.parameters() if parameter.requires_grad)
            weights = self.image_head.weight.new_empty((0, 3 if self.active_experts else 0))
            return (loss, weights, 0) if return_stats else (loss, weights)

        def select(inputs):
            return {key: value.index_select(0, torch.tensor(kept_rows, device = value.device, dtype = torch.long)) for key, value in inputs.items()}

        prefix, prefix_mask, weights = self.encode_prefix(select(text_inputs), select(image_inputs))
        if prefix_mask.sum(dim = 1).tolist() != [prefix_lengths[index] for index in kept_rows]:
            raise RuntimeError('Predicted prefix lengths disagree with encoded prefix lengths')

        embeds, mask, position_ids, labels = self.llm.build_inputs(token_rows, positions, prefix, label_rows, prefix_mask = prefix_mask)
        loss = self.llm(embeds, mask, position_ids, labels)
        return (loss, weights, (labels[:, 1:] != -100).sum().item()) if return_stats else (loss, weights)


class Full(IPMoE):
    """Image, direct interaction prefix, and all three dynamically weighted experts."""

    arm_name = 'full'
    active_experts = _EXPERT_NAMES
    keep_interaction = True
    prefix_factor = 4


class ImageOnly(_AblationIPMoE):
    """Image prefix and the original question, with no interaction or expert prefix."""

    arm_name = 'image_only'
    keep_interaction = False
    prefix_factor = 0


class NoMoE(_AblationIPMoE):
    """Image and direct interaction prefixes, with no router or experts."""

    arm_name = 'no_moe'


class SelfOnly(_AblationIPMoE):
    """Image and interaction prefixes plus the self-attention expert, weighted by 1."""

    arm_name = 'self_only'
    active_experts = ('self',)
    prefix_factor = 2


class CrossOnly(_AblationIPMoE):
    """Image and interaction prefixes plus the cross-attention expert, weighted by 1."""

    arm_name = 'cross_only'
    active_experts = ('cross',)
    prefix_factor = 2


class HybridOnly(_AblationIPMoE):
    """Image and interaction prefixes plus the hybrid expert, weighted by 1."""

    arm_name = 'hybrid_only'
    active_experts = ('hybrid',)
    prefix_factor = 2


class WithoutSelf(_AblationIPMoE):
    """Retain cross and hybrid experts with a learned two-way softmax."""

    arm_name = 'without_self'
    active_experts = ('cross', 'hybrid')
    prefix_factor = 3


class WithoutCross(_AblationIPMoE):
    """Retain self and hybrid experts with a learned two-way softmax."""

    arm_name = 'without_cross'
    active_experts = ('self', 'hybrid')
    prefix_factor = 3


class WithoutHybrid(_AblationIPMoE):
    """Retain self and cross experts with a learned two-way softmax."""

    arm_name = 'without_hybrid'
    active_experts = ('self', 'cross')
    prefix_factor = 3


ARM_CLASSES = {setup.arm_name: setup for setup in
               (Full, ImageOnly, NoMoE, SelfOnly, CrossOnly, HybridOnly, WithoutSelf, WithoutCross, WithoutHybrid)}


def arm_argument(parser, default = DEFAULT_ARM):
    parser.add_argument('--arm', choices = tuple(ARM_CLASSES), default = default,
                        help = 'model setup (default: no_moe for train/state/compare; checkpoint setup for eval/pending)')


def checkpoint_arm(checkpoint):
    """Resolve this runner's setup or a legacy Full; reject incompatible original-main ablations."""
    saved = checkpoint['args']
    legacy = saved.get('ablation', 'full')
    if legacy != 'full':
        raise ValueError(f'original main-branch ablation {legacy!r} uses a different checkpoint format; '
                         'evaluate it with the original main code, or train this setup with test.py')
    arm = saved.get('arm')
    if arm is None:
        arm = 'full'
    if not isinstance(arm, str) or arm not in ARM_CLASSES:
        raise ValueError(f'unknown checkpoint arm {arm!r}')
    return arm


def selected_checkpoint_arm(checkpoint, requested = None):
    try:
        arm = checkpoint_arm(checkpoint)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if requested is not None and requested != arm:
        raise SystemExit(f'checkpoint setup is {arm!r}, not requested --arm {requested!r}')
    return arm


## Files and Provenance

def load_checkpoint(path):
    """A checkpoint written by main.py. Its tensors are memory-mapped, so reading the metadata alone is cheap."""
    return torch.load(path, map_location = 'cpu', weights_only = True, mmap = True)

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)

    return digest.hexdigest()

def read_json(path):
    try:
        with open(path, encoding = 'utf-8') as handle:
            return json.load(handle)

    except (OSError, ValueError):
        return None

def write_json(path, value):
    """Write to a temporary file first so readers only see complete JSON."""
    with open(f'{path}.tmp', 'w', encoding = 'utf-8') as handle:
        json.dump(value, handle, indent = 1)
    os.replace(f'{path}.tmp', path)

def code_state():
    """The commit, with +uncommitted when tracked files changed or Python or shell files are untracked, as rebuttal.sh did."""
    def git(*arguments):
        try:
            result = subprocess.run(['git', *arguments], cwd = os.path.dirname(os.path.abspath(__file__)), capture_output = True, text = True)
        except OSError:
            return None

        return result.stdout.strip() if result.returncode == 0 else None

    commit = git('rev-parse', 'HEAD')
    if not commit:
        return 'unknown'

    changed = git('status', '--porcelain', '--untracked-files=no') or git('ls-files', '--others', '--exclude-standard', '--', '*.py', '*.sh')
    return f'{commit}+uncommitted' if changed else commit

def provenance():
    """The code and the library versions behind a run, which the full model's runs did not record."""
    def version(name):
        try:
            return metadata.version(name)

        except metadata.PackageNotFoundError:
            return None

    return {'code': code_state(), 'python': sys.version.split()[0], 'cuda': torch.version.cuda,
            **{name: version(name) for name in ('torch', 'transformers', 'peft', 'accelerate', 'flash-attn')}}

## Checkpoints

def settings(checkpoint):
    """What a run was trained with: its arguments except UNCOMPARED, and the COMPARED fields."""
    values = {key: value for key, value in checkpoint['args'].items() if key not in UNCOMPARED}
    values.update({key: checkpoint.get(key, UNSET) for key in COMPARED})

    return values

def provenance_values(checkpoint, include_snapshots = True):
    """Known identities; None records absence only for CUDA/optional FlashAttention, otherwise it is unknown."""
    recorded = checkpoint['args'].get('provenance') or {}
    # A required module can import from source without installed distribution metadata.
    values = {f'provenance.{key}': recorded[key] for key in RUNTIME_FIELDS
              if key in recorded and (recorded[key] is not None or key in ('cuda', 'flash-attn'))}
    if include_snapshots:
        snapshots = recorded.get('snapshots') or {}
        values.update({f'provenance.snapshots.{key}': snapshots[key] for key in SNAPSHOT_FIELDS
                       if isinstance(snapshots.get(key), str) and snapshots[key]})
    return values


def provenance_comparison(full, arm, include_snapshots = True):
    """Return known-reference conflicts and fields whose historical Full identity is unverified."""
    reference = provenance_values(full, include_snapshots)
    current = provenance_values(arm, include_snapshots)
    keys = [f'provenance.{key}' for key in RUNTIME_FIELDS]
    if include_snapshots:
        keys.extend(f'provenance.snapshots.{key}' for key in SNAPSHOT_FIELDS)
    differences, unverified = {}, []
    for key in keys:
        if key not in reference:
            unverified.append(key)
        elif key not in current or reference[key] != current[key]:
            differences[key] = (reference[key], current.get(key, UNSET))
    return differences, unverified


def report_unverified(full, arm, context, include_snapshots = True):
    _, fields = provenance_comparison(full, arm, include_snapshots)
    if fields:
        print(f'Unverified provenance for {context}: Full did not record {", ".join(fields)}; '
              'matching settings do not establish the same historical runtime/backbones.', file = sys.stderr)
    return fields


def setting_differences(full, arm, include_snapshots = True):
    """{setting: (full model's value, arm's value)} for every setting in which two runs differ."""
    identity, _ = provenance_comparison(full, arm, include_snapshots)
    full, arm = settings(full), settings(arm)

    return {**{key: (full.get(key, UNSET), arm.get(key, UNSET)) for key in sorted(full.keys() | arm.keys())
               if full.get(key, UNSET) != arm.get(key, UNSET)}, **identity}

def checkpoint_problem(checkpoint, seed = None):
    """Why a checkpoint is not a completed run of SEED, or None.

    main.py saves at epoch boundaries and refuses epochs with no optimizer steps.
    Booleans are not valid integer counters, even though bool subclasses int.
    """
    saved = checkpoint.get('args')
    if not isinstance(saved, dict):
        return 'missing or invalid checkpoint arguments'

    actual_seed = saved.get('seed')
    if type(actual_seed) is not int:
        return f'invalid saved seed {actual_seed!r}; expected an integer'
    if seed is not None and (type(seed) is not int or actual_seed != seed):
        return f'saved seed {actual_seed!r} does not match requested seed {seed!r}'

    planned, completed = saved.get('epochs'), checkpoint.get('epoch')
    if type(planned) is not int or planned <= 0:
        return f'invalid planned epochs {planned!r}; expected a positive integer'
    if type(completed) is not int or completed < 0 or completed > planned:
        return f'invalid completed epoch {completed!r} for {planned} planned epochs'
    if completed < planned:
        return f'unfinished training: completed {completed} of {planned} epochs'

    steps = checkpoint.get('step')
    if type(steps) is not int or steps <= 0:
        return f'invalid completed optimizer steps {steps!r}; expected a positive integer'
    if 'optimizer_steps_this_epoch' in checkpoint:
        last_steps = checkpoint['optimizer_steps_this_epoch']
        if type(last_steps) is not int or not 0 < last_steps <= steps:
            return f'invalid last-epoch optimizer steps {last_steps!r} for {steps} total steps'

    return None


def completed_budget_differences(full, arm):
    """Compare realized training budgets after checkpoint_problem accepted both runs.

    Last-epoch counters are compared when both checkpoints record them; older
    checkpoints without that optional field still have required total steps.
    No step count is guessed from examples, since whole batches may be skipped.
    """
    fields = ['epoch', 'step']
    if 'optimizer_steps_this_epoch' in full and 'optimizer_steps_this_epoch' in arm:
        fields.append('optimizer_steps_this_epoch')
    return {field: (full[field], arm[field]) for field in fields if full[field] != arm[field]}


def full_model_problem(checkpoint):
    """Why CHECKPOINT is not a run of the full model, or None."""
    try:
        arm = checkpoint_arm(checkpoint)
    except ValueError as error:
        return str(error)
    return None if arm == 'full' else f'it is a {arm!r} run, not the full model'

def load_reference(path, seed):
    """The full-model checkpoint given to --match."""
    if not path:
        raise SystemExit('--match needs the path of a full-model checkpoint')

    reference = load_checkpoint(path)
    problem = full_model_problem(reference) or checkpoint_problem(reference, seed)
    if problem:
        raise SystemExit(f'--match {path}: {problem}; give the full model\'s checkpoint of seed {seed}')

    return reference

def checkpoint_state(path, reference = None, arm = DEFAULT_ARM, **expected):
    """(state, details) of PATH: 'absent'; 'other' unless it is a run of this arm with the EXPECTED arguments;
    'unfinished'; 'mismatched' when its settings, recorded identity or completed budget differ from REFERENCE;
    or 'finished' (with a warning when the reference's historical provenance cannot be verified)."""
    if not os.path.exists(path):
        return 'absent', {}

    checkpoint = load_checkpoint(path)
    saved = checkpoint['args']
    try:
        actual_arm = checkpoint_arm(checkpoint)
    except ValueError as error:
        return 'other', {'arm': str(error)}
    differences = {key: saved.get(key, UNSET) for key, value in expected.items() if saved.get(key, UNSET) != value}
    if actual_arm != arm:
        differences['arm'] = actual_arm
    if differences:
        return 'other', differences

    problem = checkpoint_problem(checkpoint, expected.get('seed'))
    if problem:
        return ('unfinished' if problem.startswith('unfinished training:') else 'other'), {'training': problem}

    differences = {}
    if reference is not None:
        problem = full_model_problem(reference) or checkpoint_problem(reference, saved['seed'])
        if problem:
            return 'mismatched', {'reference': problem}
        differences = {**setting_differences(reference, checkpoint), **completed_budget_differences(reference, checkpoint)}
        if not differences:
            report_unverified(reference, checkpoint, path)
    return ('mismatched', differences) if differences else ('finished', {})

## Commands

def train(argv):
    """main.py's training with the selected setup; takes main.py's options, --arm, --match and --overwrite."""
    parser = argparse.ArgumentParser(prog = 'test.py train', add_help = False, allow_abbrev = False,
                                     description = 'Train one of the nine IP-MoE setups. '
                                                   'Every other option is main.py\'s, listed below.')
    arm_argument(parser)
    parser.add_argument('--match', metavar = 'FULL_CHECKPOINT',
                        help = 'match a completed Full run: seed, settings, GPU/data, recorded runtime/backbones and actual steps')
    parser.add_argument('--overwrite', action = 'store_true', help = 'replace a finished run of this arm in --output-dir')
    if {'-h', '--help'} & set(argv):
        parser.print_help()
        print('\nmain.py options:')

    own, rest = parser.parse_known_args(argv)
    arm = own.arm
    sys.argv = [sys.argv[0], *rest]
    args = main.parse_args()
    if args.output_dir == 'checkpoint':
        raise SystemExit('Pass --output-dir: main.py\'s default holds full-model checkpoints '
                         '(test.sh uses checkpoint/<arm>/<llm>/seed_<seed>)')

    # run_training builds the model through the global name main.IPMoE, which is replaced below.
    if 'IPMoE' not in main.run_training.__code__.co_names:
        raise SystemExit('main.run_training no longer builds main.IPMoE, so test.py cannot select a setup')

    args.arm, args.provenance = arm, provenance()
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    planned = {'args': vars(args), 'world_size': world_size, 'global_batch_size': args.batch_size * world_size,
               'protocol': PREPROCESSING_PROTOCOL, 'source': SOURCES}

    path = os.path.join(args.output_dir, 'checkpoint.pt')
    if own.match is not None and os.path.realpath(path) == os.path.realpath(own.match):
        raise SystemExit('--output-dir must not overwrite the --match reference checkpoint')
    state, differences = checkpoint_state(path, arm = arm, llm_type = args.llm_type, seed = args.seed)
    if state == 'other':
        raise SystemExit(f'{path} belongs to another run (saved {differences}); choose another --output-dir')

    if state == 'finished' and not own.overwrite:
        raise SystemExit(f'{path} holds a finished {arm} run; pass --overwrite to replace it')

    reference = load_reference(own.match, args.seed) if own.match is not None else None
    if reference is not None:
        differences = setting_differences(reference, planned, include_snapshots = False)
        if differences:
            raise SystemExit(f'This run would differ from {own.match} (full model, this run):\n'
                             + '\n'.join(f'  {key}: {full!r}, {arm!r}' for key, (full, arm) in differences.items())
                             + '\nTrain with the full model\'s settings on as many GPUs, or drop --match (MATCH=0 in test.sh).')

    rank = int(os.environ.get('RANK', '0'))
    if rank == 0:
        print(f'Arm {arm} ({ARM_CLASSES[arm].__name__}): {ARM_CLASSES[arm].__doc__} Output {args.output_dir}; '
              + (f'declared settings checked against {own.match}; backbone revisions and realized steps still pending'
                 if own.match else 'settings not checked against a full-model run'))
        print(f'Provenance: {json.dumps(args.provenance)}')

    def build(*arguments, **keywords):
        model = ARM_CLASSES[arm](*arguments, **keywords)

        # The backbone snapshots this run loaded, saved with the checkpoint's arguments.
        args.provenance['snapshots'] = {'clip': getattr(model.text_encoder.clip.config, '_commit_hash', None),
                                        'llm': getattr(model.llm.model.config, '_commit_hash', None)}
        if reference is not None:
            differences = setting_differences(reference, planned)
            if differences:
                raise SystemExit(f'Loaded model/runtime differs from {own.match}: {differences}')
            if rank == 0:
                report_unverified(reference, planned, f'training {arm} seed {args.seed}')
        if rank == 0:
            print(f'Backbone snapshots: {json.dumps(args.provenance["snapshots"])}')

        return model

    original = main.IPMoE
    main.IPMoE = build
    try:
        main.main(args)
    finally:
        main.IPMoE = original

    if reference is not None and rank == 0:
        if not os.path.isfile(path):
            raise SystemExit(f'Training did not save the checkpoint required for --match verification: {path}')
        completed = load_checkpoint(path)
        selected_checkpoint_arm(completed, arm)
        problem = checkpoint_problem(completed, args.seed)
        if problem:
            raise SystemExit(f'{path}: {problem}')
        differences = {**setting_differences(reference, completed), **completed_budget_differences(reference, completed)}
        if differences:
            raise SystemExit(f'Completed training differs from {own.match}: {differences}')

def state(argv):
    """Print absent, unfinished, finished, mismatched or other for a checkpoint of this arm."""
    parser = argparse.ArgumentParser(prog = 'test.py state', description = state.__doc__)
    arm_argument(parser)
    parser.add_argument('checkpoint')
    parser.add_argument('--llm-type', required = True)
    parser.add_argument('--seed', type = int, required = True)
    parser.add_argument('--match', metavar = 'FULL_CHECKPOINT', help = 'check settings, recorded runtime/backbones and completed steps against Full')
    args = parser.parse_args(argv)

    reference = load_reference(args.match, args.seed) if args.match is not None else None
    result, differences = checkpoint_state(args.checkpoint, reference, arm = args.arm, llm_type = args.llm_type, seed = args.seed)
    if differences:
        print(f'{args.checkpoint} is {result}: {differences}', file = sys.stderr)

    print(result)

def evaluate(argv):
    """eval.py's evaluation of the arm checkpoint IPMOE_CHECKPOINT; takes eval.py's (VLMEvalKit's) options."""
    # Already imported before Torch when this file runs as `test.py eval`.
    from VLMEvalKit import run
    import torch.distributed as distributed
    from vlmeval.api import OpenAIWrapper

    import vlmeval_adapter
    from eval import run_checked, use_completion_tokens

    checkpoint_path = os.environ.get('IPMOE_CHECKPOINT')
    if not checkpoint_path:
        raise SystemExit('Set IPMOE_CHECKPOINT to the arm\'s checkpoint; eval.py would load the full model\'s')
    checkpoint_path = os.path.abspath(checkpoint_path)

    parser = argparse.ArgumentParser(prog = 'test.py eval', add_help = False, allow_abbrev = False)
    arm_argument(parser, default = None)
    own, argv = parser.parse_known_args(argv)
    arm = selected_checkpoint_arm(load_checkpoint(checkpoint_path), own.arm)
    marker_name = f'{arm}.json'

    intervention = os.environ.get('IPMOE_INTERVENTION') or None
    # The original adapter's router-column interventions assume all three experts.
    allowed = vlmeval_adapter.INTERVENTIONS if arm == 'full' else INTERVENTIONS if ARM_CLASSES[arm].keep_interaction else ()
    if intervention is not None and intervention not in allowed:
        raise SystemExit(f'IPMOE_INTERVENTION={intervention!r} is unsupported for {arm}; allowed: {allowed}')

    seed = os.environ.get('PYTHONHASHSEED')
    if seed is None:
        raise SystemExit('Set PYTHONHASHSEED to the seed, as eval.sh does')
    main.reproduce(int(seed))

    context_limit = os.environ.get('IPMOE_CONTEXT_LIMIT')
    context_limit = int(context_limit) if context_limit is not None else None

    # The adapter builds main.build_model's modules and then the class its module names IPMoE. Selecting that class
    # keeps the full model's checkpoint checks, prompts, context limit and generation.
    def build_adapter(**options):
        original = vlmeval_adapter.IPMoE
        vlmeval_adapter.IPMoE = ARM_CLASSES[arm]
        try:
            return vlmeval_adapter.IPMoEVLMEval(**options)
        finally:
            vlmeval_adapter.IPMoE = original

    run.supported_VLM['IPMoE'] = functools.partial(build_adapter, checkpoint_path = checkpoint_path, context_limit = context_limit)

    use_completion_tokens(OpenAIWrapper)
    # OpenAIWrapper logs the judge's API key at INFO, which would reach the console and run logs.
    logging.getLogger('vlmeval.api.gpt').setLevel(logging.WARNING)

    sys.argv = [sys.argv[0], *argv]
    run.load_env()
    toolkit_args = run.parse_args()

    # Checked on the parsed options, so abbreviations such as --reuse-a or --work are caught too.
    if toolkit_args.model not in ('IPMoE', ['IPMoE']):
        raise SystemExit('test.py eval evaluates the arm under the model name IPMoE; pass --model IPMoE')

    if getattr(toolkit_args, 'reuse', False) or getattr(toolkit_args, 'mode', 'all') != 'all':
        raise SystemExit('test.py eval predicts and scores afresh (--mode all, no --reuse), so that every result comes from the checkpoint '
                         'it names; to redo a failed evaluation, run it again (test.sh does so for the benchmarks without a completed run)')

    foreign = foreign_runs(toolkit_args.work_dir, arm)
    if foreign:
        raise SystemExit(f'{toolkit_args.work_dir} holds evaluations that test.py eval did not make ({", ".join(foreign)}); '
                         'VLMEvalKit would relink their results to the arm\'s, so choose another --work-dir')

    # Mark every run directory VLMEvalKit creates with the checkpoint it evaluates; compare reads only completed runs.
    record = {'arm': arm, 'checkpoint': checkpoint_path, 'sha256': None, 'judge': toolkit_args.judge, 'data': toolkit_args.data,
              'intervention': intervention, 'context_limit': context_limit, 'provenance': provenance(), 'completed': False}
    if run.RANK == 0:
        record['sha256'] = file_sha256(checkpoint_path)
        print(f'Arm {arm}: evaluating {checkpoint_path} (sha256 {record["sha256"]}); provenance {json.dumps(record["provenance"])}')

    runs = []
    upsert = run.upsert_run_status

    def mark(run_dir, *positional, **fields):
        # Mark the directory before its first status.json where it exists already: a run directory with status.json
        # but no marker is taken for another model's.
        directory = os.path.realpath(run_dir)
        first = run.RANK == 0 and directory not in runs
        if first and os.path.isdir(directory):
            runs.append(directory)
            write_json(os.path.join(directory, marker_name), record)
            first = False

        result = upsert(run_dir, *positional, **fields)
        if first:
            runs.append(directory)
            write_json(os.path.join(directory, marker_name), record)

        return result

    run.upsert_run_status = mark
    try:
        run_checked(run, toolkit_args, distributed)
    finally:
        run.upsert_run_status = upsert

    # run_checked raises on every failure it finds, so all benchmarks of these runs passed eval.py's checks.
    for directory in runs:
        write_json(os.path.join(directory, marker_name), {**record, 'completed': True})

def result_files(dataset, judge):
    """The scored file VLMEvalKit writes for one benchmark of IPMoE, and the summary beside it with the headline score."""
    base = f'IPMoE_{dataset}'
    if dataset == 'NaturalBenchDataset':
        return f'{base}.xlsx', f'{base}_acc.csv'

    if dataset == 'POPE':
        return f'{base}_auxmatch.xlsx', f'{base}_score.csv'

    return f'{base}_{judge}_result.xlsx', f'{base}_acc.csv'

def expected_judge(dataset, judge):
    """eval.sh and test.sh score NaturalBench by exact matching, whatever the judge."""
    return 'exact_matching' if dataset == 'NaturalBenchDataset' else judge

def read_run_marker(directory):
    """Return (record, problem), recognizing every setup and the original no_moe.json format."""
    found = [(arm, os.path.join(directory, f'{arm}.json')) for arm in ARM_CLASSES
             if os.path.exists(os.path.join(directory, f'{arm}.json'))]
    if not found:
        return None, None
    if len(found) != 1:
        return None, 'multiple setup markers: ' + ', '.join(arm for arm, _ in found)
    arm, path = found[0]
    record = read_json(path)
    if not isinstance(record, dict) or record.get('arm') != arm:
        return None, f'invalid {arm}.json setup marker'
    return record, None


def run_problem(directory, dataset, judge, digest, arm = DEFAULT_ARM, require_marker = True):
    """Why a scored run is unusable. Only legacy Full references may omit a setup marker;
    any marker that exists must match ARM and DIGEST and describe a completed, unmodified evaluation."""
    status = read_json(os.path.join(directory, 'status.json'))
    records = status.get('datasets') if isinstance(status, dict) else None
    record = records.get(dataset) if isinstance(records, dict) else None
    if not isinstance(record, dict):
        return 'no status.json record'

    # eval.py's checks of a finished benchmark.
    if 'error_message' in record:
        return f'failed: {record["error_message"] or "unspecified evaluation error"}'

    if record.get('status') != 'done' or record.get('skip_reason') or not isinstance(record.get('metrics'), dict) or not record['metrics']:
        return f'status {record.get("status")!r}, skip_reason {record.get("skip_reason")!r}, metrics {record.get("metrics")!r}'

    if record.get('judge_model') != expected_judge(dataset, judge):
        return f'judged by {record.get("judge_model")!r}, not {expected_judge(dataset, judge)!r}'

    marker, problem = read_run_marker(directory)
    if problem:
        return problem
    if marker is None and not require_marker and arm == 'full':
        return None

    if not isinstance(marker, dict) or not marker.get('completed'):
        return 'not completed by test.py eval: it failed one of eval.py\'s checks or was interrupted'

    if marker.get('arm') != arm:
        return f'an evaluation of {marker.get("arm")!r}, not {arm!r}'

    if marker.get('sha256') != digest:
        return f'an evaluation of another checkpoint ({marker.get("checkpoint")}, sha256 {str(marker.get("sha256"))[:12]})'

    if marker.get('intervention'):
        return f'made under IPMOE_INTERVENTION={marker["intervention"]}'

    if record.get('source_run'):
        return f'predictions reused from {record["source_run"]}'

    return None

def headline(path, dataset):
    """The headline score in percent of DATASET's summary file PATH."""
    import pandas as pd

    table = pd.read_csv(path)
    if dataset == 'POPE':
        score = float(table.loc[table['split'] == 'Overall', 'Overall'].iloc[0])

    elif dataset == 'NaturalBenchDataset':
        score = 100 * float(table.set_index('Metric')['Score'][NATURALBENCH_METRIC])

    else:
        score = 100 * float(table['Overall'].iloc[0])

    if not math.isfinite(score):
        raise ValueError(f'{path} has no finite score')

    return score

def saved_score(work, dataset, judge, digest, arm = DEFAULT_ARM, require_marker = True):
    """(run directory, headline score in percent) of DATASET in the VLMEvalKit work directory WORK.

    The top-level files link to the latest run of each benchmark; another run is read only when it is the single run
    that passes run_problem. Summaries are not named after the judge, so the one beside the scored file is read.
    For legacy Full references (REQUIRE_MARKER False), a failed evaluation can still look complete in status.json, so runs whose
    scores differ are reported instead of trusting the link."""
    scored, summary = result_files(dataset, judge)
    directory = os.path.join(work, 'IPMoE')
    link = os.path.join(directory, scored)
    linked = os.path.dirname(os.path.realpath(link)) if os.path.isfile(link) else None

    runs = {os.path.dirname(os.path.realpath(path)) for path in glob.glob(os.path.join(directory, 'T*', scored))} | ({linked} if linked else set())
    problems, usable = {}, []
    for candidate in sorted(runs):
        problem = run_problem(candidate, dataset, judge, digest, arm, require_marker) if os.path.isfile(os.path.join(candidate, summary)) else f'no {summary}'
        if problem:
            problems[os.path.basename(candidate)] = problem
        else:
            usable.append(candidate)

    chosen = linked if linked in usable else usable[0] if len(usable) == 1 else None
    if chosen is None:
        if usable:
            raise LookupError(f'{len(usable)} usable runs in {directory} hold {scored} and the link names none of them; link the one to use')

        raise LookupError(f'no usable {scored} in {directory}' + (': ' + '; '.join(f'{name} {problem}' for name, problem in problems.items()) if problems else ''))

    if not require_marker and len(usable) > 1:
        scores = {os.path.basename(candidate): headline(os.path.join(candidate, summary), dataset) for candidate in usable}
        if max(scores.values()) - min(scores.values()) > 1e-9:
            raise LookupError(f'usable runs in {directory} disagree on {dataset} (' + '; '.join(f'{name} {score:.3f}' for name, score in scores.items())
                              + '); one may have failed eval.py\'s checks, so remove or relink the wrong one')

    return chosen, headline(os.path.join(chosen, summary), dataset)

def foreign_runs(work, arm = DEFAULT_ARM):
    """Reject unmarked, malformed, or other-setup runs before VLMEvalKit relinks their results."""
    foreign = []
    for path in glob.glob(os.path.join(work, 'IPMoE', 'T*', 'status.json')):
        record, problem = read_run_marker(os.path.dirname(path))
        if problem or record is None or record.get('arm') != arm:
            foreign.append(os.path.basename(os.path.dirname(path)))
    return sorted(foreign)

def pending(argv):
    """Print the benchmarks without a usable result of the arm checkpoint in its evaluation directory."""
    parser = argparse.ArgumentParser(prog = 'test.py pending', description = pending.__doc__)
    arm_argument(parser, default = None)
    parser.add_argument('checkpoint')
    parser.add_argument('work', help = 'the evaluation work directory of the checkpoint\'s seed')
    parser.add_argument('--judge', required = True)
    parser.add_argument('--data', nargs = '+', choices = DATASETS, default = list(DATASETS))
    args = parser.parse_args(argv)
    arm = selected_checkpoint_arm(load_checkpoint(args.checkpoint), args.arm)

    # VLMEvalKit relinks the top-level files of every IP-MoE run in a work directory to its newest run, so evaluating
    # into a directory with other models' runs, e.g. the full model's, would replace their results.
    foreign = foreign_runs(args.work, arm)
    if foreign:
        raise SystemExit(f'{args.work} holds evaluations that test.py eval did not make ({", ".join(foreign)}); choose another EVAL_DIR')

    digest = file_sha256(args.checkpoint)
    missing = []
    for dataset in dict.fromkeys(args.data):
        try:
            saved_score(args.work, dataset, args.judge, digest, arm)

        except (LookupError, OSError, ValueError, KeyError, IndexError):
            missing.append(dataset)

    # A prefix, so that test.sh can tell this line, possibly empty, from anything a library prints.
    print('pending:', *missing)

def training_row(model, seed, path, checkpoint):
    saved, counts = checkpoint['args'], checkpoint.get('data_counts', {})
    factor = ARM_CLASSES[checkpoint_arm(checkpoint)].prefix_factor
    prefix = '576' if factor == 0 else '576 + N' if factor == 1 else f'576 + {factor}N'

    return {'model': model, 'seed': seed, 'prefix_tokens': prefix,
            'trainable_parameters': sum(tensor.numel() for tensor in checkpoint['model'].values()),
            'epoch': checkpoint.get('epoch'), 'epochs': saved.get('epochs'), 'optimizer_steps': checkpoint.get('step'),
            'world_size': checkpoint.get('world_size'), 'global_batch_size': checkpoint.get('global_batch_size'),
            'val_loss': checkpoint.get('val_loss'), **{key: counts.get(key) for key in COUNTS},
            'code': saved.get('provenance', {}).get('code'), 'checkpoint': path, 'sha256': file_sha256(path)}

def compare(argv):
    """Put the arm's benchmark scores, validation loss and training counters beside the full model's, paired by seed."""
    import pandas as pd

    parser = argparse.ArgumentParser(prog = 'test.py compare', description = compare.__doc__)
    arm_argument(parser)
    parser.add_argument('--seeds', nargs = '+', type = int, required = True)
    parser.add_argument('--data', nargs = '+', choices = DATASETS, default = list(DATASETS), help = 'benchmarks to compare; Avg needs all seven')
    parser.add_argument('--judge', default = 'gpt-6-luna', help = 'the JUDGE both models were evaluated with')
    parser.add_argument('--full-checkpoints', required = True, help = 'directory of the full model\'s seed_N/checkpoint.pt')
    parser.add_argument('--full-results', required = True, help = 'directory of the full model\'s seed_N evaluation work directories')
    parser.add_argument('--arm-checkpoints', required = True, help = 'the same for this arm')
    parser.add_argument('--arm-results', required = True, help = 'the same for this arm')
    parser.add_argument('--output', required = True, help = 'directory for scores.csv, summary.csv, training.csv, sources.csv and verification.json')
    args = parser.parse_args(argv)
    args.seeds, args.data = list(dict.fromkeys(args.seeds)), list(dict.fromkeys(args.data))
    arm_name = args.arm
    reference_name = 'full_reference' if arm_name == 'full' else 'full'

    problems, rows, training, sources, checkpoints, verification = [], [], [], [], {}, []
    for model, checkpoint_root, result_root in ((reference_name, args.full_checkpoints, args.full_results),
                                                (arm_name, args.arm_checkpoints, args.arm_results)):
        is_reference = model == reference_name
        expected_arm = 'full' if is_reference else arm_name
        for seed in args.seeds:
            row = {'model': model, 'seed': seed}
            path = os.path.join(checkpoint_root, f'seed_{seed}', 'checkpoint.pt')
            require_marker = not is_reference
            if os.path.isfile(path):
                checkpoint = checkpoints[model, seed] = load_checkpoint(path)
                selected_checkpoint_arm(checkpoint, expected_arm)
                problem = checkpoint_problem(checkpoint, seed)
                if problem:
                    raise SystemExit(f'{path}: {problem}')
                # Only Full runs predating test.py have no setup marker. A new Full
                # reference must identify its evaluated checkpoint just like any candidate.
                require_marker = require_marker or checkpoint['args'].get('arm') is not None

                training.append(training_row(model, seed, path, checkpoint))
                row['val_loss'] = checkpoint.get('val_loss')
                digest = training[-1]['sha256']

            else:
                raise SystemExit(f'{model} seed {seed}: no checkpoint {path}; cannot verify the seed or completed training')

            # The arm's scores must come from completed runs of its current checkpoint.
            work = os.path.join(result_root, f'seed_{seed}')
            for dataset in args.data:
                try:
                    source, row[dataset] = saved_score(work, dataset, args.judge, digest,
                                                      arm = expected_arm, require_marker = require_marker)

                except (LookupError, OSError, ValueError, KeyError, IndexError) as error:
                    problems.append(f'{model} seed {seed} {dataset}: {error}')
                    continue

                status = read_json(os.path.join(source, 'status.json')) or {}
                sources.append({'model': model, 'seed': seed, 'dataset': dataset, 'score': row[dataset], 'run': source,
                                'commit': status.get('commit'), 'created_at': status.get('created_at'),
                                'judge': status.get('datasets', {}).get(dataset, {}).get('judge_model')})

            rows.append(row)

    for seed in args.seeds:
        full, arm = checkpoints[reference_name, seed], checkpoints[arm_name, seed]
        differences = {**setting_differences(full, arm), **completed_budget_differences(full, arm)}
        if differences:
            raise SystemExit(f'seed {seed}: training differs (full model, arm): '
                             + '; '.join(f'{key} {a!r}, {b!r}' for key, (a, b) in differences.items()))

        # The arm keeps exactly the examples the full model keeps.
        for key in ('train_budget_skipped', 'val_budget_skipped'):
            a, b = full.get('data_counts', {}).get(key), arm.get('data_counts', {}).get(key)
            if a != b:
                raise SystemExit(f'seed {seed}: {key} is {a} for the full model but {b} for the arm')

        unverified = report_unverified(full, arm, f'comparison seed {seed}')
        verification.append({'seed': seed, 'provenance': 'unverified' if unverified else 'matched',
                             'unverified_fields': unverified, 'completed_epochs': full['epoch'],
                             'optimizer_steps': full['step']})

    metrics = [*args.data, *(['Avg'] if set(args.data) == set(DATASETS) else []), 'val_loss']
    table = pd.DataFrame(rows).set_index(['model', 'seed']).reindex(columns = metrics).astype(float)
    if 'Avg' in metrics:
        # Average the seven within each seed first, as the thesis does.
        table['Avg'] = table[list(DATASETS)].mean(axis = 1, skipna = False)

    full, arm = table.xs(reference_name, level = 'model'), table.xs(arm_name, level = 'model')
    difference = arm - full
    by_seed = pd.DataFrame([{'seed': seed, 'model': model, **frame.loc[seed].to_dict()}
                            for seed in args.seeds for model, frame in ((reference_name, full), (arm_name, arm), ('difference', difference))]).set_index(['seed', 'model'])

    # Each metric over the seeds where both models have it, so the difference of the means is the mean difference.
    # The paired t interval covers the variation between training seeds only, not the sampling of benchmark questions.
    summary = {}
    for metric in metrics:
        paired = pd.DataFrame({reference_name: full[metric], arm_name: arm[metric]}).dropna()
        change = paired[arm_name] - paired[reference_name]
        half = T975.get(len(paired) - 1, 1.96) * change.std() / math.sqrt(len(paired)) if len(paired) > 1 else math.nan
        summary[metric] = {'seeds': len(paired), f'{reference_name}_mean': paired[reference_name].mean(),
                           f'{reference_name}_sd': paired[reference_name].std(),
                           f'{arm_name}_mean': paired[arm_name].mean(), f'{arm_name}_sd': paired[arm_name].std(),
                           'difference_mean': change.mean(), 'difference_sd': change.std(),
                           'difference_ci95_low': change.mean() - half, 'difference_ci95_high': change.mean() + half}
    summary = pd.DataFrame.from_dict(summary, orient = 'index')
    training = pd.DataFrame(training)
    sources = pd.DataFrame(sources, columns = ['model', 'seed', 'dataset', 'score', 'run', 'commit', 'created_at', 'judge'])

    os.makedirs(args.output, exist_ok = True)
    write_json(os.path.join(args.output, 'verification.json'), {'seeds': verification})
    by_seed.to_csv(os.path.join(args.output, 'scores.csv'))
    summary.to_csv(os.path.join(args.output, 'summary.csv'), index_label = 'metric')
    training.to_csv(os.path.join(args.output, 'training.csv'), index = False)
    sources.to_csv(os.path.join(args.output, 'sources.csv'), index = False)

    with pd.option_context('display.width', 250, 'display.max_columns', None, 'display.max_colwidth', 80):
        print(f'Scores (percent) and validation loss by seed; difference = {arm_name} - {reference_name}; judge={args.judge}')
        print(by_seed.to_string(float_format = '{:.2f}'.format, formatters = {'val_loss': '{:.4f}'.format}))
        print('\nMean and sample SD over the seeds both models have, and a 95% paired t interval of the difference over those '
              'seeds; it reflects training randomness only, not the sampling of benchmark questions, and few seeds make it wide')
        print(summary.to_string(float_format = '{:.3f}'.format))
        if not training.empty:
            print('\nTraining (data_counts from main.py; retained counts include distributed-sampler padding)')
            print(training.drop(columns = ['code', 'checkpoint', 'sha256']).to_string(index = False, float_format = '{:.4f}'.format))

    # The judge is an API model; runs judged weeks apart may not have been judged alike.
    judged = sources[sources['judge'].ne('exact_matching') & sources['created_at'].notna()]
    for model, dates in judged.groupby('model')['created_at']:
        print(f'{model}: judge-scored runs created {min(dates)[:10]} to {max(dates)[:10]}')

    print(f'\nWrote scores.csv, summary.csv, training.csv, sources.csv and verification.json to {args.output}')
    if problems:
        print('\nIncomplete or not comparable:', *problems, sep = '\n- ', file = sys.stderr)
        raise SystemExit(1)

COMMANDS = {'train': train, 'eval': evaluate, 'compare': compare, 'state': state, 'pending': pending}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f'usage: test.py {{{",".join(COMMANDS)}}} [options]; the docstring of test.py describes each')

    COMMANDS[sys.argv[1]](sys.argv[2:])
