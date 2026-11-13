"""Ablation: IP-MoE without the router and the expert pool.

The arm keeps the frozen CLIP encoders and their trained MLPs, the interaction map with its adapter, and the image and
interaction heads. It drops the MoE module (the dynamic router and the self, cross and hybrid experts), the auxiliary
tokens the experts attend to, and expert_head, so the language model's prefix is the 576 image tokens and the N
interaction tokens (576 + N) instead of 576 + 4N. Everything else follows the full model, so the two compare by seed:

train    main.py's training loop and checkpoint format, with NoMoE in place of IPMoE. Every module is built in the full
         model's order before the MoE part is dropped, so the kept modules start from the full model's initialization
         for the same seed. Each example gets the text budget the full model gives it under --context-limit, so the arm
         trains and validates on exactly the examples the full model keeps (the thesis's full runs skipped none at 4096
         tokens, so this equals budgeting with the arm's shorter prefix). --match FULL_CHECKPOINT refuses to start
         unless the settings, GPU count and data equal that full-model run's. The code revision, library versions and
         backbone snapshots are saved in the checkpoint's arguments under provenance.
eval     eval.py's VLMEvalKit evaluation of IPMOE_CHECKPOINT through the same adapter, so prompts, context limit,
         generation and checks are the full model's. It always predicts and scores afresh (no --reuse, --mode all).
         Each run directory it creates gets a no_moe.json naming the checkpoint and its SHA-256, marked completed only
         once eval.py's checks have passed. Give runs under IPMOE_INTERVENTION (swap or mean) their own --work-dir, as
         eval.sh does: pending and compare never read them.
compare  the arm's scores, validation loss and training counters beside the full model's, paired by seed, with a 95%
         paired t interval over seeds. The arm's scores come only from completed runs of its current checkpoint. The
         full model's runs can be checked only against status.json (the run's status and judge), since eval.py keeps no
         verdict of its own, so several such runs that disagree are reported instead of one being picked.
state    whether a checkpoint is a finished run of this arm, and with --match whether it has the full model's settings.
pending  the benchmarks an evaluation directory has no usable result of the arm's current checkpoint for.

test.sh runs these for every seed; run them from the repository root.
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

ARM = 'no_moe'
# Written by test.py eval into every VLMEvalKit run directory it creates.
MARKER = f'{ARM}.json'

# The seven benchmarks of the thesis's Avg, in eval.sh's order, and the NaturalBench score it uses.
DATASETS = ('NaturalBenchDataset', 'CV-Bench-2D', 'CV-Bench-3D', 'MMBench_DEV_EN_V11', 'MMStar', 'POPE', 'SEEDBench_IMG')
NATURALBENCH_METRIC = 'Acc'
# The test-time interventions of vlmeval_adapter.py that need no router or experts.
INTERVENTIONS = ('swap', 'mean')
# Two-sided 95% quantiles of Student's t by degrees of freedom, for intervals over a few seeds without scipy.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

# Saved arguments that may differ from the full model's run without changing what is learned: where and under which
# name a run is saved, how its batches are loaded and recomputed, its provenance, and the removed ablation code's metadata.
UNCOMPARED = {'output_dir', 'num_workers', 'gradient_checkpointing', 'arm', 'provenance', 'ablation', 'ablation_version', 'ablation_config'}
# Checkpoint fields besides the arguments that must match too: the process layout and the training data.
COMPARED = ('world_size', 'global_batch_size', 'protocol', 'source')
# main.py's data_counts shown by compare.
COUNTS = ('train_split', 'train_retained_this_epoch', 'train_budget_skipped', 'val_split', 'val_retained', 'val_budget_skipped')
UNSET = '<unset>'

class NoMoE(IPMoE):
    """IPMoE without the router and the expert pool: the prefix is the image tokens and the interaction tokens."""

    def __init__(self, *args, **kwargs):
        # Build every module in the full model's order, so the kept modules start from the full model's initialization
        # for the same seed. Then drop the MoE (router and experts), the auxiliary tokens only the experts attend to, and
        # the expert projection, leaving no trainable parameter unused, as DDP requires.
        super().__init__(*args, **kwargs)

        # The caller still holds the MoE it built; on the meta device its weights take no memory.
        self.moe.to('meta')
        del self.moe, self.aux, self.expert_head

    def encode_prefix(self, text_inputs, image_inputs, map_image_inputs = None):
        """IPMoE.encode_prefix without the expert tokens. With no router, the weights have no columns."""
        image_features = self.vision_encoder(**image_inputs)
        image_mask = torch.ones(image_features.shape[:2], dtype = torch.bool, device = image_features.device)

        # Process the question.
        text_features, text_mask = self.text_encoder(**text_inputs)

        # Interaction between text and vision features.
        map_features = image_features if map_image_inputs is None else self.vision_encoder(**map_image_inputs)
        similarity = self.interaction(text_features, map_features)
        interaction_tokens, _, _, token_mask = self.interaction_adapter(similarity, text_mask)

        # Pass the image and interaction tokens to the LLM.
        prefix = torch.cat([self.image_head(image_features), self.interaction_head(interaction_tokens)], dim = 1)
        prefix_mask = torch.cat([image_mask, token_mask], dim = 1)

        # One row per sample, which is how the training loop counts the examples it kept.
        return prefix, prefix_mask, image_features.new_empty((image_features.size(0), 0))

    def _prefix_lengths(self, text_inputs, image_inputs):
        # The full model's prefix is the image tokens and four groups of N tokens: the interaction tokens and one group
        # per expert. This arm keeps the image tokens and the interaction tokens.
        vision = self.vision_encoder.clip.config.vision_config
        image_tokens = (int(vision.image_size) // int(vision.patch_size)) ** 2

        return [image_tokens + (length - image_tokens) // 4 for length in super()._prefix_lengths(text_inputs, image_inputs)]

    def forward(self, questions, text_inputs, image_inputs, answers, context_limit, return_stats = False, allow_empty = False):
        """IPMoE.forward for this prefix. Each example gets the text budget the full model gives it, so both keep the
        same examples under context_limit."""
        prefix_lengths = self._prefix_lengths(text_inputs, image_inputs)
        text_budget = [context_limit - length for length in super()._prefix_lengths(text_inputs, image_inputs)]

        # The LLM receives the original question and supervises only its answer.
        try:
            token_rows, label_rows, kept_rows, positions = self.llm.tokenize_training_chats(questions, answers, max_total_tokens = text_budget)

        except NoValidTargetsError:
            if not allow_empty:
                raise

            # Empty DDP ranks must still participate in every gradient reduction.
            loss = sum(parameter.reshape(-1)[0].float() * 0 for parameter in self.parameters() if parameter.requires_grad)
            weights = self.image_head.weight.new_empty((0, 0))
            return (loss, weights, 0) if return_stats else (loss, weights)

        # The tokenizer raises NoValidTargetsError before encoding an empty selection.
        def select(inputs):
            return {key: value.index_select(0, torch.tensor(kept_rows, device = value.device, dtype = torch.long)) for key, value in inputs.items()}

        prefix, prefix_mask, weights = self.encode_prefix(select(text_inputs), select(image_inputs))
        expected = [prefix_lengths[index] for index in kept_rows]
        if prefix_mask.sum(dim = 1).tolist() != expected:
            raise RuntimeError('Predicted prefix lengths disagree with encoded prefix lengths')

        embeds, mask, position_ids, labels = self.llm.build_inputs(token_rows, positions, prefix, label_rows, prefix_mask = prefix_mask)
        loss = self.llm(embeds, mask, position_ids, labels)

        if return_stats:
            return loss, weights, (labels[:, 1:] != -100).sum().item()

        return loss, weights

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

def setting_differences(full, arm):
    """{setting: (full model's value, arm's value)} for every setting in which two runs differ."""
    full, arm = settings(full), settings(arm)

    return {key: (full.get(key, UNSET), arm.get(key, UNSET)) for key in sorted(full.keys() | arm.keys())
            if full.get(key, UNSET) != arm.get(key, UNSET)}

def full_model_problem(checkpoint):
    """Why CHECKPOINT is not a run of the full model, or None."""
    saved = checkpoint['args']
    if saved.get('arm') is not None or saved.get('ablation', 'full') != 'full':
        return f'it is an ablation run (arm={saved.get("arm")!r}, ablation={saved.get("ablation", "full")!r})'

    return None

def load_reference(path, seed):
    """The full-model checkpoint given to --match."""
    if not path:
        raise SystemExit('--match needs the path of a full-model checkpoint')

    reference = load_checkpoint(path)
    problem = full_model_problem(reference)
    if problem:
        raise SystemExit(f'--match {path}: {problem}; give the full model\'s checkpoint of seed {seed}')

    return reference

def checkpoint_state(path, reference = None, **expected):
    """(state, details) of PATH: 'absent'; 'other' unless it is a run of this arm with the EXPECTED arguments;
    'unfinished'; 'mismatched' when it finished with settings that differ from the REFERENCE run's; or 'finished'."""
    if not os.path.exists(path):
        return 'absent', {}

    checkpoint = load_checkpoint(path)
    saved = checkpoint['args']
    differences = {key: saved.get(key, UNSET) for key, value in {'arm': ARM, **expected}.items() if saved.get(key, UNSET) != value}
    if differences:
        return 'other', differences

    if checkpoint.get('epoch') != saved.get('epochs'):
        return 'unfinished', {}

    differences = setting_differences(reference, checkpoint) if reference is not None else {}
    return ('mismatched', differences) if differences else ('finished', {})

## Commands

def train(argv):
    """main.py's training with NoMoE in place of IPMoE; takes main.py's options and --match and --overwrite."""
    parser = argparse.ArgumentParser(prog = 'test.py train', add_help = False, allow_abbrev = False,
                                     description = 'Train IP-MoE without the router and the expert pool. '
                                                   'Every other option is main.py\'s, listed below.')
    parser.add_argument('--match', metavar = 'FULL_CHECKPOINT',
                        help = 'refuse to train unless the settings, GPU count and data equal this full-model run\'s')
    parser.add_argument('--overwrite', action = 'store_true', help = 'replace a finished run of this arm in --output-dir')
    if {'-h', '--help'} & set(argv):
        parser.print_help()
        print('\nmain.py options:')

    own, rest = parser.parse_known_args(argv)
    sys.argv = [sys.argv[0], *rest]
    args = main.parse_args()
    if args.output_dir == 'checkpoint':
        raise SystemExit('Pass --output-dir: main.py\'s default holds full-model checkpoints (test.sh uses checkpoint/no_moe/<llm>/seed_<seed>)')

    # run_training builds the model through the global name main.IPMoE, which is replaced below.
    if 'IPMoE' not in main.run_training.__code__.co_names:
        raise SystemExit('main.run_training no longer builds main.IPMoE, so test.py cannot replace it with NoMoE')

    args.arm, args.provenance = ARM, provenance()
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    planned = {'args': vars(args), 'world_size': world_size, 'global_batch_size': args.batch_size * world_size,
               'protocol': PREPROCESSING_PROTOCOL, 'source': SOURCES}

    path = os.path.join(args.output_dir, 'checkpoint.pt')
    state, differences = checkpoint_state(path, llm_type = args.llm_type, seed = args.seed)
    if state == 'other':
        raise SystemExit(f'{path} belongs to another run (saved {differences}); choose another --output-dir')

    if state == 'finished' and not own.overwrite:
        raise SystemExit(f'{path} holds a finished {ARM} run; pass --overwrite to replace it')

    if own.match is not None:
        differences = setting_differences(load_reference(own.match, args.seed), planned)
        if differences:
            raise SystemExit(f'This run would differ from {own.match} (full model, this run):\n'
                             + '\n'.join(f'  {key}: {full!r}, {arm!r}' for key, (full, arm) in differences.items())
                             + '\nTrain with the full model\'s settings on as many GPUs, or drop --match (MATCH=0 in test.sh).')

    rank = int(os.environ.get('RANK', '0'))
    if rank == 0:
        print(f'Arm {ARM}: no router or expert pool; prefix of image and interaction tokens; output {args.output_dir}; '
              + (f'settings equal to {own.match}' if own.match else 'settings not checked against a full-model run'))
        print(f'Provenance: {json.dumps(args.provenance)}')

    def build(*arguments, **keywords):
        model = NoMoE(*arguments, **keywords)

        # The backbone snapshots this run loaded, saved with the checkpoint's arguments.
        args.provenance['snapshots'] = {'clip': getattr(model.text_encoder.clip.config, '_commit_hash', None),
                                        'llm': getattr(model.llm.model.config, '_commit_hash', None)}
        if rank == 0:
            print(f'Backbone snapshots: {json.dumps(args.provenance["snapshots"])}')

        return model

    main.IPMoE = build
    main.main(args)

def state(argv):
    """Print absent, unfinished, finished, mismatched or other for a checkpoint of this arm."""
    parser = argparse.ArgumentParser(prog = 'test.py state', description = state.__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--llm-type', required = True)
    parser.add_argument('--seed', type = int, required = True)
    parser.add_argument('--match', metavar = 'FULL_CHECKPOINT', help = 'report a finished run with other settings as mismatched')
    args = parser.parse_args(argv)

    reference = load_reference(args.match, args.seed) if args.match is not None else None
    result, differences = checkpoint_state(args.checkpoint, reference, llm_type = args.llm_type, seed = args.seed)
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

    arm = load_checkpoint(checkpoint_path)['args'].get('arm')
    if arm != ARM:
        raise SystemExit(f'{checkpoint_path} is not a {ARM} checkpoint (arm={arm!r}); evaluate the full model with eval.sh')

    intervention = os.environ.get('IPMOE_INTERVENTION') or None
    if intervention is not None and intervention not in INTERVENTIONS:
        raise SystemExit(f'IPMOE_INTERVENTION={intervention!r}: this arm has no router or experts, so only {" and ".join(INTERVENTIONS)} apply')

    seed = os.environ.get('PYTHONHASHSEED')
    if seed is None:
        raise SystemExit('Set PYTHONHASHSEED to the seed, as eval.sh does')
    main.reproduce(int(seed))

    context_limit = os.environ.get('IPMOE_CONTEXT_LIMIT')
    context_limit = int(context_limit) if context_limit is not None else None

    # The adapter builds main.build_model's modules and then the class its module names IPMoE. Building NoMoE there
    # keeps the full model's checkpoint checks, prompts, context limit and generation.
    vlmeval_adapter.IPMoE = NoMoE
    run.supported_VLM['IPMoE'] = functools.partial(vlmeval_adapter.IPMoEVLMEval, checkpoint_path = checkpoint_path, context_limit = context_limit)

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

    foreign = foreign_runs(toolkit_args.work_dir)
    if foreign:
        raise SystemExit(f'{toolkit_args.work_dir} holds evaluations that test.py eval did not make ({", ".join(foreign)}); '
                         'VLMEvalKit would relink their results to the arm\'s, so choose another --work-dir')

    # Mark every run directory VLMEvalKit creates with the checkpoint it evaluates; compare reads only completed runs.
    record = {'arm': ARM, 'checkpoint': checkpoint_path, 'sha256': None, 'judge': toolkit_args.judge, 'data': toolkit_args.data,
              'intervention': intervention, 'context_limit': context_limit, 'provenance': provenance(), 'completed': False}
    if run.RANK == 0:
        record['sha256'] = file_sha256(checkpoint_path)
        print(f'Arm {ARM}: evaluating {checkpoint_path} (sha256 {record["sha256"]}); provenance {json.dumps(record["provenance"])}')

    runs = []
    upsert = run.upsert_run_status

    def mark(run_dir, *positional, **fields):
        # Mark the directory before its first status.json where it exists already: a run directory with status.json
        # but no marker is taken for another model's.
        directory = os.path.realpath(run_dir)
        first = run.RANK == 0 and directory not in runs
        if first and os.path.isdir(directory):
            runs.append(directory)
            write_json(os.path.join(directory, MARKER), record)
            first = False

        result = upsert(run_dir, *positional, **fields)
        if first:
            runs.append(directory)
            write_json(os.path.join(directory, MARKER), record)

        return result

    run.upsert_run_status = mark
    try:
        run_checked(run, toolkit_args, distributed)
    finally:
        run.upsert_run_status = upsert

    # run_checked raises on every failure it finds, so all benchmarks of these runs passed eval.py's checks.
    for directory in runs:
        write_json(os.path.join(directory, MARKER), {**record, 'completed': True})

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

def run_problem(directory, dataset, judge, digest):
    """Why the VLMEvalKit run DIRECTORY cannot supply DATASET's score, or None. DIGEST is the SHA-256 of the arm
    checkpoint the run must have evaluated, or None for the full model, whose runs must not be the arm's."""
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

    marker = read_json(os.path.join(directory, MARKER))
    if digest is None:
        return f'an evaluation of the {ARM} checkpoint {marker.get("checkpoint")}' if isinstance(marker, dict) else None

    if not isinstance(marker, dict) or not marker.get('completed'):
        return 'not completed by test.py eval: it failed one of eval.py\'s checks or was interrupted'

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

def saved_score(work, dataset, judge, digest = None):
    """(run directory, headline score in percent) of DATASET in the VLMEvalKit work directory WORK.

    The top-level files link to the latest run of each benchmark; another run is read only when it is the single run
    that passes run_problem. Summaries are not named after the judge, so the one beside the scored file is read.
    For the full model (DIGEST None), a failed evaluation can still look complete in status.json, so usable runs whose
    scores differ are reported instead of trusting the link."""
    scored, summary = result_files(dataset, judge)
    directory = os.path.join(work, 'IPMoE')
    link = os.path.join(directory, scored)
    linked = os.path.dirname(os.path.realpath(link)) if os.path.isfile(link) else None

    runs = {os.path.dirname(os.path.realpath(path)) for path in glob.glob(os.path.join(directory, 'T*', scored))} | ({linked} if linked else set())
    problems, usable = {}, []
    for candidate in sorted(runs):
        problem = run_problem(candidate, dataset, judge, digest) if os.path.isfile(os.path.join(candidate, summary)) else f'no {summary}'
        if problem:
            problems[os.path.basename(candidate)] = problem
        else:
            usable.append(candidate)

    chosen = linked if linked in usable else usable[0] if len(usable) == 1 else None
    if chosen is None:
        if usable:
            raise LookupError(f'{len(usable)} usable runs in {directory} hold {scored} and the link names none of them; link the one to use')

        raise LookupError(f'no usable {scored} in {directory}' + (': ' + '; '.join(f'{name} {problem}' for name, problem in problems.items()) if problems else ''))

    if digest is None and len(usable) > 1:
        scores = {os.path.basename(candidate): headline(os.path.join(candidate, summary), dataset) for candidate in usable}
        if max(scores.values()) - min(scores.values()) > 1e-9:
            raise LookupError(f'usable runs in {directory} disagree on {dataset} (' + '; '.join(f'{name} {score:.3f}' for name, score in scores.items())
                              + '); one may have failed eval.py\'s checks, so remove or relink the wrong one')

    return chosen, headline(os.path.join(chosen, summary), dataset)

def foreign_runs(work):
    """Run directories in the VLMEvalKit work directory WORK that test.py eval did not make."""
    return sorted(os.path.basename(os.path.dirname(path)) for path in glob.glob(os.path.join(work, 'IPMoE', 'T*', 'status.json'))
                  if not os.path.exists(os.path.join(os.path.dirname(path), MARKER)))

def pending(argv):
    """Print the benchmarks without a usable result of the arm checkpoint in its evaluation directory."""
    parser = argparse.ArgumentParser(prog = 'test.py pending', description = pending.__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('work', help = 'the evaluation work directory of the checkpoint\'s seed')
    parser.add_argument('--judge', required = True)
    parser.add_argument('--data', nargs = '+', choices = DATASETS, default = list(DATASETS))
    args = parser.parse_args(argv)

    # VLMEvalKit relinks the top-level files of every IP-MoE run in a work directory to its newest run, so evaluating
    # into a directory with other models' runs, e.g. the full model's, would replace their results.
    foreign = foreign_runs(args.work)
    if foreign:
        raise SystemExit(f'{args.work} holds evaluations that test.py eval did not make ({", ".join(foreign)}); choose another EVAL_DIR')

    digest = file_sha256(args.checkpoint)
    missing = []
    for dataset in dict.fromkeys(args.data):
        try:
            saved_score(args.work, dataset, args.judge, digest)

        except (LookupError, OSError, ValueError, KeyError, IndexError):
            missing.append(dataset)

    # A prefix, so that test.sh can tell this line, possibly empty, from anything a library prints.
    print('pending:', *missing)

def training_row(model, seed, path, checkpoint):
    saved, counts = checkpoint['args'], checkpoint.get('data_counts', {})

    return {'model': model, 'seed': seed, 'prefix_tokens': '576 + N' if model == ARM else '576 + 4N',
            'trainable_parameters': sum(tensor.numel() for tensor in checkpoint['model'].values()),
            'epoch': checkpoint.get('epoch'), 'epochs': saved.get('epochs'), 'optimizer_steps': checkpoint.get('step'),
            'world_size': checkpoint.get('world_size'), 'global_batch_size': checkpoint.get('global_batch_size'),
            'val_loss': checkpoint.get('val_loss'), **{key: counts.get(key) for key in COUNTS},
            'code': saved.get('provenance', {}).get('code'), 'checkpoint': path, 'sha256': file_sha256(path)}

def compare(argv):
    """Put the arm's benchmark scores, validation loss and training counters beside the full model's, paired by seed."""
    import pandas as pd

    parser = argparse.ArgumentParser(prog = 'test.py compare', description = compare.__doc__)
    parser.add_argument('--seeds', nargs = '+', type = int, required = True)
    parser.add_argument('--data', nargs = '+', choices = DATASETS, default = list(DATASETS), help = 'benchmarks to compare; Avg needs all seven')
    parser.add_argument('--judge', default = 'gpt-6-luna', help = 'the JUDGE both models were evaluated with')
    parser.add_argument('--full-checkpoints', required = True, help = 'directory of the full model\'s seed_N/checkpoint.pt')
    parser.add_argument('--full-results', required = True, help = 'directory of the full model\'s seed_N evaluation work directories')
    parser.add_argument('--arm-checkpoints', required = True, help = 'the same for this arm')
    parser.add_argument('--arm-results', required = True, help = 'the same for this arm')
    parser.add_argument('--output', required = True, help = 'directory for scores.csv, summary.csv, training.csv and sources.csv')
    args = parser.parse_args(argv)
    args.seeds, args.data = list(dict.fromkeys(args.seeds)), list(dict.fromkeys(args.data))

    problems, rows, training, sources, checkpoints = [], [], [], [], {}
    for model, checkpoint_root, result_root in (('full', args.full_checkpoints, args.full_results), (ARM, args.arm_checkpoints, args.arm_results)):
        for seed in args.seeds:
            row = {'model': model, 'seed': seed}
            path = os.path.join(checkpoint_root, f'seed_{seed}', 'checkpoint.pt')
            if os.path.isfile(path):
                checkpoint = checkpoints[model, seed] = load_checkpoint(path)
                problem = full_model_problem(checkpoint) if model == 'full' else (None if checkpoint['args'].get('arm') == ARM else 'it is not a no_moe run')
                if problem:
                    problems.append(f'{path}: {problem}')

                training.append(training_row(model, seed, path, checkpoint))
                row['val_loss'] = checkpoint.get('val_loss')

            else:
                problems.append(f'{model} seed {seed}: no checkpoint {path}')

            # The arm's scores must come from completed runs of its current checkpoint.
            digest = training[-1]['sha256'] if model == ARM and os.path.isfile(path) else None
            work = os.path.join(result_root, f'seed_{seed}')
            for dataset in args.data:
                if model == ARM and digest is None:
                    problems.append(f'{model} seed {seed} {dataset}: no checkpoint to check its results against')
                    continue

                try:
                    source, row[dataset] = saved_score(work, dataset, args.judge, digest)

                except (LookupError, OSError, ValueError, KeyError, IndexError) as error:
                    problems.append(f'{model} seed {seed} {dataset}: {error}')
                    continue

                status = read_json(os.path.join(source, 'status.json')) or {}
                sources.append({'model': model, 'seed': seed, 'dataset': dataset, 'score': row[dataset], 'run': source,
                                'commit': status.get('commit'), 'created_at': status.get('created_at'),
                                'judge': status.get('datasets', {}).get(dataset, {}).get('judge_model')})

            rows.append(row)

    for seed in args.seeds:
        if ('full', seed) not in checkpoints or (ARM, seed) not in checkpoints:
            continue

        full, arm = checkpoints['full', seed], checkpoints[ARM, seed]
        differences = setting_differences(full, arm)
        if differences:
            problems.append(f'seed {seed}: settings differ (full model, arm): '
                            + '; '.join(f'{key} {a!r}, {b!r}' for key, (a, b) in differences.items()))

        # The arm keeps exactly the examples the full model keeps.
        for key in ('train_budget_skipped', 'val_budget_skipped'):
            a, b = full.get('data_counts', {}).get(key), arm.get('data_counts', {}).get(key)
            if a != b:
                problems.append(f'seed {seed}: {key} is {a} for the full model but {b} for the arm')

    metrics = [*args.data, *(['Avg'] if set(args.data) == set(DATASETS) else []), 'val_loss']
    table = pd.DataFrame(rows).set_index(['model', 'seed']).reindex(columns = metrics).astype(float)
    if 'Avg' in metrics:
        # Average the seven within each seed first, as the thesis does.
        table['Avg'] = table[list(DATASETS)].mean(axis = 1, skipna = False)

    full, arm = table.xs('full', level = 'model'), table.xs(ARM, level = 'model')
    difference = arm - full
    by_seed = pd.DataFrame([{'seed': seed, 'model': model, **frame.loc[seed].to_dict()}
                            for seed in args.seeds for model, frame in (('full', full), (ARM, arm), ('difference', difference))]).set_index(['seed', 'model'])

    # Each metric over the seeds where both models have it, so the difference of the means is the mean difference.
    # The paired t interval covers the variation between training seeds only, not the sampling of benchmark questions.
    summary = {}
    for metric in metrics:
        paired = pd.DataFrame({'full': full[metric], ARM: arm[metric]}).dropna()
        change = paired[ARM] - paired['full']
        half = T975.get(len(paired) - 1, 1.96) * change.std() / math.sqrt(len(paired)) if len(paired) > 1 else math.nan
        summary[metric] = {'seeds': len(paired), 'full_mean': paired['full'].mean(), 'full_sd': paired['full'].std(),
                           f'{ARM}_mean': paired[ARM].mean(), f'{ARM}_sd': paired[ARM].std(),
                           'difference_mean': change.mean(), 'difference_sd': change.std(),
                           'difference_ci95_low': change.mean() - half, 'difference_ci95_high': change.mean() + half}
    summary = pd.DataFrame.from_dict(summary, orient = 'index')
    training = pd.DataFrame(training)
    sources = pd.DataFrame(sources, columns = ['model', 'seed', 'dataset', 'score', 'run', 'commit', 'created_at', 'judge'])

    os.makedirs(args.output, exist_ok = True)
    by_seed.to_csv(os.path.join(args.output, 'scores.csv'))
    summary.to_csv(os.path.join(args.output, 'summary.csv'), index_label = 'metric')
    training.to_csv(os.path.join(args.output, 'training.csv'), index = False)
    sources.to_csv(os.path.join(args.output, 'sources.csv'), index = False)

    with pd.option_context('display.width', 250, 'display.max_columns', None, 'display.max_colwidth', 80):
        print(f'Scores (percent) and validation loss by seed; difference = {ARM} - full; judge={args.judge}')
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

    print(f'\nWrote scores.csv, summary.csv, training.csv and sources.csv (the run behind each score) to {args.output}')
    if problems:
        print('\nIncomplete or not comparable:', *problems, sep = '\n- ', file = sys.stderr)
        raise SystemExit(1)

COMMANDS = {'train': train, 'eval': evaluate, 'compare': compare, 'state': state, 'pending': pending}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f'usage: test.py {{{",".join(COMMANDS)}}} [options]; the docstring of test.py describes each')

    COMMANDS[sys.argv[1]](sys.argv[2:])
