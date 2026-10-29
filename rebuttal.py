"""Evidence for IP_MoE_Final_Defense_Report.md that needs no training; rebuttal.sh runs each subcommand.

provenance  code revision, environment, the VLMEvalKit fork as an archive and a file inventory for authorship (R2, R3)
counts      per-arm counters, split hashes and wall time from the training logs, active parameters, prefix tokens (R4, R8)
records     result links, the commit behind every result and prediction agreement across commits (R2, R4)
h1          H1 from the saved items.csv files: tie-repaired sign, signed peaks, same box, pooling, regression (R1, R5)
scores      headline scores recomputed from per-question files and checked against VLMEvalKit's, with the seed SD (R6)
bootstrap   group-aware paired intervals of every benchmark, Avg and D against one reference checkpoint (R6, R7)
tables      Tables 4.5, 4.6, 4.10 and 4.11 and the Section 4.2.2 statistics from per-question files (R2, R7)
baselines   every evaluated baseline, reported or not, with protocol, published values and judge use (R10)
overlap     benchmark images among the training images, by SHA-256 and then perceptual hash (minor corrections)
init        the interaction MLPs a seed starts training from, saved so h1.py can measure them (R5 route A)
router      per-dataset router weights pooled over seeds, from router_weights.py's CSVs (H2)

Every output NAME gets NAME.provenance.json with the command, the code state, library versions and the SHA-256 of
each input read. The analyses only read saved outputs; none of them changes an original result.
"""

import argparse
import base64
import glob
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import fft, stats

LMU_ROOT = os.environ.get('LMUData', os.path.expanduser('~/LMUData'))
TRAIN_IMAGES = 'dataset/llava-next/images'
DATASETS = ['NaturalBenchDataset', 'CV-Bench-2D', 'CV-Bench-3D', 'MMBench_DEV_EN_V11', 'MMStar', 'POPE', 'SEEDBench_IMG']
# Expected answer of each NaturalBench row within its group of four (q0_i0, q0_i1, q1_i0, q1_i1), as VLMEvalKit scores it.
NATURALBENCH_PATTERN = (1, 0, 0, 1)
# The grounded questions behind D (thesis Section 4.3.4); every CV-Bench question is grounded, the rest of the three
# benchmarks is the comparison group. NaturalBench and POPE are not part of D.
GROUNDED = {
    'CV-Bench-2D': None,
    'CV-Bench-3D': None,
    'MMBench_DEV_EN_V11': ('category', {'object_localization', 'attribute_recognition', 'spatial_relationship',
                                        'attribute_comparison', 'physical_relation'}),
    'MMStar': ('l2_category', {'localization', 'object counting', 'single-instance reasoning',
                               'cross-instance attribute reasoning', 'cross-instance relation reasoning'}),
    'SEEDBench_IMG': ('category', {'Instance Attributes', 'Instance Location', 'Instances Counting', 'Spatial Relation',
                                   'Instance Interaction'}),
}
# H1: a Distance median this close to zero is a tie and gets the sign +1 (report Section 1). Seed 42's median is
# -3.7e-17, floating-point noise that h1.py's original rule (any negative median flips the map) treats as a direction.
TIE = 1e-9
RULES = ('tie', 'original')

## Provenance of Outputs

INPUTS = {}

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)

    return digest.hexdigest()

def track(path, digest = None):
    """Record a file this process read, for the provenance sidecar of its outputs."""
    real = os.path.realpath(path)
    if real not in INPUTS:
        INPUTS[real] = digest or file_sha256(real)

    return path

def git(*arguments):
    try:
        return subprocess.run(['git', *arguments], capture_output = True, check = True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None

@lru_cache(maxsize = 1)
def code_state():
    """The commit and what differs from it; VLMEvalKit's status.json records the commit alone, hiding uncommitted code."""
    commit = git('rev-parse', 'HEAD')
    if commit is None:
        return {'commit': None, 'note': 'not run inside the git checkout', 'script_sha256': file_sha256(__file__)}
    status = git('status', '--porcelain').decode().splitlines()
    untracked = [line[3:] for line in status if line.startswith('??')]

    return {'commit': commit.decode().strip(),
            'modified': [line[3:] for line in status if not line.startswith('??')],
            'diff_sha256': hashlib.sha256(git('diff', 'HEAD', '--binary')).hexdigest(),
            'untracked_code_sha256': {path: file_sha256(path) for path in untracked if path.endswith(('.py', '.sh')) and os.path.isfile(path)},
            'script_sha256': file_sha256(__file__)}

def libraries():
    import platform
    import scipy

    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__}
    # Only modules this process imported; importing torch just to name its version would slow every subcommand.
    for name in ('torch', 'transformers', 'PIL', 'vlmeval'):
        if name in sys.modules:
            versions[name] = getattr(sys.modules[name], '__version__', None)

    return versions

def clean(value):
    """JSON-safe: numpy scalars as Python values, NaN and infinity as null."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value

def write(output, name, payload):
    os.makedirs(output, exist_ok = True)
    if isinstance(payload, pd.DataFrame):
        payload.to_csv(os.path.join(output, f'{name}.csv'), index = False)
        print(payload.to_string())
    else:
        payload = clean(payload)
        with open(os.path.join(output, f'{name}.json'), 'w') as handle:
            json.dump(payload, handle, indent = 2, allow_nan = False)
        print(json.dumps(payload, indent = 2))
    sidecar = {'command': sys.argv, 'code': code_state(), 'libraries': libraries(), 'inputs': dict(sorted(INPUTS.items()))}
    with open(os.path.join(output, f'{name}.provenance.json'), 'w') as handle:
        json.dump(clean(sidecar), handle, indent = 2)

## Code, Environment, Fork and File Inventory

# The fork's .env holds the judge's API key; caches and toolkit outputs are not source.
FORK_EXCLUDE = ('.env', '.git', '__pycache__', 'outputs')
UNCONFIRMED = {'origin': 'unknown (author to confirm: written, adapted, generated or mixed)', 'source_version_license': '',
               'generation_tool_purpose': '', 'verification': ''}
GROUPS = [('VLMEvalKit/', 'fork of OpenCompass VLMEvalKit; files in fork/manifest.csv'),
          ('defense/draft/tex/', 'thesis text'), ('defense/draft/figs/', 'figures'), ('defense/draft/main.bib', 'bibliography'),
          ('checkpoint/', 'trained weights and training logs'),
          ('evaluation/', 'outputs; producing code in each status.json or provenance sidecar')]

def environment():
    from importlib import metadata

    packages = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'flash-attn', 'numpy', 'scipy', 'pandas', 'pillow', 'vlmeval'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output = True, text = True)
    gpus = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                          capture_output = True, text = True) if shutil.which('nvidia-smi') else None

    from importlib.util import find_spec

    # PYTHONPATH puts the fork first; pip's metadata may describe another installed copy.
    spec = find_spec('vlmeval')
    return {'python': sys.version, 'executable': sys.executable, 'packages': packages,
            'vlmeval_imported_from': spec.origin if spec else None,
            'pip_freeze': freeze.stdout.splitlines() if freeze.returncode == 0 else None,
            'gpus': gpus.stdout.strip().splitlines() if gpus is not None and gpus.returncode == 0 else None,
            'note': 'The environment that ran this analysis now. Training and evaluation did not record theirs; '
                    'treat this as the current environment, not as a historical record.'}

def archive_fork(root, directory):
    """The fork exactly as it is on disk, since it is not under version control, without its API key file and caches."""
    rows = []
    for parent, folders, files in os.walk(root):
        folders[:] = sorted(folder for folder in folders if folder not in FORK_EXCLUDE)
        for file in sorted(files):
            path = os.path.join(parent, file)
            if file in FORK_EXCLUDE or file.endswith('.pyc') or os.path.islink(path) or not os.path.isfile(path):
                continue
            stat = os.stat(path)
            rows.append({'path': os.path.relpath(path, root), 'bytes': stat.st_size, 'sha256': file_sha256(path),
                         'modified': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec = 'seconds')})

    os.makedirs(directory, exist_ok = True)
    archive = os.path.join(directory, 'VLMEvalKit.tar.gz')
    with open(archive, 'wb') as raw, gzip.GzipFile(fileobj = raw, mode = 'wb', mtime = 0) as zipped, \
            tarfile.open(fileobj = zipped, mode = 'w') as tar:
        for row in rows:
            tar.add(os.path.join(root, row['path']), arcname = os.path.join('VLMEvalKit', row['path']), recursive = False)
    write(directory, 'manifest', pd.DataFrame(rows))
    listing = ''.join(f'{row["path"]}\t{row["sha256"]}\n' for row in rows)

    return {'root': root, 'archive': archive, 'archive_sha256': file_sha256(archive), 'files': len(rows),
            'content_sha256': hashlib.sha256(listing.encode()).hexdigest(), 'excluded': list(FORK_EXCLUDE),
            'newest_modification': max(row['modified'] for row in rows) if rows else None,
            'note': 'No upstream commit is recorded locally. This archive is the fork itself; a patch against a pinned '
                    'upstream commit needs that commit identified first. Modification times are evidence of when files '
                    'last changed, not proof.'}

def file_inventory():
    """One row per repository file and per group outside git; origins stay unknown until the author confirms them."""
    listed = git('ls-files', '-co', '--exclude-standard', '-z') or b''
    paths = sorted(path for path in listed.decode().split('\0') if path and os.path.isfile(path))
    history, commit = {}, None
    for line in (git('log', '--reverse', '--format=commit %h %aI %an', '--name-only') or b'').decode().splitlines():
        if line.startswith('commit '):
            commit = line[len('commit '):]
        elif line and commit:
            history.setdefault(line, {'first': commit})['last'] = commit

    modified = set(code_state().get('modified', []))
    rows = [{'entry': path, 'kind': 'file', 'bytes': os.path.getsize(path), 'sha256': file_sha256(path),
             'git_first': history.get(path, {}).get('first', 'untracked'), 'git_last': history.get(path, {}).get('last', 'untracked'),
             'uncommitted_changes': path in modified, **UNCONFIRMED} for path in paths]
    rows += [{'entry': entry, 'kind': f'group: {description}', 'bytes': None, 'sha256': None, 'git_first': 'not in git',
              'git_last': 'not in git', 'uncommitted_changes': None, **UNCONFIRMED} for entry, description in GROUPS if os.path.exists(entry)]

    return pd.DataFrame(rows)

def provenance(args):
    report = {'code': code_state(), 'environment': environment(),
              'fork': archive_fork(args.fork, os.path.join(args.output, 'fork'))}
    if args.package:
        digest = file_sha256(track(args.package))
        report['package'] = {'path': args.package, 'sha256': digest, 'reviewed_sha256': args.reviewed_sha256,
                             'is_reviewed_archive': digest == args.reviewed_sha256 if args.reviewed_sha256 else None}
    inventory = file_inventory()
    report['inventory'] = {'entries': len(inventory), 'file': 'provenance_inventory.csv',
                           'note': 'git authorship records who committed a file, not who wrote, adapted or generated it. '
                                   'Fill in origin per entry; keep unresolved origins as unknown rather than inferring them.'}
    write(args.output, 'provenance_inventory', inventory)
    write(args.output, 'provenance', report)

## Checkpoints and Training Logs

def checkpoints(root):
    """Every trained checkpoint: full seeds per language model, then each ablation arm's seeds."""
    paths = sorted(glob.glob(os.path.join(root, 'llama', 'seed_*', 'checkpoint.pt')))
    paths += sorted(glob.glob(os.path.join(root, 'qwen', 'seed_*', 'checkpoint.pt')))
    paths += sorted(glob.glob(os.path.join(root, 'ablations', '*', 'seed_*', 'checkpoint.pt')))

    return paths

def seconds(elapsed):
    value = 0
    for part in elapsed.split(':'):
        value = 60 * value + int(part)

    return value

def train_log(directory, steps):
    """What the run's train.log states; a missing log is reported missing, not reconstructed."""
    path = os.path.join(directory, 'train.log')
    if not os.path.isfile(path):
        return {'train_log': None}
    with open(track(path), errors = 'replace') as handle:
        text = handle.read()

    def distinct(pattern):
        return ' '.join(sorted(set(re.findall(pattern, text)))) or None

    # Each rank prints its training bar; the slowest rank's total is the run's wall time.
    elapsed = [seconds(value) for count, value in re.findall(r'(\d+)/\1 \[((?:\d+:)?\d+:\d+)<', text) if steps and int(count) == steps]
    devices = re.findall(r'Using device: \S+, processes: (\d+), global batch size: (\d+)', text)
    skipped = re.findall(r'Skipped training examples: (\d+), sampler padding: (\d+)', text)

    return {'train_log': path,
            'log_train_ids_sha256': distinct(r'Training sample IDs SHA256: ([0-9a-f]{64})'),
            'log_val_ids_sha256': distinct(r'Validation sample IDs SHA256: ([0-9a-f]{64})'),
            'log_split': distinct(r'first-turn split: (train=[\d,]+ of [\d,]+, val=[\d,]+; validation_seed=\d+)'),
            'log_processes': int(devices[-1][0]) if devices else None,
            'log_global_batch_size': int(devices[-1][1]) if devices else None,
            'log_budget_skipped': int(skipped[-1][0]) if skipped else None,
            'log_sampler_padding': int(skipped[-1][1]) if skipped else None,
            'log_wall_seconds': max(elapsed) if elapsed else None}

def prefix_tokens(lengths, patch_size, experts, image_only, image_tokens = 576):
    """Visual prefix of IPMoE._prefix_lengths: image tokens plus the interaction and expert tokens of an L x 576 map."""
    if image_only:
        return np.full(len(lengths), image_tokens)
    interaction = -(-np.maximum(lengths, 1) // patch_size) * -(-image_tokens // patch_size)

    return image_tokens + (1 + experts) * interaction

def prompt_lengths(run, judge):
    """CLIP content tokens of every evaluation prompt, rebuilt from RUN's predictions as the adapter builds them."""
    from transformers import AutoTokenizer

    from model.encoder import MODEL_NAME
    from vlmeval_adapter import SHORT_ANSWER_INSTRUCTION, mcq_prompt

    # NaturalBench is prompted by VLMEvalKit's NaturalBenchDataset.build_prompt, not by the adapter.
    naturalbench_suffix = {'yes_no': 'Please answer Yes or No.',
                           'multiple_choice': 'Please output the letter corresponding to the correct option.'}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast = True)
    lengths = {}
    for dataset in DATASETS:
        path = os.path.join(os.path.dirname(result_file(run, dataset, judge)), f'IPMoE_{dataset}.xlsx')
        table = pd.read_excel(track(path))
        if dataset == 'NaturalBenchDataset':
            types = table['type'] if 'type' in table else table['index'].map(naturalbench_types(os.path.join(LMU_ROOT, 'NaturalBenchDataset.tsv')))
            prompts = [f'{question} {naturalbench_suffix[kind]}' for question, kind in zip(table['question'], types)]
        elif dataset == 'POPE':
            prompts = [f'{question}\n{SHORT_ANSWER_INSTRUCTION}' for question in table['question']]
        else:
            prompts = [mcq_prompt(row) for _, row in table.iterrows()]
        ids = tokenizer(prompts, truncation = False, verbose = False)['input_ids']
        lengths[dataset] = np.maximum(np.array([len(row) - 2 for row in ids]), 1)

    return lengths

def counts(args):
    import torch

    from ablation import ABLATIONS

    rows = []
    for path in checkpoints(args.checkpoint_root):
        # One read serves both the hash and the load.
        with open(path, 'rb') as handle:
            data = handle.read()
        track(path, hashlib.sha256(data).hexdigest())
        checkpoint = torch.load(io.BytesIO(data), map_location = 'cpu', weights_only = True)
        del data

        saved = Namespace(**checkpoint['args'])
        ablation = getattr(saved, 'ablation', 'full')
        configuration = getattr(saved, 'ablation_config', None) or ABLATIONS['full']
        image_only = configuration.get('prefix', 'full') == 'image'
        experts = 0 if image_only else 3 if configuration.get('expert_set', 'all') == 'all' else 1
        data_counts = checkpoint.get('data_counts', {})
        row = {'checkpoint': path, 'sha256': INPUTS[os.path.realpath(path)], 'llm_type': saved.llm_type, 'ablation': ablation,
               'seed': saved.seed, 'epoch': checkpoint.get('epoch'), 'step': checkpoint.get('step'),
               'optimizer_steps_this_epoch': checkpoint.get('optimizer_steps_this_epoch'),
               'world_size': checkpoint.get('world_size'), 'global_batch_size': checkpoint.get('global_batch_size'),
               **data_counts, 'val_loss': checkpoint.get('val_loss'),
               # Exposures include distributed-sampler padding; unique examples are known only when nothing was skipped.
               'training_exposures': data_counts.get('train_retained_this_epoch'),
               'unique_training_examples': data_counts.get('train_split') if data_counts.get('train_budget_skipped') == 0 else None,
               'protocol': checkpoint.get('protocol'),
               'data_revision': json.dumps(checkpoint.get('source'), sort_keys = True, default = list),
               'active_parameters': int(sum(tensor.numel() for tensor in checkpoint['model'].values())),
               'patch_size': saved.patch_size, 'experts': experts, 'image_only': image_only,
               'prefix_tokens': '576' if image_only else f'576 + {1 + experts}N'}
        logits = checkpoint['model'].get('moe.router_logits')
        if logits is not None:
            row.update({f'static_{name}': float(weight) for name, weight in zip(('self', 'cross', 'hybrid'), torch.softmax(logits.float(), dim = -1))})
        row.update(train_log(os.path.dirname(path), checkpoint.get('optimizer_steps_this_epoch')))
        rows.append(row)
        del checkpoint

    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit(f'No checkpoints under {args.checkpoint_root}')

    # Equal counts do not establish equal identities; the logged split hash does, where a log survives.
    logged = table['log_train_ids_sha256'].dropna()
    majority = logged.mode().iloc[0] if len(logged) else None
    table['same_split_ids_as_other_logs'] = table['log_train_ids_sha256'].map(lambda value: None if value is None or pd.isna(value) else value == majority)
    table['identity_evidence'] = [
        ('split ID hash in train.log; zero budget skips, so every split example was retained'
         if row.log_train_ids_sha256 == majority and row.train_budget_skipped == 0 else
         'no train.log: counts, protocol and data revision only; retained identities unverified'
         if row.train_log is None else 'train.log split hash differs or examples were skipped; inspect')
        for row in table.itertuples()]
    full = table[(table['llm_type'] == 'llama') & (table['ablation'] == 'full')]
    for column in ('train_budget_skipped', 'train_retained_this_epoch', 'optimizer_steps_this_epoch'):
        if column in table and len(full):
            table[f'{column}_minus_full_mean'] = table[column] - full[column].mean()
    write(args.output, 'checkpoint_counts', table)

    if args.prompts:
        lengths = prompt_lengths(args.prompts, args.judge)
        arms = table.drop_duplicates(['llm_type', 'ablation'])
        prefix = []
        for arm in arms.itertuples():
            for dataset, values in [*lengths.items(), ('all', np.concatenate(list(lengths.values())))]:
                tokens = prefix_tokens(values, arm.patch_size, arm.experts, arm.image_only)
                prefix.append({'llm_type': arm.llm_type, 'ablation': arm.ablation, 'dataset': dataset, 'prompts': len(values),
                               'prefix_formula': arm.prefix_tokens, 'mean': tokens.mean(), 'median': np.median(tokens),
                               'p95': np.percentile(tokens, 95), 'max': tokens.max(), 'active_parameters': arm.active_parameters})
        write(args.output, 'prefix_tokens', pd.DataFrame(prefix))

## Result Files

def model_name(run):
    """IP-MoE runs keep their results in RUN/IPMoE; a baseline's directory is named after its model."""
    return 'IPMoE' if os.path.isdir(os.path.join(run, 'IPMoE')) else os.path.basename(os.path.normpath(run))

def result_names(model, dataset, judge):
    """The scored per-question file and the headline summary VLMEvalKit writes for one benchmark."""
    base = f'{model}_{dataset}'
    if dataset == 'NaturalBenchDataset':
        return base + '.xlsx', base + '_acc.csv'
    if dataset == 'POPE':
        return base + '_auxmatch.xlsx', base + '_score.csv'

    return f'{base}_{judge}_result.xlsx', base + '_acc.csv'

@lru_cache(maxsize = 1024)
def analysis_artifacts(run, dataset, judge):
    """Select scored rows and their colocated summary once per analysis process.

    A valid convenience link selects its actual dated source. Broken links may
    fall back to one complete dated pair; choosing among several would invent
    provenance, so that case requires an explicit repair outside this analysis.
    """
    directory = os.path.join(run, model_name(run))
    scored, metric_name = result_names(model_name(run), dataset, judge)

    def pair(result):
        source = os.path.realpath(result)
        metric = os.path.join(os.path.dirname(source), metric_name)
        if not os.path.isfile(metric) or os.path.dirname(os.path.realpath(metric)) != os.path.dirname(source):
            raise ValueError(f'{result}: missing summary in the same result directory ({metric_name})')
        return track(source), track(os.path.realpath(metric))

    preferred = os.path.join(directory, scored)
    if os.path.isfile(preferred):
        return pair(preferred)
    candidates, incomplete = {}, []
    for result in sorted(glob.glob(os.path.join(directory, 'T*', scored))):
        if os.path.isfile(result):
            try:
                resolved = pair(result)
                candidates[resolved[0]] = resolved
            except ValueError as error:
                incomplete.append(str(error))
    if len(candidates) > 1:
        raise ValueError(f'{run} {dataset}: ambiguous dated results; repair the scored-file link explicitly: {list(candidates)}')
    if not candidates:
        detail = '; '.join(incomplete) if incomplete else f'no scored file {scored}'
        raise ValueError(f'{run} {dataset}: no complete scored-result/summary pair: {detail}')
    selected = next(iter(candidates.values()))
    print(f'{run} {dataset}: using existing dated results {selected[0]}')

    return selected

def result_file(run, dataset, judge):
    return analysis_artifacts(run, dataset, judge)[0]

@lru_cache(maxsize = 4)
def naturalbench_types(path):
    """Legacy predictions may omit type; scan the large TSV at most once."""
    table = pd.read_csv(path, sep = '\t', usecols = ['index', 'type'])
    if table['index'].isna().any() or table['index'].duplicated().any() or table['type'].isna().any():
        raise ValueError(f'{path}: missing or duplicate NaturalBench metadata')

    return table.set_index('index')['type']

def binary(values, label):
    values = pd.to_numeric(values, errors = 'raise').to_numpy(dtype = float)
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError(f'{label}: expected finite binary values')

    return values.astype(int)

def validate_hits(table, dataset):
    if table.empty:
        raise ValueError(f'{dataset}: empty results')
    indices = pd.to_numeric(table['index'], errors = 'raise')
    if not np.isfinite(indices).all() or (indices % 1 != 0).any() or indices.duplicated().any():
        raise ValueError(f'{dataset}: indices must be finite, integral and unique')
    table['index'] = indices.astype(np.int64)
    table['hit'] = binary(table['hit'], f'{dataset} hit')

    return table.sort_values('index').reset_index(drop = True)

def load_hits(run, dataset, judge):
    """One row per scored question; NaturalBench metadata normally lives in the XLSX."""
    table = pd.read_excel(result_file(run, dataset, judge))
    if dataset == 'NaturalBenchDataset':
        from vlmeval.dataset.utils.naturalbench import extract_answer

        table = table.sort_values('index').reset_index(drop = True)
        if len(table) % 4 or not len(table):
            raise ValueError(f'{run}: NaturalBench requires complete groups of four')
        if 'type' in table:
            types = table['type']
        else:
            metadata = naturalbench_types(os.path.join(LMU_ROOT, 'NaturalBenchDataset.tsv'))
            types = table['index'].map(metadata)
        if types.isna().any():
            raise ValueError(f'{run}: NaturalBench answer types are missing')
        table['extracted'] = [extract_answer(str(prediction), answer_type)
                              for prediction, answer_type in zip(table['prediction'], types)]
        table['group'] = np.arange(len(table)) // 4
        table['hit'] = (table['extracted'].to_numpy() == np.tile(NATURALBENCH_PATTERN, len(table) // 4)).astype(int)
    elif dataset == 'POPE':
        table['hit'] = binary(table['score'], f'{run} POPE score')
        if table[['extracted', 'answer', 'category']].isna().any().any():
            raise ValueError(f'{run}: POPE scoring metadata is missing')
        if not table['answer'].isin(['Yes', 'No']).all():
            raise ValueError(f'{run}: unexpected POPE answer labels')
    table = validate_hits(table, dataset)
    if dataset == 'NaturalBenchDataset' and (np.diff(table['index']) != 1).any():
        raise ValueError(f'{run}: NaturalBench rows do not form contiguous four-row groups')

    return table

def grounded_mask(table, dataset):
    rule = GROUNDED[dataset]
    if rule is None:
        return np.ones(len(table), dtype = bool)
    column, names = rule
    if column not in table or table[column].isna().any():
        raise ValueError(f'{dataset}: {column} labels are missing')

    return table[column].isin(names).to_numpy()

def units(table, dataset):
    """Arrays used by the headline metric; NB has one entry per four-row group."""
    if dataset == 'NaturalBenchDataset':
        q0_i0, q0_i1, q1_i0, q1_i1 = table['extracted'].to_numpy().reshape(-1, 4).T
        question = ((q0_i0 == 1) & (q0_i1 == 0)).astype(int) + ((q1_i1 == 1) & (q1_i0 == 0))
        image = ((q0_i0 == 1) & (q1_i0 == 0)).astype(int) + ((q1_i1 == 1) & (q0_i1 == 0))
        return {'Q_Acc': question / 2, 'I_Acc': image / 2, 'Acc': table['hit'].to_numpy().reshape(-1, 4).mean(axis = 1),
                'G_Acc': ((question == 2) & (image == 2)).astype(float)}
    if dataset == 'POPE':
        return {'predicted': (table['extracted'].astype(str) == 'Yes').to_numpy(),
                'actual': (table['answer'].astype(str) == 'Yes').to_numpy(),
                'splits': table['category'].astype(str).str.count(',').to_numpy() + 1}
    arrays = {'hit': table['hit'].to_numpy(dtype = float)}
    if dataset in ('CV-Bench-2D', 'CV-Bench-3D'):
        stratum = table['source' if dataset == 'CV-Bench-2D' else 'category']
        if stratum.isna().any():
            raise ValueError(f'{dataset}: source/task labels are missing')
        arrays['stratum'] = pd.factorize(stratum)[0]
    if dataset in GROUNDED:
        arrays['grounded'] = grounded_mask(table, dataset)

    return arrays

def headline(arrays, dataset, naturalbench_metric, draw = None):
    take = (lambda values: values) if draw is None else (lambda values: values[draw])
    if dataset == 'NaturalBenchDataset':
        return 100 * take(arrays[naturalbench_metric]).mean()
    if dataset == 'POPE':
        predicted, actual, splits = take(arrays['predicted']), take(arrays['actual']), take(arrays['splits'])
        denominator = (splits * predicted).sum() + (splits * actual).sum()
        return 100 * 2 * (splits * (predicted & actual)).sum() / denominator if denominator else 0.0
    if 'stratum' in arrays:
        stratum, hit = take(arrays['stratum']), take(arrays['hit'])
        count = np.bincount(stratum, minlength = int(arrays['stratum'].max()) + 1)
        if (count == 0).any():
            raise ValueError(f'{dataset}: resample omitted an entire source/task')
        return 100 * np.mean(np.bincount(stratum, hit, minlength = len(count)) / count)

    return 100 * take(arrays['hit']).mean()

def split_accuracy(arrays, draws = None):
    """Question-weighted accuracy on the grounded and on the other questions of the five benchmarks behind D."""
    totals = np.zeros(4)
    for dataset in GROUNDED:
        hit, grounded = arrays[dataset]['hit'], arrays[dataset]['grounded']
        if draws is not None:
            hit, grounded = hit[draws[dataset]], grounded[draws[dataset]]
        totals += (hit[grounded].sum(), grounded.sum(), hit[~grounded].sum(), (~grounded).sum())

    return 100 * totals[0] / totals[1], 100 * totals[2] / totals[3]

def d_score(run, reference):
    """D: the loss relative to the reference on grounded questions minus the loss on the other questions."""
    return (reference['Grounded'] - run['Grounded']) - (reference['Other'] - run['Other'])

def saved_headline(metric, dataset, naturalbench_metric):
    if dataset == 'POPE':
        score = pd.read_csv(metric)
        return float(score.loc[score['split'] == 'Overall', 'Overall'].iloc[0])
    if dataset == 'NaturalBenchDataset':
        return 100 * float(pd.read_csv(metric).set_index('Metric')['Score'][naturalbench_metric])

    return 100 * float(pd.read_csv(metric)['Overall'].iloc[0])

def reported(run, dataset, naturalbench_metric, judge = 'gpt-6-luna'):
    """VLMEvalKit's saved headline score; absence is an error, not an unchecked pass."""
    return saved_headline(analysis_artifacts(run, dataset, judge)[1], dataset, naturalbench_metric)

def validated(run, dataset, judge, naturalbench_metric):
    """One benchmark's per-question rows, once their recomputed headline matches the saved toolkit score."""
    table = load_hits(run, dataset, judge)
    actual = float(headline(units(table, dataset), dataset, naturalbench_metric))
    try:
        expected = reported(run, dataset, naturalbench_metric, judge)
    except (OSError, KeyError, IndexError) as error:
        raise ValueError(f'{run} {dataset}: cannot validate against its saved toolkit score') from error
    if not np.isfinite([actual, expected]).all() or abs(actual - expected) > 1e-6:
        raise ValueError(f'{run} {dataset}: recomputed {actual} disagrees with toolkit {expected}')

    return table

def run_scores(tables, naturalbench_metric, draws = None):
    arrays = {dataset: units(table, dataset) if isinstance(table, pd.DataFrame) else table for dataset, table in tables.items()}
    scores = {dataset: float(headline(arrays[dataset], dataset, naturalbench_metric, None if draws is None else draws[dataset]))
              for dataset in DATASETS if dataset in arrays}
    if all(dataset in scores for dataset in DATASETS):
        scores['Avg'] = float(np.mean([scores[dataset] for dataset in DATASETS]))
    if all(dataset in arrays for dataset in GROUNDED):
        scores['Grounded'], scores['Other'] = (float(value) for value in split_accuracy(arrays, draws))

    return scores

def validated_runs(paths, judge, naturalbench_metric):
    """Every statistics entry point verifies metric agreement before drawing conclusions."""
    runs = {run: {dataset: validated(run, dataset, judge, naturalbench_metric) for dataset in DATASETS} for run in dict.fromkeys(paths)}
    if not runs:
        raise ValueError('No result runs supplied')

    return runs, {run: run_scores(tables, naturalbench_metric) for run, tables in runs.items()}

def chosen_options(table):
    """The option letter VLMEvalKit's rule-based matcher reads from each prediction; None where only the judge could."""
    from vlmeval.utils import can_infer

    letters = [letter for letter in 'ABCDEFGH' if letter in table]
    chosen = []
    for _, row in table.iterrows():
        options = {letter: str(row[letter]) for letter in letters if not pd.isna(row[letter])}
        choice = can_infer(str(row['prediction']), options)
        chosen.append(choice or None)

    return pd.Series(chosen, index = table['index'].to_numpy())

## Evaluation Records and Commits

# Code an IP-MoE evaluation runs; the VLMEvalKit fork is not under version control and is identified by provenance.
INFERENCE_CODE = ['eval.py', 'eval.sh', 'vlmeval_adapter.py', 'ablation.py', 'main.py', 'model']

def dated_status(directory):
    path = os.path.join(directory, 'status.json')
    if not os.path.isfile(path):
        return {}
    with open(track(path)) as handle:
        return json.load(handle)

def compare_predictions(first, second, dataset):
    """Row-by-row agreement of two prediction files for the same questions, with each file's saved headline."""
    a, b = (pd.read_excel(track(path))[['index', 'prediction']] for path in (first, second))
    merged = a.merge(b, on = 'index', how = 'outer', suffixes = ('_first', '_second'), indicator = True)
    both = merged[merged['_merge'] == 'both']
    result = {'dataset': dataset, 'first': first, 'second': second, 'questions_both': len(both),
              'only_first': int((merged['_merge'] == 'left_only').sum()), 'only_second': int((merged['_merge'] == 'right_only').sum()),
              'different_predictions': int((both['prediction_first'].astype(str) != both['prediction_second'].astype(str)).sum())}
    for side, path in (('first', first), ('second', second)):
        directory = os.path.dirname(path)
        status = dated_status(directory)
        metric = os.path.join(directory, result_names(os.path.basename(path).split('_' + dataset)[0], dataset, 'judge')[1])
        result[f'{side}_commit'] = status.get('commit')
        result[f'{side}_created_at'] = status.get('created_at')
        result[f'{side}_headline'] = saved_headline(track(metric), dataset, 'Acc') if os.path.isfile(metric) else None

    return result

def predictions_file(run, dataset, judge):
    return os.path.join(os.path.dirname(result_file(run, dataset, judge)), f'{model_name(run)}_{dataset}.xlsx')

def log_segments(run):
    """Invocations in RUN's eval.log: banner time, datasets and the inference context the adapter itself logged."""
    segments = []
    for path in sorted(glob.glob(os.path.join(run, 'eval.log*'))):
        with open(track(path), errors = 'replace') as handle:
            for line in handle:
                banner = re.match(r'=== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) .*data=\[([^\]]*)\]', line)
                if banner:
                    segments.append({'time': datetime.fromisoformat(banner.group(1)).replace(tzinfo = timezone.utc),
                                     'datasets': banner.group(2).split(), 'context': None})
                elif segments and segments[-1]['context'] is None:
                    context = re.search(r'Context tokens: inference=(\d+)', line)
                    if context:
                        segments[-1]['context'] = int(context.group(1))

    return segments

@lru_cache(maxsize = None)
def adapter_logs_context(commit):
    """Whether the adapter at COMMIT logs its inference context; before 1d96b902 it applied the 4,096-token training cap."""
    source = git('show', f'{commit}:vlmeval_adapter.py')

    return None if source is None else b'Context tokens' in source

def inference_context(segments, created_at, dataset, commit):
    """The context the run's own log shows, and whether the recorded commit's adapter could have produced that log."""
    try:
        created = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return None, 'no creation time', None
    # The banner precedes VLMEvalKit's status by seconds to minutes.
    matching = [segment for segment in segments if dataset in segment['datasets'] and 0 <= (created - segment['time']).total_seconds() <= 900]
    if not matching:
        return None, 'log segment missing', None
    context = matching[-1]['context']
    logs = adapter_logs_context(commit) if commit else None

    return (context, matching[-1]['time'].isoformat(),
            None if logs is None else (context is not None) == logs)

def records(args):
    os.makedirs(args.output, exist_ok = True)
    links, results, agreement = [], [], []
    for run in args.runs:
        model = model_name(run)
        root = os.path.join(run, model)
        if not os.path.isdir(root):
            continue
        segments = log_segments(run) if model == 'IPMoE' else []
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.islink(path):
                dated = os.path.dirname(os.path.realpath(path))
                status = dated_status(dated) if os.path.isdir(dated) else {}
                links.append({'run': run, 'link': name, 'target': os.readlink(path), 'exists': os.path.exists(path),
                              'dated': os.path.basename(dated), 'commit': status.get('commit'), 'created_at': status.get('created_at')})
        for dataset in DATASETS:
            candidates = sorted(glob.glob(os.path.join(root, 'T*', f'{model}_{dataset}.xlsx')))
            try:
                selected, error = os.path.dirname(result_file(run, dataset, args.judge)), None
            except ValueError as exception:
                selected, error = None, str(exception)
            for candidate in candidates:
                directory = os.path.dirname(candidate)
                status = dated_status(directory)
                context, segment, consistent = (inference_context(segments, status.get('created_at'), dataset, status.get('commit'))
                                                if model == 'IPMoE' else (None, None, None))
                results.append({'run': run, 'dataset': dataset, 'dated': os.path.basename(directory),
                                'selected': selected is not None and os.path.realpath(directory) == os.path.realpath(selected),
                                'commit': status.get('commit'), 'created_at': status.get('created_at'),
                                'judge': status.get('datasets', {}).get(dataset, {}).get('judge_model'),
                                'logged_inference_context': context, 'log_segment': segment, 'log_matches_commit': consistent, 'error': None})
            if not candidates or selected is None:
                results.append({'run': run, 'dataset': dataset, 'dated': None, 'selected': False, 'commit': None,
                                'created_at': None, 'judge': None, 'logged_inference_context': None, 'log_segment': None,
                                'log_matches_commit': None, 'error': error or 'no predictions'})
            # A run directory holding the same benchmark twice is a rerun of the same checkpoint.
            for first, second in combinations(candidates, 2):
                agreement.append({'run': run, **compare_predictions(first, second, dataset)})

    results = pd.DataFrame(results)
    selected = results[results['selected']].set_index(['run', 'dataset'])['commit']
    crossings = []
    if args.reference:
        for (run, dataset), commit in selected.items():
            reference = selected.get((args.reference, dataset))
            if run != args.reference and reference is not None and commit != reference:
                crossings.append({'run': run, 'dataset': dataset, 'commit': commit, 'reference_commit': reference})
    commits = sorted({commit for commit in results['commit'].dropna()},
                     key = lambda commit: (git('show', '-s', '--format=%cI', commit) or commit.encode()))
    diffs = {}
    for first, second in zip(commits, commits[1:]):
        diff = git('diff', first, second, '--', *INFERENCE_CODE)
        if diff is not None:
            name = f'records_diff_{first}_{second}.patch'
            with open(os.path.join(args.output, name), 'wb') as handle:
                handle.write(diff)
            diffs[f'{first}..{second}'] = {'files': (git('diff', '--stat', first, second, '--', *INFERENCE_CODE) or b'').decode().strip(),
                                           'patch': name}

    write(args.output, 'records_links', pd.DataFrame(links))
    write(args.output, 'records_results', results)
    write(args.output, 'records_agreement', pd.DataFrame(agreement))
    write(args.output, 'records_commits', {
        'reference': args.reference, 'crossings': crossings, 'commits': commits, 'diffs': diffs,
        'broken_links': int(sum(not link['exists'] for link in links)),
        'code_differs_from_recorded_commit': int((results['log_matches_commit'] == False).sum()) if 'log_matches_commit' in results else 0,
        'notes': ['status.json records the commit only, so results made from a modified working tree are not tied to exact code. '
                  'log_matches_commit is False where the run\'s own log shows an inference context its recorded commit cannot log.',
                  'Before 1d96b902 the adapter capped inference at the 4,096-token training context and raised on longer '
                  'prompts, so a capped run either finished with prompts that fit (identical inputs to an uncapped run) or '
                  'failed; prefix_tokens.csv shows which benchmarks can exceed the cap at all.',
                  'VLMEvalKit is not under version control; the provenance step archives and hashes it.',
                  'A link to a dated run created after the thesis results points at a reconstruction; identify the intended '
                  'run before repointing it, and label reconstructed results as such.',
                  'Identical predictions from two commits show that the code change did not alter inference for those '
                  'questions; rescoring alone would not show that.']})

## H1

def h1_sign(calibration, rule):
    """A map's sign, chosen on its Distance questions so that the Depth test is not circular."""
    median = float(np.median(calibration)) if len(calibration) else 0.0
    if rule == 'tie':
        return (1 if median > 0 or abs(median) < TIE else -1), median

    return (-1 if median < 0 else 1), median

def h1_load(directory):
    with open(track(os.path.join(directory, 'summary.json'))) as handle:
        summary = json.load(handle)
    if 'pooled' in summary:
        raise ValueError(f'{directory} is pooled; pass its per-seed runs instead')
    items = pd.read_csv(track(os.path.join(directory, 'items.csv')))
    for column in ('own_top', 'own_top_neg', 'peak_box', 'peak_box_neg', 'box_fraction'):
        items[column] = items[column].map(json.loads)

    return summary, items

@lru_cache(maxsize = 2)
def h1_images(data):
    """Source image of each CV-Bench-3D question; questions on one image differ only in the boxes drawn on it."""
    table = pd.read_csv(track(data), sep = '\t', usecols = ['index', 'source_dataset', 'source_filename'])

    return dict(zip(table['index'].astype(int), table['source_dataset'] + '/' + table['source_filename']))

def wilcoxon_greater(d):
    d = np.asarray(d, dtype = float)
    return float(stats.wilcoxon(d, alternative = 'greater').pvalue) if (d != 0).any() else None

def describe_d(d, images):
    """The pooled statistics of a signed d series; the p-value is a signed-rank test, not a test of the mean alone."""
    image_d = d.groupby(d.index.map(images)).mean()

    return {'questions': len(d), 'mean_d': float(d.mean()), 'median_d': float(d.median()), 'd_positive': float((d > 0).mean()),
            'd_exact_zero': int((d == 0).sum()), 'd_near_zero': int(((d != 0) & (d.abs() < TIE)).sum()),
            'wilcoxon_p': wilcoxon_greater(d), 'images': len(image_d), 'per_image_wilcoxon_p': wilcoxon_greater(image_d)}

def h1_regression(depth, d, hits):
    """h1.py's regression of correctness on signed d, the 2D cue and log box size, over the Depth questions."""
    from h1 import logistic

    hit = hits.reindex(depth.index)
    if hit.isna().any():
        return {'error': f'predictions lack {int(hit.isna().sum())} Depth questions'}
    size = np.log([np.mean(fractions) for fractions in depth['box_fraction']])
    fit = logistic(hit.to_numpy(dtype = float), np.column_stack([d.to_numpy(), depth['shortcut'].to_numpy(dtype = float), size]))
    if fit is None:
        return {'error': 'no finite estimate'}

    return {name: {'coefficient': float(b), 'p': float(p)} for name, b, p in zip(('intercept', 'd', 'shortcut', 'log_box_fraction'), *fit)}

def h1_hits(summary, items, judge):
    """CV-Bench-3D correctness of a trained map's checkpoint: the saved hit column, else that checkpoint's evaluation."""
    if 'hit' in items and items['hit'].notna().all():
        return items.set_index('index')['hit'], 'hit column of items.csv'
    checkpoint = summary.get('checkpoint')
    if summary.get('map') != 'trained' or not checkpoint:
        return None, 'not a trained checkpoint'
    parts = os.path.normpath(os.path.relpath(os.path.dirname(checkpoint))).split(os.sep)
    run = os.path.join('evaluation', *parts[1:]) if parts[0] == 'checkpoint' else None
    try:
        if run is None:
            raise ValueError(f'no evaluation run for {checkpoint}')
        return load_hits(run, 'CV-Bench-3D', judge).set_index('index')['hit'], result_file(run, 'CV-Bench-3D', judge)
    except (OSError, ValueError) as error:
        return None, str(error)

def h1_run(summary, items, rule, images, hits):
    calibration = items[(items['category'] == 'Distance') & ~items['overlap']]['d']
    sign, median = h1_sign(calibration, rule)
    depth = items[(items['category'] == 'Depth') & ~items['overlap']].set_index('index')
    if not (depth['n_objects'] == 2).all():
        raise ValueError('the same-box statistic assumes two named objects per Depth question')
    d = depth['d'] * sign
    suffix = '' if sign > 0 else '_neg'

    def words(column):
        return np.array([value for row in depth[column] for value in row])

    owner = np.array([i for row in depth['peak_box'] for i in range(len(row))])
    peak = words('peak_box' + suffix)
    landed = peak >= 0
    result = {
        'seed': summary.get('seed'), 'map': summary.get('map'), 'ablation': summary.get('ablation'),
        'center': summary.get('center', False), 'similarity': summary.get('similarity', 'map'),
        'calibration_median': median, 'calibration_questions': len(calibration), 'sign': sign, 'saved_sign': summary.get('sign'),
        'raw_mean_d': float(depth['d'].mean()), **describe_d(d, images),
        'raw_max_in_box': float((words('peak_box') >= 0).mean()), 'raw_min_in_box': float((words('peak_box_neg') >= 0).mean()),
        'peak_in_box': float(landed.mean()), 'peak_own_box': float((peak[landed] == owner[landed]).mean()) if landed.any() else None,
        'box_area': float(np.mean([sum(fractions) for fractions in depth['box_fraction']])),
        'own_box': float(words('own_top' + suffix).mean()),
        # Both words prefer the same box by their mean values; with two boxes this does not depend on the sign.
        'same_box': float(np.mean([first != second for first, second in depth['own_top']])),
    }
    if hits is not None:
        result['regression'] = h1_regression(depth, d, hits)

    return result, d

def h1_compare(mine, other):
    """Two-sided paired Wilcoxon of signed d; runs pair by seed, and a seedless reference pairs with every run."""
    differences = []
    for seed, d in mine:
        partners = [d_other for seed_other, d_other in other if seed_other is None or seed_other == seed]
        if partners:
            differences.append(d - partners[0])
    if not differences:
        return None
    difference = pd.concat(differences, axis = 1).dropna().mean(axis = 1)

    return {'pairs': len(differences), 'questions': len(difference), 'mean_difference': float(difference.mean()),
            'wilcoxon_two_sided_p': float(stats.wilcoxon(difference).pvalue) if (difference != 0).any() else None}

H1_NOTES = {
    'tie': 'Sign +1 when |Distance median| < 1e-9, else the sign of the median (report Section 1): a transparent numerical '
           'repair and sensitivity analysis, not a preregistered protocol. The primary group\'s pooled test uses it.',
    'original': "h1.py's rule, under which any negative median (seed 42: -3.7e-17) flips the map. Superseded; kept to show "
                'the sensitivity to the tie repair.',
    'wilcoxon_p': "One-sided signed-rank test of d > 0 with SciPy's default zero handling (exact zeros dropped; d is not rounded). "
                  'It is not a test of the mean alone, and its last digits depend on the SciPy version (see the sidecar).',
    'peaks': 'raw_max_in_box and raw_min_in_box: words whose raw extremum lies in either marked box. peak_in_box: the same '
             'for the maximum of the signed map; peak_own_box: the share of those peaks in the word\'s own box.',
    'own_box': "Words whose own box has the higher signed mean value. same_box: questions in which both words prefer the same "
               'box by mean value, which is not a peak statistic; maps whose rows share one spatial preference score high '
               'without word-specific grounding.',
    'per_image': 'Mean signed d per source image, a sensitivity analysis for Depth questions that share an image.',
    'scope': 'The readout is the named-word rows of the interaction matrix on the retained CV-Bench-3D Depth questions. '
             'A null result does not show that the matrix holds no recoverable object information, nor what the experts or '
             'the language model recover. Comparisons with CLIP or with normalization controls do not by themselves '
             'identify why training produced these maps.',
}

def h1(args):
    images = h1_images(args.data)
    groups = {group[0]: [(directory, *h1_load(directory)) for directory in group[1:]] for group in args.group}
    report = {'primary': args.primary, 'notes': H1_NOTES, 'groups': {}, 'comparisons': []}
    signed = {rule: {} for rule in RULES}
    rows = []
    for name, runs in groups.items():
        settings = {(summary.get('map'), summary.get('ablation'), summary.get('center', False), summary.get('similarity', 'map'))
                    for _, summary, _ in runs}
        seeds = [summary.get('seed') for _, summary, _ in runs]
        if len(settings) > 1 or len(set(seeds)) != len(seeds):
            raise ValueError(f'{name}: a group must hold one map setting with one run per seed')
        entry = {'role': 'primary' if name == args.primary else 'secondary, not corrected for multiple comparisons',
                 'runs': {}, 'pooled': {}}
        for directory, summary, items in runs:
            hits, hit_source = h1_hits(summary, items, args.judge)
            per_rule = {}
            for rule in RULES:
                per_rule[rule], d = h1_run(summary, items, rule, images, hits if rule == 'tie' else None)
                signed[rule].setdefault(name, []).append((summary.get('seed'), d))
            per_rule['tie']['hits'] = hit_source
            entry['runs'][directory] = per_rule
            regression = per_rule['tie'].get('regression') or {}
            rows.append({'group': name, 'run': directory, **{key: value for key, value in per_rule['tie'].items() if not isinstance(value, dict)},
                         'regression_d_p': regression.get('d', {}).get('p'), 'regression_shortcut_p': regression.get('shortcut', {}).get('p'),
                         'original_sign': per_rule['original']['sign'], 'original_mean_d': per_rule['original']['mean_d'],
                         'original_wilcoxon_p': per_rule['original']['wilcoxon_p']})
        for rule in RULES:
            d = pd.concat([series for _, series in signed[rule][name]], axis = 1).dropna().mean(axis = 1)
            entry['pooled'][rule] = {'runs': len(runs), **describe_d(d, images)}
        rows.append({'group': name, 'run': f'pooled over {len(runs)}', **entry['pooled']['tie'],
                     'original_mean_d': entry['pooled']['original']['mean_d'], 'original_wilcoxon_p': entry['pooled']['original']['wilcoxon_p']})
        report['groups'][name] = entry
    for first, second in args.compare:
        if first in groups and second in groups:
            report['comparisons'].append({'first': first, 'second': second, 'difference': 'first minus second, signed d',
                                          **{rule: h1_compare(signed[rule][first], signed[rule][second]) for rule in RULES}})
    write(args.output, args.name, report)
    write(args.output, f'{args.name}_table', pd.DataFrame(rows))

## Scores and Checkpoint-Level Intervals

def scores(args):
    _, observed = validated_runs(list(dict.fromkeys(args.runs + args.full)), args.judge, args.naturalbench_metric)
    write(args.output, args.name, pd.DataFrame([{'run': run, **values} for run, values in observed.items()]))
    if len(args.full) > 1:
        frame = pd.DataFrame([observed[run] for run in args.full], index = args.full)
        write(args.output, f'{args.name}_seed_sd', {
            'runs': args.full, 'mean': frame.mean().to_dict(), 'sample_sd': frame.std(ddof = 1).to_dict(),
            'range': (frame.max() - frame.min()).to_dict(),
            'note': 'Observed mean and sample SD over these seeds; Avg is averaged within each seed first, so benchmark-specific '
                    'changes largely cancel. An SD propagated from the benchmark SDs under independence is a different '
                    'quantity and does not replace it. Three seeds give a weak estimate of variability over training runs.'})

@lru_cache(maxsize = None)
def tsv_image_digests(dataset):
    """SHA-256 of each question's image bytes in the benchmark TSV, following VLMEvalKit's references to earlier rows."""
    table = pd.read_csv(track(os.path.join(LMU_ROOT, f'{dataset}.tsv')), sep = '\t', usecols = ['index', 'image'], dtype = {'image': str})
    payloads = dict(zip(table['index'].astype(str), table['image']))
    digests, decoded = {}, {}
    for index, payload in payloads.items():
        payload = resolve_payload(payload, payloads, index)
        if payload not in decoded:
            decoded[payload] = hashlib.sha256(base64.b64decode(payload)).hexdigest()
        digests[int(index)] = decoded[payload]

    return digests

def resolve_payload(payload, payloads, index):
    """A short image value naming another row reuses that row's image."""
    visited = set()
    while len(payload) <= 64 and payload in payloads:
        if payload in visited:
            raise ValueError(f'Cyclic image reference at {index}')
        visited.add(payload)
        payload = payloads[payload]

    return payload

def cluster_labels(table, dataset):
    """One label per metric unit; questions sharing an image are resampled together."""
    if dataset == 'NaturalBenchDataset':
        return list(range(len(table) // 4)), 'four-row group (two images, two questions)'
    if dataset in ('CV-Bench-2D', 'CV-Bench-3D'):
        columns = ['source_dataset', 'source_filename']
        if not set(columns).issubset(table) or table[columns].isna().any().any():
            raise ValueError(f'{dataset}: source image IDs required for clustered resampling')
        return list(zip(table[columns[0]].astype(str), table[columns[1]].astype(str))), 'source image'
    digests = tsv_image_digests(dataset)
    missing = [index for index in table['index'] if int(index) not in digests]
    if missing:
        raise ValueError(f'{dataset}: {len(missing)} scored questions have no image in the TSV')
    unit = 'circular question group, by image bytes' if dataset.startswith('MMBench') else 'image bytes (SHA-256)'

    return [digests[int(index)] for index in table['index']], unit

def cluster_members(labels):
    members = {}
    for position, label in enumerate(labels):
        members.setdefault(label, []).append(position)

    return [np.asarray(positions, dtype = int) for positions in members.values()]

def cluster_draw(groups, rng, stratum = None):
    """Resample whole images; retry the rare draw that omits a CV source/task."""
    expected = len(np.unique(stratum)) if stratum is not None else None
    for _ in range(1000):
        chosen = rng.integers(0, len(groups), len(groups))
        draw = np.concatenate([groups[index] for index in chosen])
        if stratum is None or len(np.unique(stratum[draw])) == expected:
            return draw
    raise ValueError('Could not sample every source/task while preserving image clusters')

def bootstrap(args):
    if args.replicates < 2:
        raise ValueError('Bootstrap needs at least two replicates')
    if not args.runs or args.reference in args.runs:
        raise ValueError('Bootstrap needs runs other than the reference')
    runs, observed = validated_runs([args.reference] + args.runs, args.judge, args.naturalbench_metric)
    arrays = {run: {dataset: units(table, dataset) for dataset, table in tables.items()} for run, tables in runs.items()}
    reference = runs[args.reference]
    groups, sampling = {}, {}
    for dataset in DATASETS:
        labels, unit = cluster_labels(reference[dataset], dataset)
        for run, tables in runs.items():
            if tuple(tables[dataset]['index']) != tuple(reference[dataset]['index']):
                raise ValueError(f'{dataset}: {run} covers different questions')
            # Labels defining the metric must also agree, not just row indices.
            for column in ('answer', 'source', 'category', 'l2_category', 'l2-category'):
                if column in reference[dataset] and column in tables[dataset] and not reference[dataset][column].equals(tables[dataset][column]):
                    raise ValueError(f'{dataset}: {column} differs between {run} and the reference')
        groups[dataset] = cluster_members(labels)
        sampling[dataset] = {'unit': unit, 'clusters': len(groups[dataset]), 'metric_units': len(labels)}

    rng = np.random.default_rng(args.seed)
    metrics = DATASETS + ['Avg', 'Grounded', 'Other']
    values = {run: np.empty((args.replicates, len(metrics))) for run in runs}
    for replicate in range(args.replicates):
        # The same clusters for every run, so each interval is of a paired difference.
        draws = {dataset: cluster_draw(groups[dataset], rng, arrays[args.reference][dataset].get('stratum')) for dataset in DATASETS}
        for run in runs:
            replicate_scores = run_scores(arrays[run], args.naturalbench_metric, draws)
            values[run][replicate] = [replicate_scores[metric] for metric in metrics]

    def interval(samples):
        return [float(value) for value in np.percentile(samples, [2.5, 97.5])]

    report = {'reference': args.reference, 'replicates': args.replicates, 'seed': args.seed, 'naturalbench_metric': args.naturalbench_metric,
              'sampling': sampling, 'reference_scores': observed[args.reference],
              'scope': 'Checkpoint-level comparisons: intervals for the difference between these fitted checkpoints under '
                       'resampling of evaluation units (official metric recomputed in every replicate). They carry no '
                       'training-seed uncertainty; a one-seed arm stays one training run, and an interval that includes zero '
                       'is not evidence of equivalence. Images shared across benchmarks are not modeled.',
              'd_definition': 'D = (reference - run on grounded questions) - (reference - run on other questions), in points; '
                              'positive when the run loses more on grounded questions.',
              'runs': {}}
    grounded, other = metrics.index('Grounded'), metrics.index('Other')
    for run in args.runs:
        difference = values[run] - values[args.reference]
        entry = {'scores': observed[run], 'minus_reference': {
            metric: {'difference': observed[run][metric] - observed[args.reference][metric], 'interval_95': interval(difference[:, column])}
            for column, metric in enumerate(metrics)}}
        entry['D'] = {'value': d_score(observed[run], observed[args.reference]),
                      'interval_95': interval(difference[:, other] - difference[:, grounded])}
        report['runs'][run] = entry
    write(args.output, args.name, report)

## Thesis Tables

CATEGORY_LEVELS = {'CV-Bench-2D': ('category',), 'CV-Bench-3D': ('category',), 'MMBench_DEV_EN_V11': ('l2-category', 'category'),
                   'MMStar': ('category', 'l2_category'), 'SEEDBench_IMG': ('category',)}

def with_group_rows(frame, groups, keys, columns):
    """Mean and sample SD over each group's runs, appended as rows named 'GROUP mean' and 'GROUP sd'."""
    blocks = [frame]
    for name, members in groups.items():
        subset = frame[frame['run'].isin(members)]
        if subset['run'].nunique() < 2:
            continue
        for statistic in ('mean', 'std'):
            if keys:
                block = subset.groupby(keys, sort = False)[columns].agg(statistic).reset_index()
            else:
                block = subset[columns].agg(statistic).to_frame().T
            block.insert(0, 'run', f'{name} {"sd" if statistic == "std" else statistic}')
            blocks.append(block)

    return pd.concat(blocks, ignore_index = True)

def category_table(loaded):
    rows = []
    for run, tables in loaded.items():
        for dataset, levels in CATEGORY_LEVELS.items():
            for level in levels:
                for category, hits in tables[dataset].groupby(level, sort = True)['hit']:
                    rows.append({'run': run, 'dataset': dataset, 'level': level, 'category': category, 'n': len(hits),
                                 'accuracy': 100 * hits.mean()})

    return pd.DataFrame(rows)

def grounded_table(loaded, reference, groups):
    """Tables 4.5 and 4.11: grounded and other accuracy, losses relative to the reference and D."""
    def split(tables, datasets):
        arrays = {dataset: units(tables[dataset], dataset) for dataset in datasets}
        totals = np.zeros(4)
        for values in arrays.values():
            hit, grounded = values['hit'], values['grounded']
            totals += (hit[grounded].sum(), grounded.sum(), hit[~grounded].sum(), (~grounded).sum())
        return {'Grounded': 100 * totals[0] / totals[1], 'Other': 100 * totals[2] / totals[3],
                'grounded_n': int(totals[1]), 'other_n': int(totals[3])}

    within = ('MMBench_DEV_EN_V11', 'MMStar', 'SEEDBench_IMG')
    splits = {run: split(tables, list(GROUNDED)) for run, tables in loaded.items()}
    parts = {run: {dataset: split(tables, [dataset]) for dataset in within} for run, tables in loaded.items()}
    group_means = {name: {key: np.mean([splits[run][key] for run in members]) for key in ('Grounded', 'Other')}
                   for name, members in groups.items() if len(members) > 1}
    rows = []
    for run, values in splits.items():
        row = {'run': run, **values, 'loss_grounded': splits[reference]['Grounded'] - values['Grounded'],
               'loss_other': splits[reference]['Other'] - values['Other'], 'D': d_score(values, splits[reference]),
               # Thesis Section 4.3.4 also reports D within each of the three benchmarks, averaged.
               'D_within_benchmarks': np.mean([d_score(parts[run][dataset], parts[reference][dataset]) for dataset in within])}
        for name, mean in group_means.items():
            row[f'D_vs_{name}_mean'] = d_score(values, mean)
        rows.append(row)

    return pd.DataFrame(rows)

def depth_cue_table(loaded, cue_path, groups):
    """Table 4.6: CV-Bench-3D accuracy by whether the 2D cue of h1.shortcut gives the right answer."""
    cue = pd.read_csv(track(cue_path)).set_index('index')[['category', 'shortcut']]
    rows = []
    for run, tables in loaded.items():
        joined = tables['CV-Bench-3D'].set_index('index')[['hit']].join(cue, how = 'inner')
        for (category, right), part in joined.groupby(['category', 'shortcut']):
            correct, n = int(part['hit'].sum()), len(part)
            rows.append({'run': run, 'category': category, 'cue_right': bool(right), 'n': n, 'correct': correct,
                         'accuracy': 100 * correct / n, 'binomial_p_two_sided': stats.binomtest(correct, n, 0.5).pvalue})

    return with_group_rows(pd.DataFrame(rows), groups, ['category', 'cue_right'], ['n', 'accuracy'])

def section_422(loaded, groups, reference_group):
    """The per-question statistics of thesis Section 4.2.2 that do not come from H1."""
    choices = {run: {dataset: chosen_options(tables[dataset]) for dataset in GROUNDED} for run, tables in loaded.items()}
    report = {'agreement': {}, 'cv_bench_3d_letters': {}, 'counting_near_misses': {}, 'cv_bench_3d_against_seeds': {}}
    for name, members in groups.items():
        if len(members) < 2:
            continue
        hits = pd.concat({run: pd.concat({dataset: loaded[run][dataset].set_index('index')['hit'] for dataset in GROUNDED}) for run in members}, axis = 1)
        chosen = pd.concat({run: pd.concat(choices[run]) for run in members}, axis = 1)
        wrong = (hits == 0).all(axis = 1)
        readable = chosen[wrong].notna().all(axis = 1)
        same = chosen[wrong][readable].nunique(axis = 1) == 1
        report['agreement'][name] = {'questions': len(hits), 'all_correct': float((hits == 1).all(axis = 1).mean()),
                                     'all_wrong': float(wrong.mean()), 'disagree': float(1 - (hits == 1).all(axis = 1).mean() - wrong.mean()),
                                     'all_wrong_same_option': float(same.mean()) if len(same) else None,
                                     'all_wrong_readable': int(readable.sum()), 'all_wrong_total': int(wrong.sum())}
    for run, tables in loaded.items():
        table = tables['CV-Bench-3D']
        letters = choices[run]['CV-Bench-3D']
        # An answer such as "A car." still leads with its letter; "The lamp is closer to the camera." does not.
        lettered = table['prediction'].astype(str).str.strip().str.match(r'\(?[A-H]\)?(?:[.:,)\s]|$)')
        report['cv_bench_3d_letters'][run] = {'chosen': {str(key): float(value) for key, value in letters.fillna('unreadable').value_counts(normalize = True).items()},
                                              'answer_A': float((table['answer'] == 'A').mean()), 'not_led_by_a_letter': float(1 - lettered.mean())}
        count = tables['CV-Bench-2D'].set_index('index')
        count = count[(count['category'] == 'Count') & (count['hit'] == 0)]
        picked = choices[run]['CV-Bench-2D'].reindex(count.index)
        distances = []
        for index, letter in picked.dropna().items():
            try:
                distances.append(abs(float(count.loc[index, letter]) - float(count.loc[index, count.loc[index, 'answer']])))
            except (KeyError, TypeError, ValueError):
                continue
        report['counting_near_misses'][run] = {'wrong': len(count), 'readable': len(distances),
                                               'off_by_one': float(np.mean(np.isclose(distances, 1))) if distances else None}
    # R7: all_map_mean against the best full seed on CV-Bench-3D, beside the full seeds' own range.
    members = groups.get(reference_group, [])
    if len(members) > 1:
        full = [headline(units(loaded[run]['CV-Bench-3D'], 'CV-Bench-3D'), 'CV-Bench-3D', 'Acc') for run in members]
        report['cv_bench_3d_against_seeds'] = {
            'full_range': float(max(full) - min(full)), 'full_best': float(max(full)),
            'runs': {run: float(headline(units(tables['CV-Bench-3D'], 'CV-Bench-3D'), 'CV-Bench-3D', 'Acc') - max(full))
                     for run, tables in loaded.items() if run not in members},
            'note': 'Descriptive only: differences from the best of three seeds carry no inferential weight.'}
    report['notes'] = ['Chosen options use VLMEvalKit\'s rule-based can_infer; rows it cannot read (judge-only) are excluded '
                       'from same-option and near-miss shares and reported as unreadable.',
                       'Binomial tests in the depth-cue table treat one checkpoint\'s questions as independent and ignore '
                       'shared images; no test is given for averages over seeds.']

    return report

def tables(args):
    groups = {group[0]: group[1:] for group in args.group}
    runs = list(dict.fromkeys([args.reference, *args.runs, *(run for members in groups.values() for run in members)]))
    loaded = {run: {dataset: validated(run, dataset, args.judge, args.naturalbench_metric) for dataset in GROUNDED} for run in runs}
    write(args.output, f'{args.name}_categories', with_group_rows(category_table(loaded), groups, ['dataset', 'level', 'category'], ['n', 'accuracy']))
    grounded = grounded_table(loaded, args.reference, groups)
    write(args.output, f'{args.name}_grounded', with_group_rows(grounded, groups, [], [column for column in grounded if column != 'run']))
    write(args.output, f'{args.name}_depth_cue', depth_cue_table(loaded, args.cue, groups))
    write(args.output, f'{args.name}_section_4_2_2', section_422(loaded, groups, args.reference_group))

## External Baselines

# Published scores this repository's own sources record; every other value must be checked against its primary paper.
PUBLISHED = {
    'LLaVA-NeXT-8B': {'CV-Bench-2D': 62.2, 'CV-Bench-3D': 65.3, 'source': 'Cambrian-1, as recorded in vlmeval_baselines.py'},
    'LLaVA-NeXT-8B-Cambrian': {'CV-Bench-2D': 62.2, 'CV-Bench-3D': 65.3, 'source': 'Cambrian-1 (same model, Cambrian prompt)'},
}

def baseline_registry():
    try:
        from vlmeval_baselines import BASELINES
    except Exception as error:
        return {'error': f'{type(error).__name__}: {error}'}

    def plain(value):
        return value if isinstance(value, (int, float, str, bool, type(None))) else repr(value)

    return {name: {'wrapper': getattr(factory.func, '__name__', repr(factory.func)),
                   'settings': {key: plain(value) for key, value in factory.keywords.items()}} for name, factory in BASELINES.items()}

def judge_share(run, dataset, judge):
    """Share of raw predictions VLMEvalKit's rule-based matcher cannot read, i.e. the rows the judge decides."""
    raw = predictions_file(run, dataset, judge)
    chosen = chosen_options(pd.read_excel(track(raw)))

    return float(chosen.isna().mean())

def baselines(args):
    registry = baseline_registry()
    rows, protocols = [], {}
    for run in args.runs:
        model = model_name(run)
        unreported = run in args.unreported or 'Not used' in os.path.normpath(run).split(os.sep)
        statuses = {}
        for dataset in DATASETS:
            row = {'model': model, 'run': run, 'reported_in_thesis': not unreported, 'dataset': dataset}
            try:
                table = validated(run, dataset, args.judge, args.naturalbench_metric)
            except ValueError as error:
                rows.append({**row, 'score': None, 'status': 'not evaluated' if 'no scored file' in str(error) else str(error)})
                continue
            directory = os.path.dirname(result_file(run, dataset, args.judge))
            status = dated_status(directory)
            statuses[dataset] = {'dated': os.path.basename(directory), 'commit': status.get('commit'), 'argv': status.get('argv')}
            published = PUBLISHED.get(model, {}).get(dataset)
            score = float(headline(units(table, dataset), dataset, args.naturalbench_metric))
            rows.append({**row, 'score': score, 'status': 'evaluated',
                         'judge': status.get('datasets', {}).get(dataset, {}).get('judge_model'),
                         'judge_decided_share': judge_share(run, dataset, args.judge) if dataset not in ('NaturalBenchDataset', 'POPE') else None,
                         'published': published, 'measured_minus_published': None if published is None else score - published})
        protocols[model] = {'run': run, 'reported_in_thesis': not unreported,
                            'registry': registry.get(model, 'not in vlmeval_baselines.BASELINES') if 'error' not in registry else registry,
                            'results': statuses, 'published_source': PUBLISHED.get(model, {}).get('source')}
    table = pd.DataFrame(rows)

    # Each IP-MoE group's mean over its seeds, to check claims of the highest score against every evaluated model.
    ranking = {}
    means = {}
    for group in args.ipmoe:
        _, observed = validated_runs(group[1:], args.judge, args.naturalbench_metric)
        means[group[0]] = {dataset: float(np.mean([observed[run][dataset] for run in group[1:]])) for dataset in DATASETS}
    for dataset in DATASETS:
        entries = [{'model': model, 'score': means[model][dataset], 'reported_in_thesis': True} for model in means]
        entries += [{'model': row.model, 'score': row.score, 'reported_in_thesis': row.reported_in_thesis}
                    for row in table[(table['dataset'] == dataset) & table['score'].notna()].itertuples()]
        entries.sort(key = lambda entry: -entry['score'])
        reported_only = [entry for entry in entries if entry['reported_in_thesis']]
        ranking[dataset] = {'highest_evaluated': entries[0]['model'] if entries else None,
                            'highest_reported': reported_only[0]['model'] if reported_only else None, 'order': entries}
    write(args.output, 'baselines', table)
    write(args.output, 'baselines_protocol', {
        'protocols': protocols, 'ranking': ranking,
        'notes': ['reported_in_thesis is False for runs under evaluation/Not used; their results exist and either belong in '
                  'the comparison or need a stated reason for exclusion.',
                  'judge_decided_share: raw predictions that rule-based matching cannot read, which the judge decides.',
                  'Published values other than those listed must be verified from the primary papers; a gap between '
                  'published and measured scores stays unexplained unless a protocol difference is demonstrated.']})

## Benchmark Images in the Training Data

def phash_bytes(data):
    """64-bit DCT perceptual hash as in imagehash.phash; None for unreadable images."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data) if isinstance(data, bytes) else data) as image:
            image.draft('L', (128, 128))
            pixels = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype = float)
    except Exception:
        return None
    low = fft.dct(fft.dct(pixels, axis = 0, norm = None), axis = 1, norm = None)[:8, :8]

    return int(''.join('1' if bit else '0' for bit in (low > np.median(low)).ravel()), 2)

def phash_path(path):
    return path, phash_bytes(path)

POPCOUNT = np.array([bin(value).count('1') for value in range(256)], dtype = np.uint8)
PHASH_INDEX = {}

def phash_index(packed, threshold):
    PHASH_INDEX.update(packed = packed, threshold = threshold)

def phash_match(data):
    """A benchmark image's hash, its nearest training distance and the training images within the threshold."""
    value = phash_bytes(data)
    if value is None:
        return None, None, []
    distance = POPCOUNT[(PHASH_INDEX['packed'] ^ np.uint64(value)).view(np.uint8).reshape(-1, 8)].sum(axis = 1)

    return value, int(distance.min()), [(int(position), int(distance[position])) for position in np.flatnonzero(distance <= PHASH_INDEX['threshold'])]

def benchmark_images(dataset):
    """Each distinct image of a benchmark TSV once, as {SHA-256: (first index, bytes)}."""
    table = pd.read_csv(track(os.path.join(LMU_ROOT, f'{dataset}.tsv')), sep = '\t', usecols = ['index', 'image'], dtype = {'image': str})
    payloads = dict(zip(table['index'].astype(str), table['image']))
    images, decoded = {}, set()
    for index, payload in payloads.items():
        payload = resolve_payload(payload, payloads, index)
        if payload not in decoded:
            decoded.add(payload)
            data = base64.b64decode(payload, validate = True)
            images.setdefault(hashlib.sha256(data).hexdigest(), (index, data))

    return images

def _atomic_json(path, payload):
    """Commit a hash cache only after it is completely written."""
    import tempfile

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok = True)
    descriptor, temporary = tempfile.mkstemp(prefix = '.rebuttal-', suffix = '.json', dir = parent)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(payload, handle)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def _file_stamp(path):
    stat = os.stat(path)
    return [os.path.realpath(path), stat.st_size, stat.st_mtime_ns]

def _inventory(paths):
    """Metadata scan avoids decoding unchanged training images on subsequent audits."""
    return {os.path.abspath(path): _file_stamp(path)[1:] for path in sorted(paths)}

def cached_phashes(inventory, cache_path, workers, force = False):
    """Reuse perceptual hashes only when the image's path, size and mtime match."""
    previous = {}
    if not force:
        try:
            with open(cache_path) as handle:
                cache = json.load(handle)
            if isinstance(cache, dict) and cache.get('version') == 1 and isinstance(cache.get('images'), dict):
                previous = cache['images']
        except (OSError, ValueError, KeyError, TypeError):
            pass
    records_, pending = {}, []
    for path, stamp in inventory.items():
        record = previous.get(path)
        if (isinstance(record, dict) and record.get('stamp') == stamp and 'hash' in record
                and (record['hash'] is None or (isinstance(record['hash'], int) and 0 <= record['hash'] < 2 ** 64))):
            records_[path] = record
        else:
            pending.append(path)
    print(f'Training pHash cache: {len(records_)} reused, {len(pending)} images to hash')
    if pending:
        with ProcessPoolExecutor(workers) as pool:
            for path, value in pool.map(phash_path, pending, chunksize = 512):
                if _file_stamp(path)[1:] != inventory[path]:
                    raise RuntimeError(f'Training image changed while hashing: {path}')
                records_[path] = {'stamp': inventory[path], 'hash': value}
    _atomic_json(cache_path, {'version': 1, 'images': records_})

    return [(path, records_[path]['hash']) for path in inventory if records_[path]['hash'] is not None]

def overlap(args):
    train = sorted(path for path in glob.glob(os.path.join(args.train_images, '*', '*')) if os.path.isfile(path))
    if not train:
        raise ValueError(f'No training images under {args.train_images}')
    # Training images are stored under their SHA-256, so an exact match needs no decoding.
    train_digests = {os.path.splitext(os.path.basename(path))[0]: path for path in train}
    paths, packed = [], None
    if not args.skip_phash:
        hashed = cached_phashes(_inventory(train), args.hash_cache or os.path.join(args.output, 'train_phashes.json'), args.workers, args.force)
        paths = [path for path, _ in hashed]
        packed = np.array([value for _, value in hashed], dtype = np.uint64)
    report = {'train_images': len(train_digests), 'phash_threshold': None if args.skip_phash else args.threshold,
              'scope': 'Image overlap with the cached training images, not benchmark question or answer leakage. Exact '
                       'matches are byte-identical files; perceptual matches are candidates that need visual checks.',
              'datasets': {}}
    matches = []
    for dataset in args.datasets:
        images = benchmark_images(dataset)
        exact = [(digest, index) for digest, (index, _) in images.items() if digest in train_digests]
        matches += [{'dataset': dataset, 'benchmark_index': index, 'sha256': digest, 'train_path': train_digests[digest],
                     'kind': 'sha256', 'hamming': 0} for digest, index in exact]
        entry = {'images': len(images), 'sha256_matches': len(exact), 'sha256_share': len(exact) / len(images)}
        if packed is not None:
            digests = list(images)
            with ProcessPoolExecutor(args.workers, initializer = phash_index, initargs = (packed, args.threshold)) as pool:
                found = list(pool.map(phash_match, [images[digest][1] for digest in digests], chunksize = 8))
            matched = set()
            for digest, (_, _, near) in zip(digests, found):
                for position, distance in near:
                    matched.add(digest)
                    matches.append({'dataset': dataset, 'benchmark_index': images[digest][0], 'sha256': digest,
                                    'train_path': paths[position], 'kind': 'phash', 'hamming': distance})
            nearest = [value for _, value, _ in found if value is not None]
            entry.update(phash_matched_images = len(matched), phash_matched_share = len(matched) / len(images),
                         unreadable = sum(value is None for value, _, _ in found),
                         nearest_hamming_histogram = {int(k): int(v) for k, v in zip(*np.unique(nearest, return_counts = True))})
        report['datasets'][dataset] = entry
        print(f'{dataset}: {entry}')
        del images
    write(args.output, 'overlap_matches', pd.DataFrame(matches, columns = ['dataset', 'benchmark_index', 'sha256', 'train_path', 'kind', 'hamming']))
    write(args.output, 'overlap', report)

## Initial Interaction MLPs

def init(args):
    import torch

    from main import reproduce
    from model.encoder import encoding

    full = torch.load(args.checkpoint, map_location = 'cpu', weights_only = True)
    saved = Namespace(**full['args'])
    seed = saved.seed if args.seed is None else args.seed

    # main.run_training seeds, builds the dataset (which uses its own generators) and then calls encoding first,
    # so the MLPs initialize from the seeded global stream on the CPU before moving to the GPU.
    reproduce(seed)
    _, text_encoder, vision_encoder = encoding(saved.drop, 'cpu', saved.embed_dim)
    model = {f'{prefix}.{name}': parameter.detach().clone() for prefix, module in (('text_encoder', text_encoder), ('vision_encoder', vision_encoder))
             for name, parameter in module.named_parameters() if parameter.requires_grad}

    # If this reproduces the training start, the trained weights stay closer to it than to another seed's start.
    check = {}
    for name in ('vision_encoder.mlp.1.weight', 'text_encoder.mlp.0.weight'):
        trained = full['model'][name].float().ravel()
        check[name] = float(torch.nn.functional.cosine_similarity(model[name].float().ravel(), trained, dim = 0))
    print(f'Cosine similarity of seed {seed} initial and trained ({args.checkpoint}) weights: {check}')
    if min(check.values()) < 0.5:
        print('WARNING: the trained weights are not close to this start; it may not be the one training used')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok = True)
    torch.save({'model': model, 'args': {**full['args'], 'seed': seed}, 'initial': True, 'trained_from': args.checkpoint,
                'cosine_to_trained': check}, args.output)

## Router Weights

def router(args):
    frames = [pd.read_csv(track(path)).assign(run = path) for path in args.csv]
    weights = pd.concat(frames)
    per_seed = weights.groupby(['run', 'dataset'])[['self', 'cross', 'hybrid']].agg(['mean', 'std'])
    pooled = weights.groupby('dataset')[['self', 'cross', 'hybrid']].agg(['mean', 'std', 'min', 'max'])
    print(per_seed.round(4).to_string())
    print(pooled.round(4).to_string())
    os.makedirs(args.output, exist_ok = True)
    per_seed.to_csv(os.path.join(args.output, 'router_per_seed.csv'))
    pooled.to_csv(os.path.join(args.output, 'router_pooled.csv'))

def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default = 'evaluation/rebuttal')
    commands = parser.add_subparsers(dest = 'command', required = True)

    command = commands.add_parser('provenance')
    command.add_argument('--fork', default = 'VLMEvalKit')
    command.add_argument('--package', help = 'An archive whose SHA-256 is recorded, e.g. the package sent for review')
    command.add_argument('--reviewed-sha256', help = 'The SHA-256 the reviewers recorded for their archive')

    command = commands.add_parser('counts')
    command.add_argument('--checkpoint-root', default = 'checkpoint')
    command.add_argument('--prompts', help = 'An IP-MoE evaluation run whose predictions give the prompts for prefix-token counts')
    command.add_argument('--judge', default = 'gpt-6-luna')

    command = commands.add_parser('records')
    command.add_argument('--runs', nargs = '+', required = True)
    command.add_argument('--reference', help = 'Run whose commits the others are compared with')
    command.add_argument('--judge', default = 'gpt-6-luna')

    command = commands.add_parser('h1')
    command.add_argument('--group', nargs = '+', action = 'append', required = True, metavar = ('NAME', 'RUN'),
                         help = 'A name followed by per-seed H1 output directories of one map setting; repeat per group')
    command.add_argument('--compare', nargs = 2, action = 'append', default = [], metavar = ('FIRST', 'SECOND'),
                         help = 'Paired comparison of two groups, first minus second; repeat per pair')
    command.add_argument('--primary', default = 'full', help = 'The group whose pooled tie-rule test answers RQ1')
    command.add_argument('--data', default = os.path.join(LMU_ROOT, 'CV-Bench-3D.tsv'))
    command.add_argument('--judge', default = 'gpt-6-luna')
    command.add_argument('--name', default = 'h1')

    for name in ('scores', 'bootstrap', 'tables', 'baselines'):
        command = commands.add_parser(name)
        command.add_argument('--judge', default = 'gpt-6-luna')
        command.add_argument('--naturalbench-metric', choices = ['G_Acc', 'Acc', 'Q_Acc', 'I_Acc'], default = 'Acc',
                             help = "NaturalBench's score in Avg; the thesis uses Acc")
        command.add_argument('--name', default = name)
        if name == 'scores':
            command.add_argument('--runs', nargs = '*', default = [])
            command.add_argument('--full', nargs = '+', default = [], help = 'Full-model seeds, also reported as their observed mean and SD')
        elif name == 'bootstrap':
            command.add_argument('--reference', required = True, help = 'The checkpoint every run is compared with')
            command.add_argument('--runs', nargs = '+', required = True)
            command.add_argument('--replicates', type = int, default = 2000)
            command.add_argument('--seed', type = int, default = 0)
        elif name == 'tables':
            command.add_argument('--reference', required = True, help = 'The checkpoint D is measured against (full seed 42)')
            command.add_argument('--runs', nargs = '*', default = [])
            command.add_argument('--group', nargs = '+', action = 'append', default = [], metavar = ('NAME', 'RUN'),
                                 help = 'Seeds of one model whose mean and SD are reported; repeat per model')
            command.add_argument('--reference-group', default = 'full')
            command.add_argument('--cue', default = 'evaluation/h1/clip/items.csv', help = "H1 items.csv whose shortcut column is the 2D cue")
        else:
            command.add_argument('--runs', nargs = '+', required = True)
            command.add_argument('--ipmoe', nargs = '+', action = 'append', default = [], metavar = ('NAME', 'RUN'),
                                 help = 'IP-MoE seeds ranked by their mean; repeat per model')
            command.add_argument('--unreported', nargs = '*', default = [],
                                 help = 'Runs the thesis does not report besides those under evaluation/Not used')

    command = commands.add_parser('overlap')
    command.add_argument('--datasets', nargs = '+', default = DATASETS)
    command.add_argument('--train-images', default = TRAIN_IMAGES)
    command.add_argument('--threshold', type = int, default = 6, help = 'Largest Hamming distance for a candidate perceptual match')
    command.add_argument('--workers', type = int, default = min(32, os.cpu_count() or 1))
    command.add_argument('--skip-phash', action = 'store_true')
    command.add_argument('--force', action = 'store_true', help = 'Recompute the cached training-image hashes')
    command.add_argument('--hash-cache', help = 'Per-image perceptual-hash cache (default: OUTPUT/train_phashes.json)')

    command = commands.add_parser('init')
    command.add_argument('--checkpoint', required = True, help = "A full checkpoint whose args (and seed) the initial MLPs use")
    command.add_argument('--seed', type = int, default = None)
    command.add_argument('--output', dest = 'init_output', required = True)

    command = commands.add_parser('router')
    command.add_argument('--csv', nargs = '+', required = True)

    args = parser.parse_args()
    if args.command == 'init':
        args.output = args.init_output
    {'provenance': provenance, 'counts': counts, 'records': records, 'h1': h1, 'scores': scores, 'bootstrap': bootstrap,
     'tables': tables, 'baselines': baselines, 'overlap': overlap, 'init': init, 'router': router}[args.command](args)

if __name__ == '__main__':
    main()
