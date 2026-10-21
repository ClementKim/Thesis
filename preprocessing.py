"""First-turn LLaVA-NeXT loader for the existing single-question model.

Policy: use ONLY the original first user/assistant pair. Invalid first pairs
reject the row; later pairs are never substituted. Keep the v9 batch contract.
The legacy 8,000-character answer filter remains explicit and configurable;
it is neither a repetition detector nor a model-token/memory guarantee.

The full processed JSON cache is independent of training caps and seeds.
Validation is grouped by identical image bytes, with its own fixed seed (42).
max_train_samples now caps TRAINING ONLY, after the validation split.
This is a first-turn diagnostic baseline, not full-conversation training.
"""

import functools
import glob
import hashlib
import io
import json
import os
import random
import re
import tempfile
from array import array
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager

import pyarrow.parquet as pq
import torch.utils.data as data
from PIL import Image
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Change this when the preprocessing rules change to rebuild the caches.
PREPROCESSING_PROTOCOL = 'ipmoe-llava-next-first-turn-v10'
CACHE_DIRECTORY = 'dataset/cache'
IMAGE_DIRECTORY = 'dataset/llava-next/images'

# Retained from v9 so this revision does not also remove an existing filter.
# Set to None only after checking token budgets and loss memory in the model.
MAX_ANSWER_CHARACTERS = 8000

# Target 10%, subject to a HARD sample cap and keeping whole image groups.
MAX_VALIDATION_SAMPLES = 2000
VALIDATION_FRACTION = 0.1
VALIDATION_SEED = 42
RESERVED_MARKER = '<|ipmoe_mm|>'

# Pin the dataset version so repeated builds use the same source data.
SOURCES = {
    'llava_next': {
        'repo': 'lmms-lab/LLaVA-NeXT-Data',
        'revision': 'dff8edfca0f16f9b4f1407331f6659aa881fc016',
        'patterns': ('data/train-*.parquet',),
        'expected_shards': 250,
        'expected_rows': 779289,
    },
}

# Identify image formats from their first bytes.
_IMAGE_SIGNATURES = (
    (b'\x89PNG\r\n\x1a\n', '.png'),
    (b'\xff\xd8\xff', '.jpg'),
    (b'GIF8', '.gif'),
    (b'BM', '.bmp'),
)

_PREPARED_DIRECTORIES = set()

def save_as_json(value, path):
    """Write to a temporary file first so readers only see complete JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok = True)

    descriptor, temporary = tempfile.mkstemp(prefix = '.json-', suffix = '.tmp', dir = directory or '.')
    try:
        with os.fdopen(descriptor, 'w', encoding = 'utf-8', buffering = 8 * 1024 * 1024) as handle:
            json.dump(value, handle, ensure_ascii = False, separators = (',', ':'))

        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def load_json(path):
    with open(path, encoding = 'utf-8') as handle:
        return json.load(handle)

def parquet_files(name):
    """Download any missing parquet files and return their sorted local paths."""
    source = SOURCES[name]
    repo = source['repo']
    revision = source['revision']
    patterns = list(source['patterns'])

    try:
        local_directory = snapshot_download(repo, repo_type = 'dataset', revision = revision, allow_patterns = patterns)
    except Exception:
        # Use an existing local snapshot if the Hub is unavailable.
        local_directory = snapshot_download(repo, repo_type = 'dataset', revision = revision,
                                            allow_patterns = patterns, local_files_only = True)

    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(local_directory, pattern)))
    files.sort()

    if not files:
        raise RuntimeError(f'No parquet shards for {repo!r} under {local_directory!r}')

    # A local fallback can be incomplete, so compare it with the pinned revision.
    expected_shards = source['expected_shards']
    if len(files) != expected_shards:
        raise RuntimeError(f'Expected {expected_shards} parquet shards for {repo!r}, found {len(files)} under {local_directory!r}')

    expected_rows = source['expected_rows']
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if rows != expected_rows:
        raise RuntimeError(f'Expected {expected_rows:,} rows for {repo!r}, found {rows:,}')

    return files

def iter_rows(name, columns, *, desc = None, batch_size = 64):
    """Read small batches of parquet rows without decoding their images."""
    files = parquet_files(name)

    # Read only the metadata to get the progress bar's total.
    total = None
    if desc:
        total = 0
        for path in files:
            total += pq.ParquetFile(path).metadata.num_rows

    with tqdm(total = total, desc = desc, disable = desc is None) as progress:
        for path in files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size = batch_size, columns = list(columns)):
                rows = batch.to_pylist()
                progress.update(len(rows))

                yield from rows

def encoded_image_bytes(image_ref):
    """Read image bytes from raw data, a {'bytes', 'path'} record, or a list."""
    if isinstance(image_ref, (list, tuple)):
        if not image_ref:
            return None
        image_ref = image_ref[0]

    if image_ref is None:
        return None

    if isinstance(image_ref, (bytes, bytearray, memoryview)):
        raw = bytes(image_ref)
        return raw or None

    if isinstance(image_ref, Mapping):
        raw = image_ref.get('bytes')
        if raw:
            return bytes(raw)

        # Some rows store a file path instead of image bytes.
        path = image_ref.get('path')
        if path and os.path.exists(path):
            with open(path, 'rb') as handle:
                return handle.read() or None

        return None

    raise TypeError(f'Unsupported image reference type: {type(image_ref)!r}')

def _extension(raw):
    """Choose the file extension from the image's contents."""
    for signature, suffix in _IMAGE_SIGNATURES:
        if raw.startswith(signature):
            return suffix

    # WebP has a second marker after the four-byte length field.
    if raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
        return '.webp'

    # PIL can identify other formats when it opens the file.
    return '.img'

def _decodes(raw):
    """Decode the whole image once. PIL's verify() misses truncated JPEGs."""
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
        return True

    except Exception:
        return False

def _prepare_directory(directory):
    """Spread images across 256 folders to avoid one very large directory."""
    if directory in _PREPARED_DIRECTORIES:
        return

    for prefix in range(256):
        os.makedirs(os.path.join(directory, f'{prefix:02x}'), exist_ok = True)

    _PREPARED_DIRECTORIES.add(directory)

def image_path(raw, digest, directory):
    return os.path.join(directory, digest[:2], f'{digest}{_extension(raw)}')

def persist_image(raw, digest, directory, written):
    """Save the original bytes once per image. Identical bytes share a path."""
    _prepare_directory(directory)
    path = image_path(raw, digest, directory)

    # Track saved hashes to avoid checking the same file repeatedly.
    if digest in written:
        return path

    # Verify existing bytes during a build, and repair a damaged saved image.
    try:
        matches = os.path.getsize(path) == len(raw) and _file_sha256(path) == digest
    except OSError:
        matches = False

    if not matches:
        descriptor, temporary = tempfile.mkstemp(prefix = '.image-', dir = os.path.dirname(path))
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(raw)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    written.add(digest)

    return path

def _first_user_assistant_pair(conversations):
    """Read positions 0 and 1 only; never salvage a later, history-dependent turn."""
    if not isinstance(conversations, (list, tuple)) or len(conversations) < 2:
        return None

    values = []
    for turn, allowed_roles in zip(conversations[:2], ({'human', 'user'}, {'gpt', 'assistant'})):
        if not isinstance(turn, Mapping):
            return None
        role = turn.get('from', turn.get('role', ''))
        value = turn.get('value', turn.get('content', ''))
        if not isinstance(role, str) or role.strip().lower() not in allowed_roles:
            return None
        if not isinstance(value, str) or not value.strip():
            return None
        values.append(value.strip())

    return tuple(values)

def _clean_question(question):
    # The model inserts image features itself, so remove the old marker.
    question = re.sub(r'<\s*image\s*>', ' ', question, flags = re.IGNORECASE)
    question = question.replace('\r\n', '\n').replace('\r', '\n')

    return re.sub(r'[^\S\n]+', ' ', question).strip()

def _log_filter_counts(counts, *, cached = False):
    values = [f'{key}={value:,}' for key, value in sorted(counts.items()) if key != 'by_source']
    print('[llava-next] ' + ('cached ' if cached else '') + ', '.join(values))

def llava_next_instruct():
    """Build at most one sample per row from its ORIGINAL first QA pair.

    Counts are in source rows: rows_seen equals accepted plus all rejections.
    Identical examples are deduplicated within each data_source, as in v9.
    """
    directory = os.path.abspath(IMAGE_DIRECTORY)
    columns = ('id', 'image', 'conversations', 'data_source')

    samples = {}
    written = set()
    decoded = {}
    counters = defaultdict(int)
    by_source = defaultdict(lambda: defaultdict(int))

    def count(source_name, event):
        counters[event] += 1
        by_source[source_name][event] += 1

    for row_index, item in enumerate(iter_rows('llava_next', columns, desc = 'llava-next')):
        source_name = str(item.get('data_source') or 'unknown')
        count(source_name, 'rows_seen')
        pair = _first_user_assistant_pair(item.get('conversations'))
        if pair is None:
            count(source_name, 'malformed_first_pair')
            continue

        question, answer = pair
        question = _clean_question(question)
        if not question:
            count(source_name, 'empty_first_question')
            continue
        if MAX_ANSWER_CHARACTERS is not None and len(answer) > MAX_ANSWER_CHARACTERS:
            count(source_name, 'overlong_first_answer')
            continue
        if RESERVED_MARKER in question or RESERVED_MARKER in answer:
            count(source_name, 'reserved_marker')
            continue

        try:
            raw = encoded_image_bytes(item.get('image'))
        except (TypeError, ValueError, OSError):
            count(source_name, 'invalid_image_reference')
            continue
        if raw is None:
            count(source_name, 'no_image')
            continue

        image_digest = hashlib.sha256(raw).hexdigest()
        if image_digest not in decoded:
            decoded[image_digest] = _decodes(raw)
        if not decoded[image_digest]:
            count(source_name, 'undecodable')
            continue

        content = [image_digest, question, answer]
        content_digest = hashlib.sha256(
            json.dumps(content, ensure_ascii = False, separators = (',', ':')).encode('utf-8')
        ).hexdigest()
        sample_id = f'{source_name}/{content_digest}'
        if sample_id in samples:
            count(source_name, 'duplicate_sample')
            continue

        saved_image_path = persist_image(raw, image_digest, directory, written)
        samples[sample_id] = {
            'question': question,
            'answer': answer,
            'img_path': saved_image_path,
            'image_sha256': image_digest,
            'source_name': source_name,
            'source_id': str(item.get('id') or ''),
            'source_row_index': row_index,
            'turn_index': 0,
        }
        count(source_name, 'accepted')

    _log_filter_counts(counters)

    if not samples:
        raise RuntimeError('No LLaVA-NeXT samples survived filtering')

    counts = dict(counters)
    counts['by_source'] = {name: dict(values) for name, values in sorted(by_source.items())}
    return samples, counts

class CustomDataset(data.Dataset):
    """Index cached samples and load each image only when it is requested."""

    def __init__(self, nested):
        super().__init__()

        self.dataset_names = []
        self.sample_ids = []
        self.questions = []
        self.image_paths = []
        self.answers = []

        # Each list uses the same index for a sample.
        for dataset_name, samples in tqdm(nested.items(), desc = 'indexing dataset'):
            for sample_id, sample in samples.items():
                image_path = sample.get('img_path')

                question = sample.get('question')
                answer = sample.get('answer')
                if not image_path or not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f'Invalid cached single-turn sample: {sample_id!r}')

                self.dataset_names.append(str(dataset_name))
                self.sample_ids.append(str(sample_id))
                self.questions.append(question)
                self.image_paths.append(str(image_path))
                self.answers.append(answer)

    def __len__(self):
        return len(self.questions)

    def load_image(self, index):
        """Convert grayscale or transparent images to RGB for CLIP."""
        path = self.image_paths[index]

        try:
            with Image.open(path) as image:
                return image.convert('RGB')

        except Exception as exc:
            raise RuntimeError(f'Failed to decode image at dataset index {index}: {path!r}') from exc

    def __getitem__(self, index):
        dataset_name = self.dataset_names[index]
        question = self.questions[index]
        image = self.load_image(index)
        answer = self.answers[index]

        return dataset_name, question, image, answer

def collate_fn(batch, processor):
    """Collect dataset items into a batch and prepare the CLIP inputs."""
    if not batch:
        raise ValueError('Cannot collate an empty batch')

    dataset_names, questions, images, answers = zip(*batch)
    dataset_names = list(dataset_names)
    questions = list(questions)
    images = list(images)
    answers = list(answers)

    if any(not isinstance(value, str) or not value.strip() for value in questions + answers):
        raise TypeError('Single-turn collate requires nonempty string questions and answers')

    # Keep the full question. TextEncoder splits long inputs into CLIP chunks.
    text_inputs = processor.tokenizer(questions, return_tensors = 'pt', padding = True, truncation = False, verbose = False)

    image_inputs = processor.image_processor(images = images, return_tensors = 'pt')

    return dataset_names, questions, text_inputs, image_inputs, answers

def make_collate_fn(processor):
    """Attach the processor while keeping the function picklable for DataLoader workers."""
    return functools.partial(collate_fn, processor = processor)

def _cache_is_current(manifest, config):
    """Check one loaded manifest and its cached training data."""
    if not isinstance(manifest, Mapping) or manifest.get('config') != config:
        return False

    path = manifest.get('train_path')
    if not path:
        return False
    try:
        digest = _file_sha256(path)
    except OSError:
        return False

    return digest == manifest.get('train_sha256')

def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _cache_config():
    """Fingerprint preprocessing only; subsets do not trigger source rebuilds."""
    source = SOURCES['llava_next']
    return {
        'protocol': PREPROCESSING_PROTOCOL,
        'source': source['repo'],
        'revision': source['revision'],
        'patterns': list(source['patterns']),
        'expected_shards': source['expected_shards'],
        'expected_rows': source['expected_rows'],
        'selection': 'original_first_user_assistant_pair',
        'max_answer_characters': MAX_ANSWER_CHARACTERS,
        'reserved_marker': RESERVED_MARKER,
        'image_directory': os.path.abspath(IMAGE_DIRECTORY),
    }

@contextmanager
def _build_lock(path):
    """Serialize cold-cache builders on the Linux/macOS training host."""
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError('Prepare this cache on Linux/macOS before using it on this platform') from exc

    with open(path, 'a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

def _load_cache(manifest_path, config):
    try:
        manifest = load_json(manifest_path)
    except (ValueError, OSError):
        return None, None
    if not _cache_is_current(manifest, config):
        return None, None

    train = load_json(manifest['train_path'])
    paths = {sample['img_path'] for samples in train.values() for sample in samples.values()}
    if not paths or not all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in paths):
        print('Cached images are missing or empty; rebuilding the cache')
        return None, None
    # Images were decoded and byte-verified at build time. Re-reading every
    # image on every launch would defeat the cache; runtime decode still fails
    # explicitly if a nonempty file has subsequently been corrupted.
    return train, manifest

def grouped_train_val_split(dataset, seed = 42):
    """Split full data by identical-image paths, with a hard validation cap.

    A large group may exceed the 10% target, but never the hard cap. If every
    group exceeds the cap, fail explicitly rather than return an empty split.
    Differently encoded copies of one image are not detected by byte hashing.
    """
    if MAX_VALIDATION_SAMPLES < 1 or not 0 < VALIDATION_FRACTION < 1:
        raise ValueError('Validation requires a positive sample cap and a fraction between 0 and 1')
    groups = defaultdict(list)
    for index, image_path in enumerate(dataset.image_paths):
        groups[image_path].append(index)

    if len(groups) < 2:
        raise ValueError('At least two distinct images are required for a train/validation split')

    image_paths = sorted(groups)
    random.Random(seed).shuffle(image_paths)
    target = max(1, min(round(len(dataset) * VALIDATION_FRACTION), MAX_VALIDATION_SAMPLES))
    val_indices = []

    # Never split an image group or exceed the target during greedy filling.
    for image_path in image_paths:
        indices = groups[image_path]
        if len(val_indices) + len(indices) <= target and len(val_indices) + len(indices) < len(dataset):
            val_indices.extend(indices)
        if len(val_indices) == target:
            break

    if not val_indices:
        candidates = [path for path in image_paths if len(groups[path]) <= MAX_VALIDATION_SAMPLES]
        if not candidates:
            raise ValueError(
                f'No complete image group fits MAX_VALIDATION_SAMPLES={MAX_VALIDATION_SAMPLES}; '
                'increase this cap explicitly. Image groups will not be split.'
            )
        closest_image = min(candidates, key = lambda path: abs(len(groups[path]) - target))
        val_indices = groups[closest_image]

    val_indices = set(val_indices)
    train_indices = array('q', (index for index in range(len(dataset)) if index not in val_indices))
    if not train_indices or not val_indices:
        raise ValueError('Both training and validation must contain at least one sample')

    return data.Subset(dataset, train_indices), data.Subset(dataset, array('q', sorted(val_indices)))

def _sample_digest(dataset, indices):
    digest = hashlib.sha256()
    for index in indices:
        digest.update(json.dumps(dataset.sample_ids[index], ensure_ascii = False).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()

def train_dataset(seed = 42, *, max_train_samples = None, validation_seed = VALIDATION_SEED):
    """Return single-turn training/validation subsets with the existing API.

    Validation is chosen from ALL accepted first-turn records before any cap.
    Its seed is independent of the training subset seed. max_train_samples
    limits only training (unlike v9), and may be 1 for a smoke test. At least
    two image groups must still exist in the full processed source.

    The first build still processes the full pinned source, even for a small
    cap. Later caps/seeds reuse it. Run this file before DDP to prepare caches.
    """
    if max_train_samples is not None and (isinstance(max_train_samples, bool) or not isinstance(max_train_samples, int) or max_train_samples < 1):
        raise ValueError(f'max_train_samples must be a positive integer or None, got {max_train_samples!r}')
    if MAX_ANSWER_CHARACTERS is not None and (not isinstance(MAX_ANSWER_CHARACTERS, int) or MAX_ANSWER_CHARACTERS < 1):
        raise ValueError('MAX_ANSWER_CHARACTERS must be a positive integer or None')

    train_config = _cache_config()
    cache_key = hashlib.sha256(json.dumps(train_config, sort_keys = True).encode('utf-8')).hexdigest()
    configuration_directory = os.path.abspath(os.path.join(CACHE_DIRECTORY, cache_key))
    manifest_path = os.path.join(configuration_directory, 'manifest.json')
    train, manifest = _load_cache(manifest_path, train_config)

    if train is None:
        os.makedirs(configuration_directory, exist_ok = True)
        with _build_lock(os.path.join(configuration_directory, 'build.lock')):
            # Another process may have completed the build while this one waited.
            train, manifest = _load_cache(manifest_path, train_config)
            if train is None:
                train_samples, filter_counts = llava_next_instruct()
                train = {'llava_next': train_samples}
                generation_directory = tempfile.mkdtemp(prefix = 'generation-', dir = configuration_directory)
                train_path = os.path.join(generation_directory, 'train.json')
                save_as_json(train, train_path)
                manifest = {
                    'protocol': PREPROCESSING_PROTOCOL,
                    'config': train_config,
                    'accepted_first_turn_samples': len(train_samples),
                    'unique_image_byte_hashes': len({sample['image_sha256'] for sample in train_samples.values()}),
                    'filter_counts': filter_counts,
                    'train_path': train_path,
                    'train_sha256': _file_sha256(train_path),
                }
                # Publish only the complete immutable cache generation.
                save_as_json(manifest, manifest_path)

    _log_filter_counts(manifest.get('filter_counts', {}), cached = True)
    dataset = CustomDataset(train)
    train_set, val_set = grouped_train_val_split(dataset, validation_seed)
    full_train_count = len(train_set)

    # A single deterministic permutation gives nested subsets as the cap grows.
    if max_train_samples is not None and max_train_samples < len(train_set):
        selected = array('q', train_set.indices)
        random.Random(seed).shuffle(selected)
        train_set = data.Subset(dataset, array('q', sorted(selected[:max_train_samples])))

    selection = {
        'split_protocol': 'fixed-byte-image-groups-v1',
        'preprocessing_protocol': PREPROCESSING_PROTOCOL,
        'cache_sha256': manifest['train_sha256'],
        'validation_seed': validation_seed,
        'validation_fraction': VALIDATION_FRACTION,
        'max_validation_samples': MAX_VALIDATION_SAMPLES,
        'sampling_seed': seed,
        'max_train_samples': max_train_samples,
        'full_training_count': full_train_count,
        'training_count': len(train_set),
        'validation_count': len(val_set),
        'training_sample_ids_sha256': _sample_digest(dataset, train_set.indices),
        'validation_sample_ids_sha256': _sample_digest(dataset, val_set.indices),
        'validation_sample_ids': [dataset.sample_ids[index] for index in val_set.indices],
    }
    selection_key = hashlib.sha256(json.dumps(selection, sort_keys = True).encode('utf-8')).hexdigest()
    selection_path = os.path.join(configuration_directory, 'selections', selection_key + '.json')
    save_as_json(selection, selection_path)

    # A small training cap must not keep the entire JSON dataset attached to
    # every DataLoader worker. Retain only selected training and validation
    # records in the returned dataset, preserving both subsets' sample order.
    if len(train_set) < full_train_count:
        kept_indices = list(train_set.indices) + list(val_set.indices)
        compact_samples = {
            dataset.sample_ids[index]: train['llava_next'][dataset.sample_ids[index]]
            for index in kept_indices
        }
        compact = CustomDataset({'llava_next': compact_samples})
        train_count = len(train_set)
        train_set = data.Subset(compact, array('q', range(train_count)))
        val_set = data.Subset(compact, array('q', range(train_count, len(compact))))

    print(f'LlaVA-NeXT first-turn split: train={len(train_set):,} of {full_train_count:,}, val={len(val_set):,}; validation_seed={validation_seed}')
    print(f'Training sample IDs SHA256: {selection["training_sample_ids_sha256"]}')
    print(f'Validation sample IDs SHA256: {selection["validation_sample_ids_sha256"]}')
    print(f'Selection manifest: {selection_path}')

    return train_set, val_set

if __name__ == '__main__':
    train_set, val_set = train_dataset()

    print(f'train={len(train_set):,}, val={len(val_set):,}')
