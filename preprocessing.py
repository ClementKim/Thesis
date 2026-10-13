import functools
import glob
import hashlib
import json
import os
import random
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping

import pyarrow.parquet as pq
import torch.utils.data as data
from PIL import Image
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Change this when the preprocessing rules change to rebuild the caches.
PREPROCESSING_PROTOCOL = 'simoe-llava-next-singleturn-v2'
CACHE_DIRECTORY = 'dataset/cache'

# Pin the dataset version so repeated builds use the same source data.
SOURCES = {
    'llava_next': {
        'repo': 'lmms-lab/LLaVA-NeXT-Data',
        'revision': 'dff8edfca0f16f9b4f1407331f6659aa881fc016',
        'patterns': ('data/train-*.parquet',),
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
                return handle.read()

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

def _prepare_directory(directory):
    """Spread images across 256 folders to avoid one very large directory."""
    if directory in _PREPARED_DIRECTORIES:
        return

    for prefix in range(256):
        os.makedirs(os.path.join(directory, f'{prefix:02x}'), exist_ok = True)

    _PREPARED_DIRECTORIES.add(directory)

def persist_image(raw, directory, written):
    """Save the original bytes once per image. Identical bytes share a path."""
    digest = hashlib.sha256(raw).hexdigest()
    _prepare_directory(directory)
    path = os.path.join(directory, digest[:2], f'{digest}{_extension(raw)}')

    # Track saved hashes to avoid checking the same file repeatedly.
    if digest in written:
        return path

    # Rename only after writing so interrupted writes do not leave broken images.
    if not os.path.exists(path):
        temporary = f'{path}.tmp.{os.getpid()}'

        with open(temporary, 'wb') as handle:
            handle.write(raw)

        os.replace(temporary, path)

    written.add(digest)

    return path

def _first_user_assistant_pair(conversations):
    """Keep the first nonempty user question and the next nonempty assistant answer."""
    question = None

    for turn in conversations:
        role = turn.get('from', turn.get('role', ''))
        value = turn.get('value', turn.get('content', ''))
        role = str(role).strip().lower()
        value = str(value or '').strip()

        if not value:
            continue

        if role in {'human', 'user'}:
            if question is not None:
                return '', ''
            question = value

        elif question is not None and role in {'gpt', 'assistant'}:
            return question, value

    return '', ''

def llava_next_instruct(*, max_samples = None):
    """Build {sample_id: record}, skipping missing images, incomplete pairs, and duplicate IDs."""
    directory = 'dataset/llava-next/images'
    columns = ('id', 'image', 'conversations', 'data_source')

    samples = {}
    written = set()
    counters = defaultdict(int)

    for index, item in enumerate(iter_rows('llava_next', columns, desc = 'llava-next')):
        if max_samples is not None and counters['accepted'] >= max_samples:
            break

        raw = encoded_image_bytes(item.get('image'))
        if raw is None:
            counters['no_image'] += 1
            continue

        question, answer = _first_user_assistant_pair(item.get('conversations') or [])

        # The model inserts image features itself, so remove the old marker.
        question = re.sub(r'<\s*image\s*>', ' ', question, flags = re.IGNORECASE)
        question = question.replace('\r\n', '\n').replace('\r', '\n')
        question = re.sub(r'[^\S\n]+', ' ', question).strip()

        if not question or not answer:
            counters['malformed'] += 1
            continue

        # Different sources may use the same ID.
        source_name = str(item.get('data_source') or 'unknown')
        source_id = item.get('id') or index
        sample_id = f'{source_name}/{source_id}'
        if sample_id in samples:
            counters['duplicate_id'] += 1
            continue

        image_path = persist_image(raw, directory, written)
        samples[sample_id] = {
            'question': question,
            'answer': answer,
            'img_path': image_path,
        }

        counters['accepted'] += 1

    counts = [f'{key}={value:,}' for key, value in sorted(counters.items())]
    print('[llava-next] ' + ', '.join(counts))

    if not samples:
        raise RuntimeError('No LLaVA-NeXT samples survived filtering')

    return samples

def grouped_train_val_split(samples, *, val_fraction, seed):
    """Keep all questions for each image together, targeting val_fraction of samples."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError('val_fraction must be between 0 and 1')

    groups = defaultdict(list)
    for sample_id, sample in samples.items():
        image_path = str(sample['img_path'])
        groups[image_path].append(sample_id)

    if len(groups) < 2:
        raise ValueError('At least two distinct images are required for a grouped split')

    # Sort first so the same seed selects the same images regardless of input order.
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)

    target = max(1, round(len(samples) * val_fraction))
    validation = {}

    # Reserve the last image group for training. Whole groups may exceed the target.
    for group_id in group_ids[:-1]:
        if len(validation) >= target:
            break

        for sample_id in groups[group_id]:
            validation[sample_id] = samples[sample_id]

    train = {}
    for sample_id, sample in samples.items():
        if sample_id not in validation:
            train[sample_id] = sample

    if not train or not validation:
        raise RuntimeError('Grouped split produced an empty partition')

    return train, validation

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

                if not image_path:
                    continue

                self.dataset_names.append(str(dataset_name))
                self.sample_ids.append(str(sample_id))
                self.questions.append(str(sample.get('question', '')))
                self.image_paths.append(str(image_path))
                self.answers.append(str(sample.get('answer', '')))

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

    # Keep the full question. TextEncoder splits long inputs into CLIP chunks.
    text_inputs = processor.tokenizer(questions, return_tensors = 'pt', padding = True, truncation = False, verbose = False)

    image_inputs = processor.image_processor(images = images, return_tensors = 'pt')

    return dataset_names, questions, text_inputs, image_inputs, answers

def make_collate_fn(processor):
    """Attach the processor while keeping the function picklable for DataLoader workers."""
    return functools.partial(collate_fn, processor = processor)

def _cache_is_current(manifest, config):
    """Check one loaded manifest and its immutable partition files."""
    if not isinstance(manifest, Mapping) or manifest.get('config') != config:
        return False

    for name in ('train', 'validation'):
        path = manifest.get(f'{name}_path')
        if not path:
            return False
        try:
            digest = hashlib.sha256()
            with open(path, 'rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
        except OSError:
            return False
        if digest.hexdigest() != manifest.get(f'{name}_sha256'):
            return False

    return True

def train_dataset(val_fraction = 0.01, seed = 42, *, max_train_samples = None):
    """Load cached train/validation datasets, or build and save a new split."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError('val_fraction must be between 0 and 1')

    # Changing any of these settings requires a new split.
    source = SOURCES['llava_next']
    train_config = {
        'protocol': PREPROCESSING_PROTOCOL,
        'source': source['repo'],
        'revision': source['revision'],
        'patterns': list(source['patterns']),
        'max_train_samples': max_train_samples,
        'val_fraction': val_fraction,
        'seed': seed,
    }

    cache_key = hashlib.sha256(json.dumps(train_config, sort_keys = True).encode('utf-8')).hexdigest()
    configuration_directory = os.path.abspath(os.path.join(CACHE_DIRECTORY, cache_key))
    manifest_path = os.path.join(configuration_directory, 'manifest.json')
    try:
        manifest = load_json(manifest_path)
    except (ValueError, OSError):
        manifest = None

    if _cache_is_current(manifest, train_config):
        train = load_json(manifest['train_path'])
        validation = load_json(manifest['validation_path'])
    else:
        complete = llava_next_instruct(max_samples = max_train_samples)

        train_samples, validation_samples = grouped_train_val_split(complete, val_fraction = val_fraction, seed = seed)

        train = {'llava_next': train_samples}
        validation = {'llava_next': validation_samples}

        os.makedirs(configuration_directory, exist_ok = True)
        generation_directory = tempfile.mkdtemp(prefix = 'generation-', dir = configuration_directory)
        train_path = os.path.join(generation_directory, 'train.json')
        validation_path = os.path.join(generation_directory, 'validation.json')
        save_as_json(train, train_path)
        save_as_json(validation, validation_path)

        manifest = {
            'protocol': PREPROCESSING_PROTOCOL,
            'config': train_config,
            'train_samples': len(train_samples),
            'validation_samples': len(validation_samples),
        }

        for name, path in (('train', train_path), ('validation', validation_path)):
            digest = hashlib.sha256()
            with open(path, 'rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            manifest[f'{name}_path'] = path
            manifest[f'{name}_sha256'] = digest.hexdigest()

        # Only completed generations are published; their files are never overwritten.
        save_as_json(manifest, manifest_path)

        print(f'Grouped LLaVA-NeXT split: train={len(train_samples):,}, val={len(validation_samples):,}')

    return CustomDataset(train), CustomDataset(validation)

if __name__ == '__main__':
    train_set, validation_set = train_dataset()

    print(f'train={len(train_set):,}, val={len(validation_set):,}')
