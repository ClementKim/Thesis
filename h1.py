"""H1: does each object word's interaction map point at that object's box in CV-Bench-3D?

The map is the bilinear text-patch similarity that IPMoE.encode_prefix builds before any spatial
averaging, so only the CLIP encoders and their trained MLPs are loaded, not the language model.
For named objects i, j, d = A_i + A_j - 1, where A_i is the probability that word i's map ranks a random
cell of box i above a random cell of box j (ties count half). This is how much more word i prefers box i
over box j than word j does, so a component both words weigh equally, such as image saliency, box size or
the drawn boxes, cancels. A word-level prior does not, e.g. maps that follow saliency more closely for words
naming large objects. d depends only on each map's ranks, so how peaked a model's maps are does not change it.

Primary test: the full arm's Depth d, averaged over seeds per question, exceeds zero (one-sided Wilcoxon,
after each run's sign is fixed on Distance). Per-seed tests, comparisons with the all_map_mean arm and
frozen CLIP, and the accuracy regression are secondary and not corrected for multiple comparisons.
"""

import argparse
import ast
import base64
import io
import json
import os
import re
from argparse import Namespace
from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import special, stats
from tqdm import tqdm

from model.encoder import encoding
from model.interaction import Interaction
from vlmeval_adapter import mcq_prompt

LMU_ROOT = os.environ.get('LMUData', os.path.expanduser('~/LMUData'))

# Bump when d or the saved columns change, so older outputs are measured again instead of pooled or compared.
VERSION = 2

# "the table (highlighted by a red box)"; every CV-Bench-3D question names its objects this way.
PATTERN = re.compile(r'\bthe ([A-Za-z][A-Za-z -]*?) \(highlighted by an? (\w+) box\)')
CHANNELS = {'red': 0, 'green': 1, 'blue': 2}

def similarity_of(text, patches, similarity):
    """'cosine' L2-normalizes every token and patch first; 'dot' keeps the raw features."""
    if similarity == 'cosine':
        text, patches = F.normalize(text, dim = -1), F.normalize(patches, dim = -1)

    return torch.bmm(text, patches.transpose(1, 2))

class TrainedMap(nn.Module):
    """The checkpoint's text and vision MLPs on frozen CLIP, combined as in IPMoE.encode_prefix."""

    def __init__(self, text_encoder, vision_encoder, interaction, similarity = 'map'):
        super().__init__()

        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.interaction = interaction
        self.similarity = similarity

    def forward(self, text_inputs, image_inputs):
        text_features, _ = self.text_encoder(**text_inputs)
        patches = self.vision_encoder(**image_inputs)
        if self.similarity == 'map':
            return self.interaction(text_features, patches)

        return similarity_of(text_features, patches, self.similarity)

class CLIPMap(nn.Module):
    """Frozen CLIP cosine similarity between the same text tokens and penultimate patches, without trained MLPs."""

    def __init__(self, text_encoder, similarity = 'map'):
        super().__init__()

        # Reuse the text encoder's chunking with CLIP's own projection only.
        text_encoder.mlp = nn.Identity()
        text_encoder.add_chunk_position = False
        self.text_encoder = text_encoder
        self.clip = text_encoder.clip
        self.similarity = 'cosine' if similarity == 'map' else similarity

    def forward(self, text_inputs, image_inputs):
        text_features, _ = self.text_encoder(**text_inputs)
        hidden = self.clip.vision_model(pixel_values = image_inputs['pixel_values'], output_hidden_states = True).hidden_states[-2][:, 1:]
        patches = self.clip.visual_projection(self.clip.vision_model.post_layernorm(hidden))

        return similarity_of(text_features, patches, self.similarity)

def load_map(checkpoint_path, device, similarity = 'map'):
    # similarity 'map' keeps each map's own definition; the other choices test what normalization alone explains.
    if checkpoint_path is None:
        processor, text_encoder, _ = encoding(0.0, device)
        return processor, CLIPMap(text_encoder, similarity).to(device).eval(), {'map': 'clip', 'similarity': similarity}

    checkpoint = torch.load(checkpoint_path, map_location = 'cpu', weights_only = True)
    args = Namespace(**checkpoint['args'])
    saved = checkpoint['model']
    if not isinstance(saved, Mapping):
        raise TypeError('Checkpoint model must be a parameter mapping')

    processor, text_encoder, vision_encoder = encoding(args.drop, device, args.embed_dim)

    # Same strict key and shape checks as the evaluation adapter, restricted to the two encoders.
    for prefix, module in (('text_encoder.', text_encoder), ('vision_encoder.', vision_encoder)):
        parameters = {name: parameter for name, parameter in module.named_parameters() if parameter.requires_grad}
        found = {name[len(prefix):]: value for name, value in saved.items() if name.startswith(prefix)}
        if set(found) != set(parameters):
            hint = '; image_only checkpoints have no interaction map' if not found else ''
            raise ValueError(f'Checkpoint {prefix[:-1]} keys differ: missing={sorted(set(parameters) - set(found))}, '
                             f'unexpected={sorted(set(found) - set(parameters))}{hint}')

        with torch.no_grad():
            for name, parameter in parameters.items():
                if found[name].shape != parameter.shape:
                    raise ValueError(f'Checkpoint shape mismatch for {prefix}{name}: {tuple(found[name].shape)} != {tuple(parameter.shape)}')
                parameter.copy_(found[name])

    model = TrainedMap(text_encoder, vision_encoder, Interaction(args.interaction_type), similarity).to(device).eval()
    # rebuttal.py init saves the MLPs a seed starts training from, which must not pool with trained maps.
    meta = {'map': 'initial' if checkpoint.get('initial') else 'trained', 'checkpoint': checkpoint_path, 'ablation': getattr(args, 'ablation', 'full'),
            'llm_type': args.llm_type, 'seed': args.seed, 'similarity': similarity}

    return processor, model, meta

def load_image(line, image_root):
    # Prefer the file VLMEvalKit dumped, which is what evaluation fed the model.
    path = os.path.join(image_root, line['image_path'])
    if os.path.exists(path):
        with Image.open(path) as image:
            return image.convert('RGB')

    return Image.open(io.BytesIO(base64.b64decode(line['image']))).convert('RGB')

def drawn_colour(pixels, box, band = 3):
    """Colour of the rectangle CV-Bench drew around a box; saturated border pixels vote."""
    height, width, _ = pixels.shape
    x1, x2 = (int(round(min(max(value, 0), width - 1))) for value in (box[0], box[2]))
    y1, y2 = (int(round(min(max(value, 0), height - 1))) for value in (box[1], box[3]))

    strips = [pixels[max(0, y - band):y + band + 1, x1:x2 + 1] for y in (y1, y2)]
    strips += [pixels[y1:y2 + 1, max(0, x - band):x + band + 1] for x in (x1, x2)]
    border = np.concatenate([strip.reshape(-1, 3) for strip in strips]).astype(int)

    votes = {}
    for colour, channel in CHANNELS.items():
        others = np.delete(border, channel, axis = 1).max(axis = 1)
        votes[colour] = np.mean((border[:, channel] > 150) & (others < 90))

    return max(votes, key = votes.get)

def box_cells(box, width, height, grid):
    """Grid cells whose centres fall inside the box; the processor resizes to a square without cropping."""
    x1, y1, x2, y2 = box
    xs = (np.arange(grid) + 0.5) * width / grid
    ys = (np.arange(grid) + 0.5) * height / grid
    cells = (ys[:, None] >= y1) & (ys[:, None] <= y2) & (xs[None] >= x1) & (xs[None] <= x2)

    # A box smaller than one cell keeps the cell containing its centre.
    if not cells.any():
        row = min(max(int((y1 + y2) / 2 * grid / height), 0), grid - 1)
        col = min(max(int((x1 + x2) / 2 * grid / width), 0), grid - 1)
        cells[row, col] = True

    return cells.ravel()

def word_rows(tokenizer, prompt, span):
    """Text-feature rows of prompt[span]; CLIP splits words before BPE, so the prefix keeps its tokens."""
    def ids(text):
        return tokenizer(text, verbose = False)['input_ids'][1:-1]

    content, before, word = ids(prompt), ids(prompt[:span[0]]), ids(prompt[span[0]:span[1]])
    rows = list(range(len(before), len(before) + len(word)))
    if not word or content[rows[0]:rows[-1] + 1] != word:
        raise ValueError(f'Could not locate {prompt[span[0]:span[1]]!r} in the CLIP tokens')

    return rows

def preference(word_map, own, other):
    """Probability that the map ranks a random cell of the own box above one of the other box, ties counting half."""
    own, other = word_map[own][:, None], word_map[other][None]

    return float(np.mean(own > other) + 0.5 * np.mean(own == other))

def shortcut(category, boxes, answer):
    """Whether a 2D cue alone gives the answer: the lower box for Depth, the box nearer the reference for Distance."""
    if category == 'Depth':
        guess = 'AB'[int(boxes[-1][3] > boxes[-2][3])]
    else:
        centres = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in boxes])
        distances = np.linalg.norm(centres[-2:] - centres[0], axis = 1)
        guess = 'AB'[int(distances[1] < distances[0])]

    return guess == answer

@torch.no_grad()
def measure(line, processor, model, device, image_root, center = False):
    """Score one question, or return the reason it was skipped."""
    prompt = mcq_prompt(line)
    offset = prompt.index(line['question'])
    objects = [(match.group(1), match.group(2), (offset + match.start(1), offset + match.end(1)))
               for match in PATTERN.finditer(line['question'])]
    boxes = ast.literal_eval(line['bbox'])

    # Box lists are not in naming order for every task, so match each box to its object by the drawn colour.
    image = load_image(line, image_root)
    drawn = [drawn_colour(np.asarray(image), box) for box in boxes]
    named = [colour for _, colour, _ in objects]
    if len(objects) < 2 or len(objects) != len(boxes):
        return 'question does not name one object per box'
    if sorted(drawn) != sorted(named) or len(set(drawn)) != len(drawn):
        return 'drawn box colours do not match the question'
    # Options A and B name the last two objects, so the answer letter identifies a box.
    if [line['A'], line['B']] != [name for name, _, _ in objects[-2:]]:
        return 'options do not name the last two objects'
    boxes = [boxes[drawn.index(colour)] for colour in named]

    ## Interaction Map
    text_inputs = processor.tokenizer([prompt], return_tensors = 'pt', padding = True, truncation = False, verbose = False)
    image_inputs = processor.image_processor(images = [image], return_tensors = 'pt')
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
    similarity = model(text_inputs, image_inputs)[0].float().cpu().numpy()
    if center:
        # Remove each patch's mean over the tokens, i.e. whatever every word's map shares, such as image saliency.
        similarity = similarity - similarity.mean(axis = 0, keepdims = True)

    if similarity.shape[0] != max(int(text_inputs['attention_mask'].sum()) - 2, 1):
        raise RuntimeError('Interaction map rows do not match the CLIP content tokens')
    grid = int(round(similarity.shape[1] ** 0.5))
    if grid * grid != similarity.shape[1]:
        raise RuntimeError('Interaction map columns do not form a square patch grid')

    ## Word-by-Box Scores
    # Every score below depends on each word's ranking of the patches, so the maps need no normalization.
    maps = np.stack([similarity[word_rows(processor.tokenizer, prompt, span)].mean(axis = 0) for _, _, span in objects])
    cells = np.stack([box_cells(box, image.width, image.height, grid) for box in boxes])
    scores = np.array([[word_map[box].mean() for box in cells] for word_map in maps])

    def peak_box(cell):
        # The named box holding a map's extreme cell, or -1; boxes of the questions that are scored do not overlap.
        inside = np.flatnonzero(cells[:, cell])
        return int(inside[0]) if len(inside) else -1

    k = len(objects)
    pairs = list(combinations(range(k), 2))

    return {
        'index': int(line['index']),
        'category': line['category'],
        'objects': [name for name, _, _ in objects],
        'colours': named,
        'n_objects': k,
        'overlap': any((cells[i] & cells[j]).any() for i, j in pairs),
        'd': float(np.mean([preference(maps[i], cells[i], cells[j]) + preference(maps[j], cells[j], cells[i]) - 1 for i, j in pairs])),
        # A negative sign flips every comparison, so both directions are kept until the sign is chosen.
        'own_top': [bool(scores[i].argmax() == i) for i in range(k)],
        'own_top_neg': [bool(scores[i].argmin() == i) for i in range(k)],
        'peak_box': [peak_box(word_map.argmax()) for word_map in maps],
        'peak_box_neg': [peak_box(word_map.argmin()) for word_map in maps],
        'box_fraction': [float(box.mean()) for box in cells],
        'shortcut': shortcut(line['category'], boxes, line['answer']),
    }

def logistic(y, X):
    """Logistic regression by Newton's method: coefficients and two-sided Wald p-values, intercept first."""
    X = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(X.shape[1])
    try:
        for _ in range(100):
            p = special.expit(X @ beta)
            information = X.T @ (X * (p * (1 - p))[:, None])
            step = np.linalg.solve(information, X.T @ (y - p))
            beta = beta + step
            if np.abs(step).max() < 1e-8:
                break
        else:
            # Separable data has no finite estimate.
            return None
        se = np.sqrt(np.diag(np.linalg.inv(information)))
    except np.linalg.LinAlgError:
        return None

    return beta, 2 * stats.norm.sf(np.abs(beta / se))

def summarize(records, meta, skipped):
    # The map's sign is arbitrary (frozen CLIP's comes out inverted), so fix it on Distance, then test on Depth.
    # Superseded for reporting: this rule flips seed 42 on a median of -3.7e-17; rebuttal.py h1 recomputes every
    # statistic from items.csv with |median| < 1e-9 as a tie of sign +1, and keeps this rule as its sensitivity check.
    calibration = [r['d'] for r in records if r['category'] == 'Distance' and not r['overlap']]
    sign = -1 if calibration and np.median(calibration) < 0 else 1
    suffix = '' if sign > 0 else '_neg'

    summary = {**meta, 'sign': sign, 'sign_chosen_on': 'Distance', 'skipped': skipped, 'categories': {}}
    for category in sorted({r['category'] for r in records}):
        rows = [r for r in records if r['category'] == category]
        kept = [r for r in rows if not r['overlap']]
        test = category != 'Distance'
        d = np.array([sign * r['d'] for r in kept])
        own = np.array([hit for r in kept for hit in r['own_top' + suffix]], dtype = bool)
        localized = np.array([all(r['own_top' + suffix]) for r in kept], dtype = bool)

        # A map that ignores the word sends every word's peak to the same place, so 1/k of the peaks that land
        # in a named box land in their own whatever the box sizes. Landing in any named box is not word-specific.
        peaks = np.array([peak for r in kept for peak in r['peak_box' + suffix]], dtype = int)
        owners = np.array([i for r in kept for i in range(r['n_objects'])], dtype = int)
        landed = peaks >= 0

        result = {
            'role': 'test' if test else 'calibration',
            'items': len(rows),
            'overlapping_excluded': len(rows) - len(kept),
            'd_mean': float(d.mean()) if len(d) else None,
            'd_median': float(np.median(d)) if len(d) else None,
            'd_positive': float(np.mean(d > 0)) if len(d) else None,
            # Testing Distance would be circular, since it chose the sign.
            'd_wilcoxon_p': float(stats.wilcoxon(d, alternative = 'greater').pvalue) if test and len(d) else None,
            'own_box_top': float(own.mean()) if len(own) else None,
            'pointing_own_box': float(np.mean(peaks[landed] == owners[landed])) if landed.any() else None,
            'own_box_chance': 1 / rows[0]['n_objects'],
            'all_words_own_box': float(localized.mean()) if len(localized) else None,
            'peak_in_any_box': float(landed.mean()) if len(landed) else None,
            'peak_in_any_box_chance': float(np.mean([sum(r['box_fraction']) for r in kept])) if kept else None,
        }

        if 'hit' in rows[0] and len(kept):
            hit = np.array([r['hit'] for r in kept], dtype = float)
            cue = np.array([r['shortcut'] for r in kept], dtype = float)
            size = np.log([np.mean(r['box_fraction']) for r in kept])
            result['accuracy'] = float(hit.mean())
            result['shortcut_accuracy'] = float(cue.mean())
            # Grounding and correctness share causes, so adjust for the 2D shortcut and box size.
            fit = logistic(hit, np.column_stack([d, cue, size]))
            result['hit_logit'] = None if fit is None else {
                name: {'coefficient': float(b), 'p': float(p)} for name, b, p in zip(('intercept', 'd', 'shortcut', 'log_box_fraction'), *fit)}

        summary['categories'][category] = result

    return summary

def depth_d(items, summary):
    """Signed d of a run's Depth questions without overlapping boxes, indexed by question."""
    kept = items[(items['category'] == 'Depth') & ~items['overlap']]

    return kept.set_index('index')['d'] * summary['sign']

def load_runs(directories):
    """Outputs of one arm's seeds, or of a single reference without a seed, with their signed Depth d."""
    runs = []
    for directory in directories:
        with open(os.path.join(directory, 'summary.json')) as handle:
            summary = json.load(handle)
        if 'pooled' in summary:
            raise ValueError(f'{directory} is pooled; pass its per-seed runs instead')
        if summary.get('version') != VERSION:
            raise ValueError(f'{directory} was measured by another version of h1.py; measure it again')
        runs.append((summary, depth_d(pd.read_csv(os.path.join(directory, 'items.csv')), summary)))

    seeds = [summary.get('seed') for summary, _ in runs]
    if len(set(seeds)) != len(seeds) or (None in seeds and len(seeds) > 1):
        raise ValueError(f'Expected one run per seed or a single reference: {directories}')

    return runs

def check_compatible(summaries):
    """Runs that are pooled or compared must cover the same questions and, if trained, use the same language model."""
    # Runs measured before --center or --similarity existed are uncentred and use each map's own similarity.
    centers = {summary.get('center', False) for summary in summaries}
    if len(centers) > 1:
        raise ValueError('Runs differ in center: centred and uncentred maps cannot be pooled or compared')
    if len({summary.get('similarity', 'map') for summary in summaries}) > 1:
        raise ValueError('Runs differ in similarity and cannot be pooled or compared')
    for key in ('limit', 'llm_type'):
        values = {summary[key] for summary in summaries if key in summary}
        if len(values) > 1:
            raise ValueError(f'Runs differ in {key}: {sorted(values, key = str)}')

def pool(runs, directories):
    """Average each question's signed Depth d over one arm's seeds; for the full arm this is the primary test."""
    summaries = [summary for summary, _ in runs]
    for key in ('map', 'ablation'):
        if len({summary.get(key) for summary in summaries}) > 1:
            raise ValueError(f'Pooled runs differ in {key}')

    table = pd.concat({f'seed_{summary.get("seed")}': d for summary, d in runs}, axis = 1).dropna()
    d = table.mean(axis = 1)
    table['mean'] = d
    ablation = summaries[0].get('ablation')

    summary = {
        'version': VERSION, 'pooled': directories, 'map': summaries[0]['map'], 'ablation': ablation,
        'llm_type': summaries[0].get('llm_type'), 'limit': summaries[0]['limit'], 'center': summaries[0].get('center', False),
        'similarity': summaries[0].get('similarity', 'map'),
        'seeds': [summary.get('seed') for summary in summaries], 'signs': [summary['sign'] for summary in summaries],
        'Depth': {
            'test': 'primary' if ablation == 'full' else 'secondary',
            'items': len(d),
            'd_mean': float(d.mean()) if len(d) else None,
            'd_median': float(d.median()) if len(d) else None,
            'd_positive': float((d > 0).mean()) if len(d) else None,
            'd_wilcoxon_p': float(stats.wilcoxon(d, alternative = 'greater').pvalue) if len(d) else None,
            'per_seed': [{'seed': summary.get('seed'), **{key: summary['categories']['Depth'][key]
                          for key in ('d_mean', 'd_wilcoxon_p', 'own_box_top', 'pointing_own_box')}} for summary in summaries],
        },
    }

    return summary, table

def compare(mine, others, directories):
    """Two-sided paired Wilcoxon on Depth d against another arm; runs pair by seed, and a reference pairs with every run."""
    differences = {}
    for summary, d in mine:
        paired = [other for other_summary, other in others if other_summary.get('seed') in (None, summary.get('seed'))]
        if paired:
            differences[summary.get('seed')] = d - paired[0]

    result = {'against': directories, 'seeds': list(differences), 'items': 0}
    if differences:
        difference = pd.concat(differences.values(), axis = 1).dropna().mean(axis = 1)
        result.update(items = len(difference), d_mean_difference = float(difference.mean()),
                      wilcoxon_p = float(stats.wilcoxon(difference).pvalue),
                      per_seed = [{'seed': seed, 'd_mean_difference': float(value.mean())} for seed, value in differences.items()])

    return result

def main():
    parser = argparse.ArgumentParser(description = 'H1 grounding test on CV-Bench-3D boxes')
    source = parser.add_mutually_exclusive_group(required = True)
    source.add_argument('--checkpoint', help = 'IPMoE checkpoint whose interaction map is measured')
    source.add_argument('--reference', choices = ['clip'], help = 'Measure frozen CLIP word-patch similarity instead')
    source.add_argument('--pool', nargs = '+', metavar = 'RUN', help = "Output directories of one arm's seeds; tests their seed-averaged Depth d")
    parser.add_argument('--data', default = os.path.join(LMU_ROOT, 'CV-Bench-3D.tsv'))
    parser.add_argument('--image-root', default = os.path.join(LMU_ROOT, 'images', 'CV-Bench-3D'))
    parser.add_argument('--predictions', default = None, help = 'VLMEvalKit CV-Bench-3D result .xlsx with a hit column')
    parser.add_argument('--compare-with', nargs = '+', action = 'append', default = [], metavar = 'RUN',
                        help = 'Output directories of another arm, paired by seed, or of the CLIP reference; repeat for each arm')
    parser.add_argument('--output-dir', required = True)
    parser.add_argument('--limit', type = int, default = None, help = 'Questions per category, for a quick check')
    parser.add_argument('--center', action = 'store_true', help = "Subtract each patch's mean over the tokens before scoring")
    parser.add_argument('--similarity', choices = ['map', 'cosine', 'dot'], default = 'map',
                        help = "map: the map's own (trained: raw dot product, CLIP: cosine); cosine or dot for either map")
    args = parser.parse_args()
    if args.pool and (args.predictions or args.limit is not None or args.center or args.similarity != 'map'):
        parser.error('--predictions, --limit, --center and --similarity apply when measuring, not with --pool')

    # Read everything to compare with first, so a mismatch fails before the measurement.
    others = [load_runs(directories) for directories in args.compare_with]
    compared = [summary for group in others for summary, _ in group]

    if args.pool:
        runs = load_runs(args.pool)
        check_compatible([summary for summary, _ in runs] + compared)
        summary, table = pool(runs, args.pool)

        os.makedirs(args.output_dir, exist_ok = True)
        table.to_csv(os.path.join(args.output_dir, 'items.csv'))

    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        processor, model, meta = load_map(args.checkpoint, device, args.similarity)
        meta.update(version = VERSION, limit = args.limit, center = args.center)
        check_compatible([meta] + compared)

        data = pd.read_csv(args.data, sep = '\t')
        if args.limit is not None:
            data = data.groupby('category', sort = False).head(args.limit)

        hits = None
        if args.predictions:
            hits = pd.read_excel(args.predictions).set_index('index')['hit']
            missing = set(data['index']) - set(hits.index)
            if missing:
                raise ValueError(f'{args.predictions} lacks {len(missing)} questions, e.g. {sorted(missing)[:5]}')
            meta['predictions'] = args.predictions

        records, skipped = [], {}
        for _, line in tqdm(data.iterrows(), total = len(data), desc = 'H1'):
            record = measure(line, processor, model, device, args.image_root, center = args.center)
            if isinstance(record, str):
                skipped[record] = skipped.get(record, 0) + 1
                continue
            if hits is not None:
                record['hit'] = int(hits.loc[record['index']])
            records.append(record)

        summary = summarize(records, meta, skipped)
        items = pd.DataFrame(records)
        runs = [(summary, depth_d(items, summary))]

        os.makedirs(args.output_dir, exist_ok = True)
        for column in ('objects', 'colours', 'own_top', 'own_top_neg', 'peak_box', 'peak_box_neg', 'box_fraction'):
            items[column] = items[column].map(json.dumps)
        items.to_csv(os.path.join(args.output_dir, 'items.csv'), index = False)

    summary['comparisons'] = [compare(runs, group, directories) for group, directories in zip(others, args.compare_with)]
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as handle:
        json.dump(summary, handle, indent = 2)

    print(json.dumps(summary, indent = 2))

if __name__ == '__main__':
    main()
