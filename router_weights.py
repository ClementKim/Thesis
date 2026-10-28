"""RQ2: the expert weights that a checkpoint's router assigns to every question of the evaluated benchmarks.

The weights depend only on the image and the question, so only IPMoE.encode_prefix runs and nothing is generated.
Messages are built as VLMEvalKit's inference loop builds them, and inputs as IPMoEVLMEval.generate_inner does.
Run with PYTHONPATH="$PWD/VLMEvalKit" like eval.py, e.g.
python router_weights.py --checkpoint checkpoint/llama/seed_42/checkpoint.pt --output evaluation/router/llama/seed_42.csv
"""

import argparse
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from vlmeval.dataset import build_dataset

from vlmeval_adapter import IPMoEVLMEval

DATASETS = ['NaturalBenchDataset', 'CV-Bench-2D', 'CV-Bench-3D', 'MMBench_DEV_EN_V11', 'MMStar', 'POPE', 'SEEDBench_IMG']
# Ability labels kept for the per-category analysis; each benchmark has a subset of them.
LABELS = ('category', 'l2-category', 'l2_category', 'source', 'type')

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description = 'Router weights of an IPMoE checkpoint on VLMEvalKit benchmarks')
    parser.add_argument('--checkpoint', required = True)
    parser.add_argument('--data', nargs = '+', default = DATASETS)
    parser.add_argument('--output', required = True, help = 'CSV with one row per question')
    args = parser.parse_args()

    model = IPMoEVLMEval(args.checkpoint)
    if model.model.ablation['router_mode'] != 'learned' or model.model.ablation.get('expert_set', 'all') != 'all':
        raise ValueError('Only arms with a learned router over all three experts have input-dependent weights')

    rows = []
    for name in args.data:
        dataset = build_dataset(name)
        model.set_dump_image(dataset.dump_image)
        for _, line in tqdm(dataset.data.iterrows(), total = len(dataset.data), desc = name):
            message = model.build_prompt(line, name) if model.use_custom_prompt(name) else dataset.build_prompt(line)
            question = '\n'.join(item['value'] for item in message if item['type'] == 'text')
            images = [item['value'] for item in message if item['type'] == 'image']
            if len(images) != 1:
                raise ValueError(f'{name} {line["index"]}: IPMoE requires exactly one image; received {len(images)}')

            text_inputs = model.processor.tokenizer([question], return_tensors = 'pt', padding = True, truncation = False, verbose = False)
            with Image.open(images[0]) as image:
                image_inputs = model.processor.image_processor(images = [image.convert('RGB')], return_tensors = 'pt')
            text_inputs = {key: value.to(model.device) for key, value in text_inputs.items()}
            image_inputs = {key: value.to(model.device) for key, value in image_inputs.items()}

            _, _, weights = model.model.encode_prefix(text_inputs, image_inputs)
            rows.append({'dataset': name, 'index': line['index'], **{label: line[label] for label in LABELS if label in line},
                         **dict(zip(('self', 'cross', 'hybrid'), weights[0].float().tolist()))})

    weights = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok = True)
    weights.to_csv(args.output, index = False)

    # Mean and spread over questions; a router that ignores its input has a spread near zero.
    print(weights.groupby('dataset')[['self', 'cross', 'hybrid']].agg(['mean', 'std', 'min', 'max']).round(4).to_string())

if __name__ == '__main__':
    main()
