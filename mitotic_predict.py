#!/usr/bin/env python3
"""
mitotic_predict.py - 3-class mitotic classifier over all tiles of each slide,
applied as a refinement of predictions.json.

Classes:
    0 - non-mitotic
    1 - Mitotic Figure
    2 - Bizarre Figure

A tile is positive when the arg-max class is 1 or 2 and its probability is at
least --threshold; otherwise it is non-mitotic.

Update rules per tile (apply_mitotic_results, also used by update_clusters.py):
  - positive: mitotic = predicted label, every other label = "None".
    The labels being replaced are kept under `non_mitotic_fallback`.
  - non-mitotic, but the entry currently says mitotic: mitotic = "None" and the
    other labels are restored from `non_mitotic_fallback`.
  - non-mitotic and the entry is non-mitotic: untouched.
  - positive and the tile has no entry: a new entry is inserted (standalone run only).
`mitotic_prob` and `mitotic_model_prob` hold this classifier's probability for
the label it assigned.

Usage:
    python mitotic_predict.py \
        --checkpoint mitotic_classifier.pth \
        --slides /path/to/slide_a /path/to/slide_b \
        --batch_size 64 \
        --threshold 0.9
"""
import os
import re
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RAW_SUBDIR    = os.path.join('tiles1', 'nonbox')
LABEL_COLS    = ['mitotic', 'chromatin', 'nucleoli', 'prominent', 'shape']
OTHER_COLS    = [c for c in LABEL_COLS if c != 'mitotic']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES   = ['non-mitotic', 'Mitotic Figure', 'Bizarre Figure']
POSITIVE      = ('Mitotic Figure', 'Bizarre Figure')
FALLBACK_KEY  = 'non_mitotic_fallback'


# ---------------------------------------------------------------------------
# Model - must match mitotic_train.py exactly
# ---------------------------------------------------------------------------

class MitoticClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))  # (B, 3)


def load_mitotic_model(checkpoint: str) -> tuple[MitoticClassifier, dict]:
    ckpt  = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = MitoticClassifier().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Tile discovery
# ---------------------------------------------------------------------------

def find_all_tiles(slide_dir: str) -> list[dict]:
    d = os.path.join(slide_dir, RAW_SUBDIR)
    if not os.path.isdir(d):
        return []
    seen = {}
    for fname in os.listdir(d):
        nums = re.findall(r'\d+', os.path.splitext(fname)[0])
        if not nums:
            continue
        tn = int(nums[-1])
        if tn not in seen:
            seen[tn] = {'tile_num': tn, 'path': os.path.join(d, fname)}
    return sorted(seen.values(), key=lambda t: t['tile_num'])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class TileImageDataset(Dataset):
    def __init__(self, tiles: list[dict], transform):
        self.tiles     = tiles
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img = Image.open(self.tiles[idx]['path']).convert('RGB')
        return self.transform(img), idx


def run_inference(
    model: MitoticClassifier,
    tiles: list[dict],
    batch_size: int,
    threshold: float,
    num_workers: int = 4,
) -> dict[int, dict]:
    """
    {tile_num: {'pred_class', 'label', 'prob', 'probs'}} for tiles given as
    [{'tile_num', 'path'}]. 'prob' is the probability of the assigned class.
    """
    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    loader = DataLoader(
        TileImageDataset(tiles, tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )
    model.eval()
    results: dict[int, dict] = {}
    with torch.inference_mode():
        for imgs, indices in loader:
            logits   = model(imgs.to(device, non_blocking=True))
            probs    = torch.softmax(logits.float(), dim=1).cpu().numpy()
            top      = np.argmax(probs, axis=1)
            top_prob = probs[np.arange(len(top)), top]
            for i, idx in enumerate(indices.tolist()):
                cls = int(top[i]) if top_prob[i] >= threshold else 0
                results[tiles[idx]['tile_num']] = {
                    'pred_class': cls,
                    'label':      CLASS_NAMES[cls],
                    'prob':       round(float(probs[i, cls]), 4),
                    'probs':      [round(float(p), 4) for p in probs[i]],
                }
    return results


# ---------------------------------------------------------------------------
# Refinement of predictions.json
# ---------------------------------------------------------------------------

def apply_mitotic_results(
    predictions: list[dict],
    inference_results: dict[int, dict],
    protected_tile_nums=frozenset(),
    insert_missing: bool = False,
) -> dict:
    """Apply the mitotic classifier to prediction records in place; returns counters."""
    stats = {'set_mitotic': 0, 'set_bizarre': 0, 'cleared_mitotic': 0,
             'new_positive': 0, 'untouched': 0, 'protected': 0}
    by_tile = {p['tile_num']: p for p in predictions if 'tile_num' in p}

    for tn, res in inference_results.items():
        if tn in protected_tile_nums:
            stats['protected'] += 1
            continue
        positive = res['pred_class'] in (1, 2)
        entry    = by_tile.get(tn)

        if entry is None:
            if positive and insert_missing:
                entry = {'tile_num': tn, 'desc': f'Tile {tn}', 'mitotic': res['label'],
                         'mitotic_prob': res['prob'], 'mitotic_model_prob': res['prob']}
                for col in OTHER_COLS:
                    entry[col] = 'None'
                    entry[f'{col}_prob'] = None
                predictions.append(entry)
                by_tile[tn] = entry
                stats['new_positive'] += 1
            continue

        currently_positive = entry.get('mitotic') in POSITIVE
        if positive:
            if not currently_positive and FALLBACK_KEY not in entry:
                entry[FALLBACK_KEY] = {key: entry.get(key)
                                       for col in OTHER_COLS for key in (col, f'{col}_prob')}
            for col in OTHER_COLS:
                entry[col] = 'None'
                entry[f'{col}_prob'] = None
            entry['mitotic']            = res['label']
            entry['mitotic_prob']       = res['prob']
            entry['mitotic_model_prob'] = res['prob']
            stats['set_mitotic' if res['pred_class'] == 1 else 'set_bizarre'] += 1
        elif currently_positive:
            fallback = entry.pop(FALLBACK_KEY, None)
            if fallback:
                entry.update(fallback)
            entry['mitotic']            = 'None'
            entry['mitotic_prob']       = res['prob']
            entry['mitotic_model_prob'] = res['prob']
            stats['cleared_mitotic'] += 1
        else:
            stats['untouched'] += 1

    predictions.sort(key=lambda p: p.get('tile_num', -1))
    return stats


def write_json_atomic(path: str, obj) -> None:
    tmp = f'{path}.tmp.{os.getpid()}'
    try:
        with open(tmp, 'w') as f:
            json.dump(obj, f, separators=(',', ':'))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model, ckpt = load_mitotic_model(args.checkpoint)
    print(
        f"Model loaded | best_auc={ckpt.get('best_auc', 'N/A')} | "
        f"class_names={ckpt.get('class_names', CLASS_NAMES)} | "
        f"threshold={args.threshold}",
        flush=True,
    )

    for slide_dir in args.slides:
        slide_dir = os.path.abspath(slide_dir)
        print(f"\nSlide: {os.path.basename(slide_dir)}", flush=True)

        pred_path = os.path.join(slide_dir, 'predictions.json')
        if not os.path.exists(pred_path):
            print("  No predictions.json, skipping", flush=True)
            continue
        with open(pred_path) as f:
            predictions = json.load(f)

        tiles = find_all_tiles(slide_dir)
        if not tiles:
            print("  No tiles found", flush=True)
            continue
        print(f"  {len(tiles)} tiles | {len(predictions)} existing predictions", flush=True)

        inference_results = run_inference(model, tiles, args.batch_size, args.threshold)
        stats = apply_mitotic_results(predictions, inference_results, insert_missing=True)
        write_json_atomic(pred_path, predictions)

        for k, v in stats.items():
            print(f"  {k:<22}: {v}", flush=True)
        print(f"  Saved -> {pred_path}", flush=True)

    print("\nDone.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--slides',     nargs='+', required=True)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--threshold',  type=float, default=0.7,
                        help='Minimum arg-max probability for a positive prediction')
    return parser.parse_args()


if __name__ == '__main__':
    main()
