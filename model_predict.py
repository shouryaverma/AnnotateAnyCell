#!/usr/bin/env python3
"""
model_predict.py - MultiHeadClassifier inference over every tile of each slide.

Writes <slide>/predictions.json for the annotator, and hosts the inference
building blocks shared with update_clusters.py and preprocess.py:
  * tile discovery
  * Virchow2 feature caching
  * batched head probabilities
  * valid-combination decoding (with non-mitotic fallback)
  * optional dedicated mitotic refinement via mitotic_predict.py
"""
import os
import re
import json
import argparse
import contextlib

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from model_train import MultiHeadClassifier, LABEL_COLS, Virchow2Encoder

try:
    from mitotic_predict import (
        apply_mitotic_results,
        load_mitotic_model,
        run_inference as run_mitotic_inference,
    )
    HAS_MITOTIC_MODULE = True
except ImportError:
    HAS_MITOTIC_MODULE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RAW_SUBDIR   = os.path.join('tiles1', 'nonbox')
NUC_SUBDIR   = os.path.join('tiles1', 'nuclei')
CACHE_SUBDIR = os.path.join('tiles1', 'virchow2_cache')
ROTATIONS    = (0, 90, 180, 270)
NUM_WORKERS  = min(4, os.cpu_count() or 1)
FALLBACK_KEY = 'non_mitotic_fallback'


def index_tiles(directory: str) -> dict[int, str]:
    """{tile_num: path} for every file in `directory` whose stem ends in a number."""
    tiles: dict[int, str] = {}
    if not os.path.isdir(directory):
        return tiles
    with os.scandir(directory) as it:
        for entry in it:
            nums = re.findall(r'\d+', os.path.splitext(entry.name)[0])
            if nums:
                tiles.setdefault(int(nums[-1]), entry.path)
    return tiles


def find_all_tiles(slide_dir: str) -> list[dict]:
    """All tiles of a slide sorted by tile number, with nuclei image and feature cache paths."""
    raw = index_tiles(os.path.join(slide_dir, RAW_SUBDIR))
    if not raw:
        return []
    nuc = index_tiles(os.path.join(slide_dir, NUC_SUBDIR))
    cache_dir = os.path.join(slide_dir, CACHE_SUBDIR)
    os.makedirs(cache_dir, exist_ok=True)
    return [{
        'tile_num':   tn,
        'desc':       f'Tile {tn}',
        'raw_path':   raw[tn],
        'nuc_path':   nuc.get(tn),
        'feat_cache': os.path.join(cache_dir, f'{tn}.npy'),
    } for tn in sorted(raw)]


def rotation_cache_path(feat_cache: str, rot: int) -> str:
    return feat_cache if rot == 0 else f'{os.path.splitext(feat_cache)[0]}_r{rot}.npy'


def pending_feature_jobs(tiles: list[dict], rotated_indices=()) -> list[tuple]:
    """
    (raw_path, rotation, cache_path) for every missing feature file. Rotation 0
    is required for all tiles; 90/180/270 only for `rotated_indices`.
    """
    rotated = set(rotated_indices)
    jobs = []
    for i, t in enumerate(tiles):
        for rot in (ROTATIONS if i in rotated else (0,)):
            cache = rotation_cache_path(t['feat_cache'], rot)
            if not os.path.exists(cache):
                jobs.append((t['raw_path'], rot, cache))
    return jobs


class _FeatureJobDataset(Dataset):
    def __init__(self, jobs, transform):
        self.jobs = jobs
        self.transform = transform

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, idx):
        raw_path, rot, _ = self.jobs[idx]
        tensor = self.transform(Image.open(raw_path).convert('RGB'))
        if rot:
            tensor = torch.rot90(tensor, rot // 90, dims=[1, 2])
        return tensor, idx


def save_npy_atomic(path: str, array: np.ndarray) -> None:
    tmp = f'{path}.tmp.{os.getpid()}.npy'
    np.save(tmp, array)
    os.replace(tmp, path)


def extract_virchow2_features(jobs: list[tuple], batch_size: int, on_progress=None) -> None:
    """Run Virchow2 over the pending jobs and write each feature vector to its cache file."""
    if not jobs:
        return
    encoder = Virchow2Encoder().to(device)
    encoder.eval()
    transform = create_transform(**resolve_data_config(encoder.vit.pretrained_cfg, model=encoder.vit))
    loader = DataLoader(_FeatureJobDataset(jobs, transform), batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    done = 0
    with torch.inference_mode():
        for batch, indices in loader:
            amp = (torch.autocast(device_type='cuda', dtype=torch.float16)
                   if device.type == 'cuda' else contextlib.nullcontext())
            with amp:
                feats = encoder(batch.to(device, non_blocking=True))
            feats = feats.float().cpu().numpy()
            for feat, idx in zip(feats, indices.tolist()):
                save_npy_atomic(jobs[idx][2], feat)
            done += len(feats)
            if on_progress:
                on_progress(done, len(jobs))
    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def load_sem_uint8(nuc_path: str | None, sem_size: int) -> np.ndarray:
    """Nuclei tile as (sem_size, sem_size, 3) uint8; zeros when the tile has no nuclei image."""
    sem = cv2.imread(nuc_path) if nuc_path else None
    if sem is None:
        return np.zeros((sem_size, sem_size, 3), dtype=np.uint8)
    if sem.shape[0] != sem_size or sem.shape[1] != sem_size:
        sem = cv2.resize(sem, (sem_size, sem_size))
    return sem


class CachedInferenceDataset(Dataset):
    """Returns (virchow_feat, sem_tensor, idx) from disk cache."""
    def __init__(self, tiles: list[dict], sem_size: int = 128):
        self.tiles    = tiles
        self.sem_size = sem_size

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        t            = self.tiles[idx]
        virchow_feat = torch.from_numpy(np.load(t['feat_cache'])).float()
        sem          = load_sem_uint8(t.get('nuc_path'), self.sem_size).astype(np.float32) / 255.0
        return virchow_feat, torch.from_numpy(sem).permute(2, 0, 1), idx


def load_multihead_checkpoint(path: str):
    ckpt  = torch.load(path, map_location='cpu', weights_only=True)
    model = MultiHeadClassifier(ckpt['head_sizes']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    valid_combinations = [tuple(int(v) for v in combo) for combo in ckpt['valid_combinations']]
    return model, ckpt['vocabs'], ckpt['head_sizes'], valid_combinations


def predict_probabilities(model: MultiHeadClassifier, dataset: CachedInferenceDataset,
                          batch_size: int, on_progress=None) -> np.ndarray:
    """Softmax of every head, concatenated in LABEL_COLS order: (N, sum(head sizes))."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda')
    model.eval()
    out = None
    with torch.inference_mode():
        for i, (virchow_feat, sem, indices) in enumerate(loader):
            outputs = model(virchow_feat.to(device, non_blocking=True),
                            sem.to(device, non_blocking=True))
            probs = torch.cat([torch.softmax(outputs[col], dim=1) for col in LABEL_COLS], dim=1)
            probs = probs.float().cpu().numpy()
            if out is None:
                out = np.zeros((len(dataset), probs.shape[1]), dtype=np.float32)
            out[indices.numpy()] = probs
            if on_progress:
                on_progress(i + 1, len(loader))
    return out if out is not None else np.zeros((0, 0), dtype=np.float32)


def head_offsets(vocabs: dict) -> tuple[list[int], list[int]]:
    sizes = [len(vocabs[col]) for col in LABEL_COLS]
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    return sizes, offsets


def decode_valid_combinations(prob_matrix: np.ndarray, vocabs: dict,
                              valid_combinations: list[tuple]) -> dict:
    _, offsets = head_offsets(vocabs)
    combos = np.array(sorted(valid_combinations), dtype=np.int64)
    n = prob_matrix.shape[0]
    rows = np.arange(n)

    joint = np.ones((n, combos.shape[0]), dtype=np.float64)
    for h in range(len(LABEL_COLS)):
        joint *= prob_matrix[:, offsets[h] + combos[:, h]]

    def pick(mask):
        candidates = np.flatnonzero(mask)
        best = combos[candidates[np.argmax(joint[:, candidates], axis=1)]]
        marg = np.stack([prob_matrix[rows, offsets[h] + best[:, h]]
                         for h in range(len(LABEL_COLS))], axis=1)
        return best, marg

    mitotic_h = LABEL_COLS.index('mitotic')
    best, marg = pick(np.ones(combos.shape[0], dtype=bool))
    fb_best, fb_marg = pick(combos[:, mitotic_h] == vocabs['mitotic']['None'])
    return {'combos': best, 'probs': marg, 'fallback_combos': fb_best, 'fallback_probs': fb_marg}


def build_prediction_records(tiles: list[dict], decoded: dict, vocabs: dict) -> list[dict]:
    idx_to_label = {col: {v: k for k, v in vocabs[col].items()} for col in LABEL_COLS}
    mitotic_h    = LABEL_COLS.index('mitotic')
    mitotic_none = vocabs['mitotic']['None']
    records = []
    for i, t in enumerate(tiles):
        rec = {'tile_num': t['tile_num'], 'desc': f'Tile {t["tile_num"]}'}
        for h, col in enumerate(LABEL_COLS):
            rec[col]           = idx_to_label[col].get(int(decoded['combos'][i, h]), 'None')
            rec[f'{col}_prob'] = round(float(decoded['probs'][i, h]), 4)
        if decoded['combos'][i, mitotic_h] != mitotic_none:
            fallback = {}
            for h, col in enumerate(LABEL_COLS):
                if h == mitotic_h:
                    continue
                fallback[col] = idx_to_label[col].get(int(decoded['fallback_combos'][i, h]), 'None')
                fallback[f'{col}_prob'] = round(float(decoded['fallback_probs'][i, h]), 4)
            rec[FALLBACK_KEY] = fallback
        records.append(rec)
    return records


def apply_optional_mitotic_refinement(predictions: list[dict], tiles: list[dict],
                                      mitotic_checkpoint: str, batch_size: int,
                                      threshold: float, protected_tile_nums=None) -> dict | None:
    if not mitotic_checkpoint or not os.path.exists(mitotic_checkpoint):
        return None
    if not HAS_MITOTIC_MODULE:
        print("Warning: mitotic_predict.py not available - skipping mitotic refinement.", flush=True)
        return None
    mitotic_model, _ = load_mitotic_model(mitotic_checkpoint)
    mitotic_results = run_mitotic_inference(
        mitotic_model,
        [{'tile_num': t['tile_num'], 'path': t['raw_path']} for t in tiles],
        batch_size,
        threshold,
    )
    del mitotic_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return apply_mitotic_results(
        predictions, mitotic_results,
        protected_tile_nums=protected_tile_nums or set(),
    )


def write_json_atomic(path: str, obj) -> None:
    tmp = f'{path}.tmp.{os.getpid()}'
    try:
        with open(tmp, 'w') as f:
            json.dump(obj, f, separators=(',', ':'))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def predict_slide(slide_dir: str, checkpoint: str, batch_size: int = 256,
                  mitotic_checkpoint: str = '', mitotic_threshold: float = 0.5,
                  on_extract_progress=None, on_infer_progress=None) -> int:
    """Run full inference for one slide directory and write predictions.json."""
    tiles = find_all_tiles(slide_dir)
    if not tiles:
        return 0
    jobs = pending_feature_jobs(tiles)
    if jobs:
        extract_virchow2_features(jobs, batch_size, on_progress=on_extract_progress)
    model, vocabs, _, valid_combinations = load_multihead_checkpoint(checkpoint)
    prob_matrix = predict_probabilities(
        model, CachedInferenceDataset(tiles), batch_size, on_progress=on_infer_progress
    )
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    decoded = decode_valid_combinations(prob_matrix, vocabs, valid_combinations)
    results = build_prediction_records(tiles, decoded, vocabs)
    apply_optional_mitotic_refinement(
        results, tiles, mitotic_checkpoint, batch_size, mitotic_threshold
    )
    out_path = os.path.join(slide_dir, 'predictions.json')
    write_json_atomic(out_path, results)
    return len(results)


def main():
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model, vocabs, _, valid_combinations = load_multihead_checkpoint(args.checkpoint)
    print("Model loaded.", flush=True)

    slides = []
    jobs   = []
    for slide_dir in args.slides:
        slide_dir = os.path.abspath(slide_dir)
        tiles     = find_all_tiles(slide_dir)
        print(f"Slide: {os.path.basename(slide_dir)} | {len(tiles)} tiles", flush=True)
        if tiles:
            slides.append((slide_dir, tiles))
            jobs.extend(pending_feature_jobs(tiles))

    if jobs:
        print(f"Extracting Virchow2 features for {len(jobs)} tiles...", flush=True)
        extract_virchow2_features(jobs, args.batch_size)

    for slide_dir, tiles in slides:
        prob_matrix = predict_probabilities(model, CachedInferenceDataset(tiles), args.batch_size)
        decoded     = decode_valid_combinations(prob_matrix, vocabs, valid_combinations)
        results     = build_prediction_records(tiles, decoded, vocabs)
        stats = apply_optional_mitotic_refinement(
            results, tiles, args.mitotic_checkpoint, args.batch_size, args.mitotic_threshold
        )
        if stats:
            print(f"  Mitotic refinement: {stats}", flush=True)
        out_path = os.path.join(slide_dir, 'predictions.json')
        write_json_atomic(out_path, results)
        print(f"  Saved {len(results)} predictions -> {out_path}", flush=True)

    print("\nDone.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions.json from trained model')
    parser.add_argument('--checkpoint',         required=True)
    parser.add_argument('--slides',             nargs='+', required=True)
    parser.add_argument('--batch_size',         type=int,   default=256)
    parser.add_argument('--mitotic_checkpoint', default='')
    parser.add_argument('--mitotic_threshold',  type=float, default=0.5)
    return parser.parse_args()


if __name__ == '__main__':
    main()
