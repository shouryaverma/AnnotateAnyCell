#!/usr/bin/env python3
"""
update_clusters.py - Active-learning model update behind the "Update Clusters" button.

1. Loads the base MultiHeadClassifier checkpoint (or slide's fine-tuned one if --from_finetuned)
2. Caches Virchow2 features: rotation 0 for every tile, 90/180/270 only for user-labeled tiles
3. Fine-tunes the classifier on user-labeled tiles with valid-combination regularization
4. Predicts all tiles and decodes the most probable valid label combination
5. Builds supervision codes: user labels + high-confidence pseudo-labels
6. Trains an image autoencoder on raw+nuclei tiles (ignoring unlabeled tiles in CE loss),
   and runs semi-supervised UMAP on its 256-dim latent space
7. Refines mitotic predictions with the dedicated mitotic classifier (when provided)
8. Writes predictions_<user>.json (or predictions.json) and saves the embedding to SQLite/MongoDB

Progress lines match the format parsed by annotator_server.py:
  "Progress: 42.50% | message"
"""
import os
import re
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import umap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storage import create_store
from model_train import (
    LABEL_COLS,
    MultiHeadClassifier,
    build_valid_mask_from_combos,
    invalid_combo_penalty,
)
from model_predict import (
    ROTATIONS,
    CachedInferenceDataset,
    apply_optional_mitotic_refinement,
    build_prediction_records,
    decode_valid_combinations,
    extract_virchow2_features,
    find_all_tiles,
    load_multihead_checkpoint,
    load_sem_uint8,
    pending_feature_jobs,
    predict_probabilities,
    rotation_cache_path,
    write_json_atomic,
)

LABEL_STR = {
    'mitotic':   {0: 'None', 1: 'Mitotic Figure', 2: 'Bizarre Figure'},
    'chromatin': {0: 'None', 1: 'Vesicular Chromatin'},
    'nucleoli':  {0: 'None', 1: 'Single Nucleoli', 2: 'Multiple Nucleoli'},
    'prominent': {0: 'None', 1: 'Prominent Nucleoli'},
    'shape':     {0: 'None', 1: 'Circle', 2: 'Oval', 3: 'Irregular Shape'},
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]', '_', value or 'default')


def progress(pct: float, msg: str = ''):
    print(f'Progress: {pct:.2f}% | {msg}', flush=True)


def scaled_progress(start: float, end: float, label: str):
    def report(done: int, total: int):
        progress(start + done / max(total, 1) * (end - start), f'{label} {done}/{total}')
    return report


# ---------------------------------------------------------------------------
# User labels
# ---------------------------------------------------------------------------

def load_user_labels(store, username: str, slide_name: str,
                     tiles: list[dict]) -> tuple[list[int], dict[int, dict]]:
    """Indices into `tiles` of user-labeled tiles and their label strings."""
    doc = store.get_features_assigned(username, slide_name)
    labeled_indices: list[int]       = []
    labels_by_index: dict[int, dict] = {}
    if not doc:
        return labeled_indices, labels_by_index
    tile_to_idx = {t['tile_num']: i for i, t in enumerate(tiles)}
    doc_tiles   = doc.get('tile_nums')
    samples     = doc.get('samples', [])
    for j, (lbl, prob) in enumerate(zip(doc.get('labels', []), doc.get('probabilities', []))):
        if prob != '100' or len(str(lbl)) != len(LABEL_COLS) or not str(lbl).isdigit():
            continue
        if doc_tiles:
            idx = tile_to_idx.get(doc_tiles[j]) if j < len(doc_tiles) else None
        else:
            idx = samples[j] if j < len(samples) and 0 <= samples[j] < len(tiles) else None
        if idx is None:
            continue
        labeled_indices.append(idx)
        labels_by_index[idx] = {
            col: LABEL_STR[col].get(int(str(lbl)[k]), 'None')
            for k, col in enumerate(LABEL_COLS)
        }
    return labeled_indices, labels_by_index


# ---------------------------------------------------------------------------
# Fine-tuning on labeled tiles
# ---------------------------------------------------------------------------

def finetune_heads(model: MultiHeadClassifier,
                   tiles: list[dict],
                   labeled_indices: list[int],
                   labels_by_index: dict[int, dict],
                   vocabs: dict,
                   valid_combinations: list[tuple],
                   num_epochs: int,
                   batch_size: int,
                   lr: float,
                   combo_weight: float = 0.5,
                   sem_size: int = 128,
                   progress_start: float = 8.,
                   progress_end: float = 15.) -> MultiHeadClassifier:
    if not labeled_indices:
        progress(progress_end, 'No labeled tiles - skipping fine-tune')
        return model

    n_rot   = len(ROTATIONS)
    total   = len(labeled_indices) * n_rot
    feats   = None
    sems    = torch.empty((total, 3, sem_size, sem_size), dtype=torch.uint8)
    targets = {col: torch.empty(total, dtype=torch.long) for col in LABEL_COLS}
    for a, tile_idx in enumerate(labeled_indices):
        t   = tiles[tile_idx]
        sem = load_sem_uint8(t.get('nuc_path'), sem_size)
        for r, rot in enumerate(ROTATIONS):
            row  = a * n_rot + r
            feat = torch.from_numpy(np.load(rotation_cache_path(t['feat_cache'], rot))).float()
            if feats is None:
                feats = torch.empty((total, feat.shape[0]), dtype=torch.float32)
            feats[row] = feat
            rotated    = np.rot90(sem, k=rot // 90).copy() if rot else sem
            sems[row]  = torch.from_numpy(rotated).permute(2, 0, 1)
            for col in LABEL_COLS:
                label = labels_by_index[tile_idx].get(col, 'None')
                targets[col][row] = vocabs[col].get(label, vocabs[col]['None'])

    for param in model.parameters():
        param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    opt       = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    valid_mask = build_valid_mask_from_combos(valid_combinations, vocabs)
    invalid_mask_dev = (1.0 - valid_mask).to(device)
    n_batches = max(1, (total + batch_size - 1) // batch_size)
    model.train()

    for epoch in range(num_epochs):
        perm    = torch.randperm(total)
        ep_loss = 0.0
        for i in range(0, total, batch_size):
            rows         = perm[i:i + batch_size]
            if len(rows) == 1 and total > 1:
                continue  # avoid BatchNorm error on size-1 trailing batch
            virchow_feat = feats[rows].to(device)
            sem          = sems[rows].to(device).float().div_(255.0)
            opt.zero_grad()
            outputs = model(virchow_feat, sem)
            loss    = sum(criterion(outputs[col], targets[col][rows].to(device))
                          for col in LABEL_COLS) / len(LABEL_COLS)
            if combo_weight > 0:
                loss = loss + combo_weight * invalid_combo_penalty(outputs, invalid_mask_dev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        pct = progress_start + (epoch + 1) / num_epochs * (progress_end - progress_start)
        progress(pct, f'Fine-tune epoch [{epoch+1}/{num_epochs}] loss={ep_loss / n_batches:.4f}')

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Supervision codes for the autoencoder and supervised UMAP
# ---------------------------------------------------------------------------

def build_umap_labels(labeled_indices: list[int],
                      labels_by_index: dict[int, dict],
                      decoded: dict,
                      vocabs: dict,
                      n: int,
                      top_n_pseudo: int = 50,
                      conf_threshold: float = 0.70) -> tuple[np.ndarray, set]:
    none_idx  = np.array([vocabs[col]['None'] for col in LABEL_COLS], dtype=np.int64)
    combos    = decoded['combos']
    probs     = decoded['probs']
    rows      = np.arange(n)

    is_none       = combos == none_idx[None, :]
    masked        = np.where(is_none, -1.0, probs)
    dominant_col  = np.argmax(masked, axis=1)
    dominant_cls  = combos[rows, dominant_col]
    dominant_prob = probs[rows, dominant_col]
    has_label     = ~np.all(is_none, axis=1)
    dominant_code = np.where(has_label, dominant_col * 10 + dominant_cls, -1).astype(float)

    umap_y  = np.full(n, -1.0)
    labeled = np.zeros(n, dtype=bool)
    for idx in labeled_indices:
        if idx >= n:
            continue
        labeled[idx] = True
        for h, col in enumerate(LABEL_COLS):
            cls = vocabs[col].get(labels_by_index[idx].get(col, 'None'), int(none_idx[h]))
            if cls != none_idx[h]:
                umap_y[idx] = h * 10 + cls
                break

    candidates = np.flatnonzero(~labeled & has_label & (dominant_prob >= conf_threshold))
    candidates = candidates[np.argsort(-dominant_prob[candidates])]
    per_code: dict[float, int] = {}
    pseudo_indices: set[int] = set()
    for i in candidates:
        code = dominant_code[i]
        if per_code.get(code, 0) >= top_n_pseudo:
            continue
        umap_y[i]      = code
        per_code[code] = per_code.get(code, 0) + 1
        pseudo_indices.add(int(i))
    return umap_y, pseudo_indices


# ---------------------------------------------------------------------------
# Image autoencoder
# ---------------------------------------------------------------------------

def load_ae_images(tiles: list[dict], img_size: int = 128) -> torch.Tensor:
    """All tiles as one uint8 tensor (N, 6, H, W): raw BGR + nuclei BGR."""
    data = torch.zeros((len(tiles), 6, img_size, img_size), dtype=torch.uint8)

    def load(i):
        t   = tiles[i]
        raw = cv2.imread(t['raw_path'])
        if raw is None:
            raw = np.zeros((img_size, img_size, 3), np.uint8)
        elif raw.shape[0] != img_size or raw.shape[1] != img_size:
            raw = cv2.resize(raw, (img_size, img_size))
        nuc = load_sem_uint8(t.get('nuc_path'), img_size)
        data[i] = torch.from_numpy(np.concatenate([raw, nuc], axis=2)).permute(2, 0, 1)

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
        list(pool.map(load, range(len(tiles))))
    return data


class ImageAutoencoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, n_codes: int, img_size: int = 128):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32,  3, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32,  64,  3, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        with torch.no_grad():
            dummy               = torch.zeros(1, in_channels, img_size, img_size)
            enc_out             = self.enc_conv(dummy)
            self._spatial_shape = enc_out.shape[1:]
            flat_size           = enc_out.flatten(1).shape[1]

        self.enc_fc = nn.Sequential(
            nn.Linear(flat_size, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, latent_dim),
        )
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.GELU(),
            nn.Linear(512, flat_size),  nn.ReLU(),
        )
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,  in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(latent_dim, max(1, n_codes))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_fc(self.enc_conv(x).flatten(1))

    def forward(self, x: torch.Tensor):
        latent = self.encode(x)
        recon  = self.dec_conv(self.dec_fc(latent).view(-1, *self._spatial_shape))
        return latent, recon, self.classifier(latent)


def train_image_autoencoder(tiles: list[dict],
                            ae_labels: np.ndarray,
                            n_codes: int,
                            latent_dim: int   = 256,
                            num_epochs: int   = 60,
                            batch_size: int   = 64,
                            lr: float         = 1e-3,
                            img_size: int     = 128,
                            progress_start: float = 23.,
                            progress_end: float   = 85.) -> np.ndarray:
    """
    Train the autoencoder on all tiles and return the latent vector of every tile.
    Unlabeled tiles carry label -100 so they contribute to reconstruction loss
    without being penalized by the classification head.
    """
    progress(progress_start, f'Loading {len(tiles)} tiles into memory')
    images     = load_ae_images(tiles, img_size)
    labels     = torch.from_numpy(ae_labels).long()
    n          = len(tiles)
    model      = ImageAutoencoder(6, latent_dim, max(1, n_codes), img_size).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    recon_crit = nn.MSELoss()
    cls_crit   = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            rows = perm[i:i + batch_size]
            if len(rows) == 1 and n > 1:
                continue
            imgs = images[rows].to(device, non_blocking=True).float().div_(255.0)
            lbls = labels[rows].to(device)
            _, recon, logits = model(imgs)
            recon_loss = recon_crit(recon, imgs)
            if n_codes > 0 and (lbls != -100).any():
                cls_loss = cls_crit(logits, lbls)
                loss = recon_loss * 0.8 + cls_loss * 0.2
            else:
                loss = recon_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pct = progress_start + (epoch + 1) / num_epochs * (progress_end - progress_start)
        progress(pct, f'Image AE epoch [{epoch+1}/{num_epochs}]')

    model.eval()
    latents = []
    with torch.inference_mode():
        for i in range(0, n, 512):
            imgs = images[i:i + 512].to(device).float().div_(255.0)
            latents.append(model.encode(imgs).cpu().numpy())
    del model, images
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(latents, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    store = create_store(
        sqlite_path=args.sqlite_path,
        mongo_uri=args.mongo_uri,
        mongo_username=args.mongo_username,
        mongo_password=args.mongo_password,
        mongo_host=args.mongo_host,
        mongo_db=args.mongo_db,
        backend=args.storage,
    )

    try:
        tiles = find_all_tiles(args.slide_path)
        if not tiles:
            raise SystemExit(f'No tiles found under {args.slide_path}')
        labeled_indices, labels_by_index = load_user_labels(store, args.username, args.slide_name, tiles)
        progress(1., f'Found {len(tiles)} tiles, {len(labeled_indices)} labeled ({store.backend})')

        user_tag = safe_name(args.username)
        finetuned_ckpt = os.path.join(
            args.slide_path,
            'finetuned_classifier.pth' if args.shared_outputs else f'finetuned_classifier_{user_tag}.pth',
        )
        load_path = finetuned_ckpt if (args.from_finetuned and os.path.exists(finetuned_ckpt)) else args.checkpoint
        progress(2., f'Loading checkpoint: {os.path.basename(load_path)}')
        model, vocabs, head_sizes, valid_combinations = load_multihead_checkpoint(load_path)

        # --- Step 1: Virchow2 feature cache ---
        jobs = pending_feature_jobs(tiles, rotated_indices=labeled_indices)
        if jobs:
            progress(3., f'Extracting Virchow2 features for {len(jobs)} tile/rotation pairs')
            extract_virchow2_features(jobs, args.batch_size,
                                      on_progress=scaled_progress(3., 8., 'Extraction'))
        else:
            progress(8., 'All feature caches exist, skipping extraction')

        # --- Step 2: Fine-tune multihead on user-labeled tiles ---
        model = finetune_heads(
            model, tiles, labeled_indices, labels_by_index, vocabs, valid_combinations,
            args.num_epochs, args.batch_size, args.lr,
            progress_start=8., progress_end=15.,
        )
        if labeled_indices:
            tmp = f'{finetuned_ckpt}.tmp.{os.getpid()}'
            torch.save({
                'model_state':        model.state_dict(),
                'vocabs':             vocabs,
                'head_sizes':         head_sizes,
                'valid_combinations': valid_combinations,
            }, tmp)
            os.replace(tmp, finetuned_ckpt)
            progress(15.5, f'Saved finetuned checkpoint to {os.path.basename(finetuned_ckpt)}')

        # --- Step 3: Multihead inference + valid-combination decoding ---
        prob_matrix = predict_probabilities(model, CachedInferenceDataset(tiles), args.batch_size,
                                            on_progress=scaled_progress(16., 21., 'Inference batch'))
        decoded     = decode_valid_combinations(prob_matrix, vocabs, valid_combinations)
        predictions = build_prediction_records(tiles, decoded, vocabs)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # --- Step 4: Supervision codes ---
        umap_y, pseudo_indices = build_umap_labels(
            labeled_indices, labels_by_index, decoded, vocabs,
            n=len(tiles), top_n_pseudo=args.top_n_pseudo, conf_threshold=args.conf_threshold)
        n_coded = int((umap_y >= 0).sum())
        progress(22., f'Supervision: {n_coded - len(pseudo_indices)} labeled + '
                      f'{len(pseudo_indices)} pseudo + {len(tiles) - n_coded} unlabeled')

        unique_codes = np.unique(umap_y[umap_y >= 0])
        ae_labels    = np.full(len(tiles), -100, dtype=np.int64)
        coded        = umap_y >= 0
        if coded.any():
            ae_labels[coded] = np.searchsorted(unique_codes, umap_y[coded])

        # --- Step 5: Autoencoder ---
        latent = train_image_autoencoder(
            tiles, ae_labels, len(unique_codes),
            latent_dim=256, num_epochs=args.ae_epochs, batch_size=64, lr=1e-3,
            progress_start=23., progress_end=85.)

        # --- Step 6: UMAP on AE latent ---
        progress(86., 'Running UMAP on AE latent')
        if len(tiles) < 4:
            embeddings = np.zeros((len(tiles), 2), dtype=np.float64)
        else:
            umap_kwargs = dict(
                n_neighbors=max(2, min(15, len(tiles) - 1)),
                min_dist=0.8, spread=1.0,
                n_components=2, n_jobs=1, random_state=42,
            )
            if coded.any() and len(unique_codes) >= 2:
                umap_kwargs.update(target_metric='categorical', target_weight=0.2)
                y_arg = umap_y
            else:
                y_arg = None
            embeddings = umap.UMAP(**umap_kwargs).fit_transform(latent, y=y_arg)
            embeddings = np.nan_to_num(np.asarray(embeddings, dtype=np.float64))
        progress(90., 'UMAP complete')

        # --- Step 7: Mitotic classifier refinement (user-labeled tiles are left alone) ---
        if args.mitotic_checkpoint and os.path.exists(args.mitotic_checkpoint):
            progress(91., f'Applying mitotic classifier from {os.path.basename(args.mitotic_checkpoint)}')
            protected = {tiles[idx]['tile_num'] for idx in labeled_indices}
            stats = apply_optional_mitotic_refinement(
                predictions, tiles, args.mitotic_checkpoint,
                args.batch_size, args.mitotic_threshold,
                protected_tile_nums=protected,
            )
            if stats:
                progress(97., 'Mitotic filter done - '
                              f'set_mitotic={stats.get("set_mitotic", 0)} '
                              f'set_bizarre={stats.get("set_bizarre", 0)} '
                              f'cleared={stats.get("cleared_mitotic", 0)} '
                              f'gt_protected={stats.get("protected", 0)}')
            else:
                progress(97., 'Mitotic module unavailable - skipping mitotic filter')
        else:
            progress(97., 'No mitotic checkpoint provided - skipping mitotic filter')

        # --- Step 8: Persist ---
        pred_name = 'predictions.json' if args.shared_outputs else f'predictions_{user_tag}.json'
        pred_path = os.path.join(args.slide_path, pred_name)
        write_json_atomic(pred_path, predictions)
        # Also write shared predictions.json if none exists yet so other views have a baseline
        shared_pred_path = os.path.join(args.slide_path, 'predictions.json')
        if not os.path.exists(shared_pred_path):
            write_json_atomic(shared_pred_path, predictions)
        progress(98., f'Wrote {len(predictions)} predictions to {pred_name}')

        store.save_umap_embeddings(
            args.username, args.slide_name,
            [t['tile_num'] for t in tiles],
            embeddings.tolist(),
        )
        progress(100., 'Done')
        print('completed', flush=True)
    finally:
        store.close()


def parse_args():
    env = os.environ.get
    p = argparse.ArgumentParser()
    p.add_argument('--slide_path',     required=True)
    p.add_argument('--slide_name',     required=True)
    p.add_argument('--checkpoint',     default=os.path.join('weights', 'multihead_classifier.pth'))
    p.add_argument('--storage',        default=env('STORAGE_BACKEND', 'auto'),
                   choices=['auto', 'sqlite', 'mongo'])
    p.add_argument('--sqlite_path',    default=env('SQLITE_PATH', 'data/annotator.db'))
    p.add_argument('--mongo_uri',      default=env('MONGO_URI', ''))
    p.add_argument('--mongo_username', default=env('MONGO_USERNAME', ''))
    p.add_argument('--mongo_password', default=env('MONGO_PASSWORD', ''))
    p.add_argument('--mongo_host',     default=env('MONGO_HOST', ''))
    p.add_argument('--mongo_db',       default=env('MONGO_DB', 'annotator_db'))
    p.add_argument('--username',       required=True)
    p.add_argument('--num_epochs',     type=int,   default=15)
    p.add_argument('--batch_size',     type=int,   default=256)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--top_n_pseudo',   type=int,   default=50)
    p.add_argument('--conf_threshold', type=float, default=0.70)
    p.add_argument('--ae_epochs',      type=int,   default=60)
    p.add_argument('--from_finetuned', action='store_true',
                   help='Continue fine-tuning from previous finetuned checkpoint instead of base')
    p.add_argument('--shared_outputs', action='store_true',
                   help='Write shared predictions.json / finetuned_classifier.pth instead of per-user files')
    p.add_argument('--mitotic_checkpoint', default='')
    p.add_argument('--mitotic_threshold',  type=float, default=0.5)
    return p.parse_args()


if __name__ == '__main__':
    main()
