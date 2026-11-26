#!/usr/bin/env python3
"""
model_train.py - Train the multi-head histopathology tile classifier on Virchow2
features + nuclei/cell mask geometry.

Usage:
    python model_train.py \
        --slide_dirs ./slides/ExampleSlide1 ./slides/ExampleSlide2 \
        --checkpoint_out multihead_classifier.pth
"""
import os
import re
import cv2
import glob
import shutil
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_COLS = ['mitotic', 'chromatin', 'nucleoli', 'prominent', 'shape']


def find_tile_file(slide_dir: str, tile_num: int, subdir: str) -> str | None:
    d = os.path.join(slide_dir, subdir)
    if not os.path.isdir(d):
        return None
    for fname in os.listdir(d):
        nums = re.findall(r'\d+', os.path.splitext(fname)[0])
        if nums and int(nums[-1]) == tile_num:
            return os.path.join(d, fname)
    return None


def load_annotation_csvs(slide_dirs):
    frames = []
    for slide_dir in slide_dirs:
        csvs = sorted(glob.glob(os.path.join(slide_dir, 'annotation_data_*.csv')))
        if not csvs:
            print(f"No annotation CSV in {slide_dir}", flush=True)
            continue
        df = pd.read_csv(csvs[-1])
        df.columns = [c.strip().lower() for c in df.columns]
        df['slide_dir'] = slide_dir
        tile_col = df['desc'].astype(str).str.extract(r'(\d+)')[0]
        df['tile_num'] = pd.to_numeric(tile_col, errors='coerce')
        df = df.dropna(subset=['tile_num'])
        df['tile_num'] = df['tile_num'].astype(int)
        frames.append(df)
        print(f"Loaded {len(df)} rows from {csvs[-1]}", flush=True)
    if not frames:
        raise RuntimeError("No annotation CSVs found in any of the provided --slide_dirs.")
    return pd.concat(frames, ignore_index=True)


def build_vocabs(df):
    vocabs = {}
    for col in LABEL_COLS:
        vals = df[col].fillna('None').astype(str).str.strip().replace({'': 'None'})
        unique = sorted(vals.unique())
        if 'None' not in unique:
            unique = sorted(['None'] + unique)
        vocabs[col] = {v: i for i, v in enumerate(unique)}
        print(f"  {col}: {vocabs[col]}", flush=True)
    return vocabs


def build_records(df, vocabs):
    RAW_SUBDIR   = os.path.join('tiles1', 'nonbox')
    NUC_SUBDIR   = os.path.join('tiles1', 'nuclei')
    CACHE_SUBDIR = os.path.join('tiles1', 'virchow2_cache')
    records = []
    missing = 0
    for _, row in df.iterrows():
        slide_dir = row['slide_dir']
        tile_num  = int(row['tile_num'])
        raw_path  = find_tile_file(slide_dir, tile_num, RAW_SUBDIR)
        sem_path  = find_tile_file(slide_dir, tile_num, NUC_SUBDIR)
        if raw_path is None:
            missing += 1
            continue
        cache_dir = os.path.join(slide_dir, CACHE_SUBDIR)
        os.makedirs(cache_dir, exist_ok=True)
        labels = {}
        for col in LABEL_COLS:
            val = str(row[col]).strip() if pd.notna(row[col]) else 'None'
            labels[col] = vocabs[col].get(val, vocabs[col]['None'])
        records.append({
            'raw':        raw_path,
            'sem':        sem_path,
            'labels':     labels,
            'feat_cache': os.path.join(cache_dir, f'{tile_num}.npy'),
        })
    print(f"Records: {len(records)} valid, {missing} missing tiles", flush=True)
    return records


def build_valid_combinations(records, vocabs):
    none_indices   = {col: vocabs[col]['None'] for col in LABEL_COLS}
    mitotic_none   = none_indices['mitotic']
    nucleoli_none  = none_indices['nucleoli']
    prominent_none = none_indices['prominent']

    valid = set()

    for label, idx in vocabs['mitotic'].items():
        if label != 'None':
            combo = tuple(
                idx if col == 'mitotic' else none_indices[col]
                for col in LABEL_COLS
            )
            valid.add(combo)

    shape_vals    = [idx for lbl, idx in vocabs['shape'].items() if lbl != 'None']
    chromatin_all = list(vocabs['chromatin'].values())
    nucleoli_all  = list(vocabs['nucleoli'].values())
    prominent_all = list(vocabs['prominent'].values())

    for shape_idx in shape_vals:
        for chrom_idx in chromatin_all:
            for nuc_idx in nucleoli_all:
                for prom_idx in prominent_all:
                    if prom_idx != prominent_none and nuc_idx == nucleoli_none:
                        continue
                    combo = []
                    for col in LABEL_COLS:
                        if col == 'mitotic':
                            combo.append(mitotic_none)
                        elif col == 'chromatin':
                            combo.append(chrom_idx)
                        elif col == 'nucleoli':
                            combo.append(nuc_idx)
                        elif col == 'prominent':
                            combo.append(prom_idx)
                        elif col == 'shape':
                            combo.append(shape_idx)
                    valid.add(tuple(combo))

    sizes = [len(vocabs[col]) for col in LABEL_COLS]
    mask  = torch.zeros(sizes, dtype=torch.float32)
    for combo in valid:
        mask[combo] = 1.0
    return mask, valid


def build_valid_mask_from_combos(valid_combinations, vocabs) -> torch.Tensor:
    sizes = [len(vocabs[col]) for col in LABEL_COLS]
    mask = torch.zeros(sizes, dtype=torch.float32)
    for combo in valid_combinations:
        mask[tuple(int(v) for v in combo)] = 1.0
    return mask


def compute_head_weights(records, vocabs):
    weights = {}
    for col in LABEL_COLS:
        y = np.array([r['labels'][col] for r in records])
        present = np.unique(y)
        n_classes = len(vocabs[col])
        w = np.ones(n_classes, dtype=np.float32)
        if len(present) > 0:
            cw = compute_class_weight('balanced', classes=present, y=y)
            for cls_idx, cls_w in zip(present, cw):
                w[int(cls_idx)] = float(cls_w)
        weights[col] = torch.tensor(w, dtype=torch.float32).to(device)
    return weights


def _create_virchow2_vit(pretrained: bool):
    try:
        return timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=pretrained,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
    except Exception as e:
        msg = str(e)
        if any(tok in msg for tok in ('401', '403', 'Gated', 'gated', 'Unauthorized', 'token')):
            raise RuntimeError(
                "Could not download gated foundation model 'paige-ai/Virchow2' from Hugging Face.\n"
                "1. Request access at: https://huggingface.co/paige-ai/Virchow2\n"
                "2. Authenticate via:  huggingface-cli login   (or export HF_TOKEN=hf_...)"
            ) from e
        raise


def build_virchow2_transform():
    tmp = _create_virchow2_vit(pretrained=False)
    t = create_transform(**resolve_data_config(tmp.pretrained_cfg, model=tmp))
    del tmp
    return t


class Virchow2Encoder(nn.Module):
    """Used only during feature extraction. Not stored in the training checkpoint."""
    def __init__(self):
        super().__init__()
        self.vit = _create_virchow2_vit(pretrained=True)
        for p in self.vit.parameters():
            p.requires_grad = False

    def forward(self, pixel_values):
        features     = self.vit.forward_features(pixel_values)
        num_prefix   = self.vit.num_prefix_tokens
        class_token  = features[:, 0]
        patch_tokens = features[:, num_prefix:]
        return torch.cat([class_token, patch_tokens.mean(dim=1)], dim=1)


def extract_and_cache_virchow2_features(records, transform, batch_size=32):
    ROTATIONS = [0, 90, 180, 270]
    needed = []
    for r in records:
        base = os.path.splitext(r['feat_cache'])[0]
        for rot in ROTATIONS:
            cache = r['feat_cache'] if rot == 0 else f'{base}_r{rot}.npy'
            if not os.path.exists(cache):
                needed.append((r['raw'], rot, cache))

    if not needed:
        print("All Virchow2 feature caches exist, skipping extraction.", flush=True)
        return

    print(f"Extracting Virchow2 features for {len(needed)} tile/rotation pairs...", flush=True)
    encoder = Virchow2Encoder().to(device)
    encoder.eval()

    by_rot = defaultdict(list)
    for path, rot, cache in needed:
        by_rot[rot].append((path, cache))

    for rot, items in by_rot.items():
        paths  = [p for p, _ in items]
        caches = [c for _, c in items]

        class _RotatedImageDataset(Dataset):
            def __init__(self, paths, transform, rot):
                self.paths = paths; self.transform = transform; self.rot = rot
            def __len__(self): return len(self.paths)
            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert('RGB')
                if self.rot != 0:
                    img = img.rotate(self.rot, expand=False)
                return self.transform(img)

        loader = DataLoader(_RotatedImageDataset(paths, transform, rot),
                            batch_size=batch_size, shuffle=False,
                            num_workers=min(4, os.cpu_count() or 1),
                            pin_memory=device.type == 'cuda')
        idx = 0
        with torch.inference_mode():
            for batch in loader:
                feats = encoder(batch.to(device)).cpu().numpy()
                for feat in feats:
                    np.save(caches[idx], feat)
                    idx += 1

    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print("Feature extraction complete.", flush=True)


class CachedTileDataset(Dataset):
    def __init__(self, records, sem_size=128, augment=True):
        self.records   = records
        self.sem_size  = sem_size
        self.rotations = [0, 90, 180, 270] if augment else [0]

    def __len__(self):
        return len(self.records) * len(self.rotations)

    def __getitem__(self, idx):
        rec_idx = idx // len(self.rotations)
        rot     = self.rotations[idx % len(self.rotations)]
        rec     = self.records[rec_idx]

        base  = os.path.splitext(rec['feat_cache'])[0]
        cache = rec['feat_cache'] if rot == 0 else f'{base}_r{rot}.npy'
        virchow_feat = torch.from_numpy(np.load(cache)).float()

        sem_path = rec.get('sem')
        if sem_path and os.path.exists(sem_path):
            sem = cv2.imread(sem_path)
            if sem is None:
                sem = np.zeros((self.sem_size, self.sem_size, 3), dtype=np.float32)
            else:
                sem = cv2.resize(sem, (self.sem_size, self.sem_size)).astype(np.float32) / 255.0
        else:
            sem = np.zeros((self.sem_size, self.sem_size, 3), dtype=np.float32)

        if rot != 0:
            sem = np.rot90(sem, k=rot // 90).copy()

        sem_tensor = torch.from_numpy(sem).permute(2, 0, 1).float()
        labels     = {col: torch.tensor(rec['labels'][col], dtype=torch.long) for col in LABEL_COLS}
        return virchow_feat, sem_tensor, labels


class SemanticEncoder(nn.Module):
    """Lightweight CNN for the nuclei/cell mask (256-dim output)."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16, 512),
            nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.GELU(),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class MultiHeadClassifier(nn.Module):
    """
    Accepts pre-extracted Virchow2 features (2560-dim) + live SemanticEncoder (256-dim).
    Architecture is 100% compatible with existing multihead_classifier.pth checkpoints.
    """
    def __init__(self, head_sizes):
        super().__init__()
        self.sem_encoder = SemanticEncoder()

        self.shared_fc = nn.Sequential(
            nn.Linear(2560 + 256, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.GELU(),
        )

        self.heads = nn.ModuleDict()
        for col, n_classes in head_sizes.items():
            if col == 'shape':
                self.heads[col] = nn.Sequential(
                    nn.Linear(256 + 256, 256), nn.GELU(), nn.Dropout(0.25),
                    nn.Linear(256, 128), nn.GELU(),
                    nn.Linear(128, n_classes),
                )
            else:
                self.heads[col] = nn.Sequential(
                    nn.Linear(256, 128), nn.GELU(),
                    nn.Linear(128, n_classes),
                )

    def forward(self, virchow_feat, sem_images):
        sem_feat = self.sem_encoder(sem_images)
        shared   = self.shared_fc(torch.cat([virchow_feat, sem_feat], dim=1))
        outputs  = {}
        for col, head in self.heads.items():
            if col == 'shape':
                outputs[col] = head(torch.cat([shared, sem_feat], dim=1))
            else:
                outputs[col] = head(shared)
        return outputs

    def extract_features(self, virchow_feat: torch.Tensor,
                         sem_images: torch.Tensor) -> torch.Tensor:
        sem_feat = self.sem_encoder(sem_images)
        return torch.cat([virchow_feat, sem_feat], dim=1)

    def forward_with_features(self, virchow_feat: torch.Tensor,
                              sem_images: torch.Tensor):
        sem_feat = self.sem_encoder(sem_images)
        shared   = self.shared_fc(torch.cat([virchow_feat, sem_feat], dim=1))
        outputs  = {}
        for col, head in self.heads.items():
            if col == 'shape':
                outputs[col] = head(torch.cat([shared, sem_feat], dim=1))
            else:
                outputs[col] = head(shared)
        return outputs, torch.cat([virchow_feat, sem_feat], dim=1)


def invalid_combo_penalty(outputs: dict, invalid_mask_dev: torch.Tensor) -> torch.Tensor:
    probs = [torch.softmax(outputs[col], dim=1) for col in LABEL_COLS]
    B = probs[0].shape[0]
    joint = probs[0]
    for k in range(1, len(probs)):
        p = probs[k]
        joint = joint.unsqueeze(-1) * p.view([B] + [1] * k + [p.shape[1]])
    invalid_mass = (joint * invalid_mask_dev).sum(dim=list(range(1, len(LABEL_COLS) + 1)))
    return invalid_mass.mean()


def train_epoch(model, loader, optimizer, criteria, valid_mask, combo_weight=1.0):
    model.train()
    total_loss = 0.0
    n = 0
    invalid_mask_dev = (1.0 - valid_mask).to(device)
    for virchow_feat, sem_images, labels in loader:
        virchow_feat = virchow_feat.to(device)
        sem_images   = sem_images.to(device)
        labels       = {col: labels[col].to(device) for col in LABEL_COLS}
        optimizer.zero_grad()
        outputs = model(virchow_feat, sem_images)
        loss = sum(criteria[col](outputs[col], labels[col]) for col in LABEL_COLS)
        if combo_weight > 0:
            loss = loss + combo_weight * invalid_combo_penalty(outputs, invalid_mask_dev)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def compute_val_loss(model, loader, criteria, valid_mask, combo_weight=1.0):
    model.eval()
    total_loss = 0.0
    n = 0
    invalid_mask_dev = (1.0 - valid_mask).to(device)
    with torch.inference_mode():
        for virchow_feat, sem_images, labels in loader:
            virchow_feat = virchow_feat.to(device)
            sem_images   = sem_images.to(device)
            labels       = {col: labels[col].to(device) for col in LABEL_COLS}
            outputs = model(virchow_feat, sem_images)
            loss = sum(criteria[col](outputs[col], labels[col]) for col in LABEL_COLS)
            if combo_weight > 0:
                loss = loss + combo_weight * invalid_combo_penalty(outputs, invalid_mask_dev)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def collect_predictions(model, loader):
    model.eval()
    all_preds = {col: [] for col in LABEL_COLS}
    all_true  = {col: [] for col in LABEL_COLS}
    with torch.inference_mode():
        for virchow_feat, sem_images, labels in loader:
            virchow_feat = virchow_feat.to(device)
            sem_images   = sem_images.to(device)
            outputs = model(virchow_feat, sem_images)
            for col in LABEL_COLS:
                all_preds[col].extend(outputs[col].argmax(dim=1).cpu().numpy())
                all_true[col].extend(labels[col].numpy())
    return all_preds, all_true


def evaluate_and_collect(model, loader, vocabs, verbose=True):
    all_preds, all_true = collect_predictions(model, loader)
    run_metrics = {}
    for col in LABEL_COLS:
        idx_to_label = {v: k for k, v in vocabs[col].items()}
        labels_idx = sorted(idx_to_label)
        target_names = [idx_to_label[i] for i in labels_idx]
        report = classification_report(
            all_true[col], all_preds[col],
            labels=labels_idx, target_names=target_names,
            zero_division=0, output_dict=True,
        )
        if verbose:
            print(f"\n--- {col} ---", flush=True)
            print(classification_report(
                all_true[col], all_preds[col],
                labels=labels_idx, target_names=target_names,
                zero_division=0,
            ), flush=True)
        run_metrics[col] = {'report': report, 'true': all_true[col], 'pred': all_preds[col]}
    return run_metrics


def plot_confusion_matrices(run_metrics, vocabs, out_dir, run_idx):
    fig, axes = plt.subplots(1, len(LABEL_COLS), figsize=(5 * len(LABEL_COLS), 4))
    for ax, col in zip(axes, LABEL_COLS):
        idx_to_label = {v: k for k, v in vocabs[col].items()}
        labels_idx = sorted(idx_to_label)
        labels = [idx_to_label[i] for i in labels_idx]
        cm = confusion_matrix(run_metrics[col]['true'], run_metrics[col]['pred'], labels=labels_idx)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels,
                    ax=ax, cmap='Blues', cbar=False)
        ax.set_title(col)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=45)
    plt.suptitle(f'Run {run_idx} Confusion Matrices', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'confusion_run{run_idx}.png')
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()


def plot_f1_bars(run_metrics, vocabs, out_dir, run_idx):
    fig, axes = plt.subplots(1, len(LABEL_COLS), figsize=(5 * len(LABEL_COLS), 4))
    for ax, col in zip(axes, LABEL_COLS):
        report = run_metrics[col]['report']
        idx_to_label = {v: k for k, v in vocabs[col].items()}
        labels = [idx_to_label[i] for i in sorted(idx_to_label)]
        f1s = [report.get(lbl, {}).get('f1-score', 0.0) for lbl in labels]
        ax.bar(labels, f1s, color='steelblue')
        ax.set_ylim(0, 1.1)
        ax.set_title(col)
        ax.set_ylabel('F1-score')
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(f1s):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    plt.suptitle(f'Run {run_idx} Per-Class F1', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'f1_run{run_idx}.png')
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()


def plot_loss_curve(train_losses, val_losses, out_dir, run_idx):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Run {run_idx} Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f'loss_run{run_idx}.png')
    plt.savefig(path, dpi=120)
    plt.close()


def plot_mean_std_f1(all_run_metrics, vocabs, out_dir):
    fig, axes = plt.subplots(1, len(LABEL_COLS), figsize=(5 * len(LABEL_COLS), 4))
    for ax, col in zip(axes, LABEL_COLS):
        idx_to_label = {v: k for k, v in vocabs[col].items()}
        labels = [idx_to_label[i] for i in sorted(idx_to_label)]
        f1_matrix = []
        for run_metrics in all_run_metrics:
            report = run_metrics[col]['report']
            f1_matrix.append([report.get(lbl, {}).get('f1-score', 0.0) for lbl in labels])
        f1_matrix = np.array(f1_matrix)
        means = f1_matrix.mean(axis=0)
        stds  = f1_matrix.std(axis=0)
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.15)
        ax.set_title(col)
        ax.set_ylabel('F1-score')
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + 0.03, f'{m:.2f}', ha='center', fontsize=8)
    n_runs = len(all_run_metrics)
    plt.suptitle(f'Mean ± Std F1 across {n_runs} runs', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'mean_std_f1.png')
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()


def plot_macro_f1_across_runs(all_run_metrics, out_dir):
    plt.figure(figsize=(8, 5))
    for col in LABEL_COLS:
        macro_f1s = [rm[col]['report']['macro avg']['f1-score'] for rm in all_run_metrics]
        plt.plot(macro_f1s, marker='o', label=col)
    plt.xlabel('Run')
    plt.ylabel('Macro F1')
    plt.title('Macro F1 per head across runs')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'macro_f1_runs.png')
    plt.savefig(path, dpi=120)
    plt.close()


def run_training(args, train_recs, val_recs, vocabs, run_idx):
    torch.manual_seed(args.seed + run_idx)
    np.random.seed(args.seed + run_idx)

    train_ds = CachedTileDataset(train_recs, sem_size=args.sem_size, augment=True)
    val_ds   = CachedTileDataset(val_recs,   sem_size=args.sem_size, augment=False)
    drop_last = len(train_ds) > args.batch_size
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                              drop_last=drop_last)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    head_sizes = {col: len(vocabs[col]) for col in LABEL_COLS}
    model = MultiHeadClassifier(head_sizes).to(device)

    head_weights = compute_head_weights(train_recs, vocabs)
    criteria = {col: nn.CrossEntropyLoss(weight=head_weights[col]) for col in LABEL_COLS}

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    valid_mask, valid_combos = build_valid_combinations(train_recs, vocabs)

    train_losses = []
    val_losses   = []
    best_val_loss = float('inf')
    best_path = os.path.join(args.out_dir, f'best_run{run_idx}.pth')

    for epoch in range(args.num_epochs):
        tr_loss = train_epoch(model, train_loader, optimizer, criteria, valid_mask, combo_weight=1.0)
        va_loss = compute_val_loss(model, val_loader, criteria, valid_mask, combo_weight=1.0)
        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        pct = (epoch + 1) / args.num_epochs * 100
        print(f"[Run {run_idx}] {pct:.1f}% | Epoch [{epoch+1}/{args.num_epochs}] | "
              f"Train: {tr_loss:.4f} | Val: {va_loss:.4f}", flush=True)
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({
                'model_state': model.state_dict(),
                'vocabs': vocabs,
                'head_sizes': head_sizes,
                'valid_combinations': list(valid_combos),
            }, best_path)
        if (epoch + 1) % args.eval_every == 0 or epoch == args.num_epochs - 1:
            evaluate_and_collect(model, val_loader, vocabs, verbose=True)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)['model_state'])
    print(f"\n[Run {run_idx}] Final evaluation (best val-loss checkpoint):", flush=True)
    run_metrics = evaluate_and_collect(model, val_loader, vocabs, verbose=True)

    plot_confusion_matrices(run_metrics, vocabs, args.out_dir, run_idx)
    plot_f1_bars(run_metrics, vocabs, args.out_dir, run_idx)
    plot_loss_curve(train_losses, val_losses, args.out_dir, run_idx)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return run_metrics, best_val_loss, best_path


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_annotation_csvs(args.slide_dirs)
    print(f"Total annotated tiles: {len(df)}", flush=True)

    print("Building label vocabularies:", flush=True)
    vocabs = build_vocabs(df)

    print("Resolving tile paths...", flush=True)
    records = build_records(df, vocabs)
    if len(records) == 0:
        raise RuntimeError("No valid records. Check slide_dirs and tile directory structure.")

    print("Building Virchow2 image transform...", flush=True)
    transform = build_virchow2_transform()

    print("Caching Virchow2 features (skips already-cached tiles)...", flush=True)
    extract_and_cache_virchow2_features(records, transform, batch_size=args.batch_size)

    train_recs, val_recs = train_test_split(records, test_size=args.val_split, random_state=args.seed)
    print(f"Train: {len(train_recs)}, Val: {len(val_recs)}", flush=True)

    all_run_metrics   = []
    best_overall_loss = float('inf')
    best_overall_path = None

    for run_idx in range(1, args.num_runs + 1):
        print(f"\n{'='*60}\nRUN {run_idx} / {args.num_runs}\n{'='*60}", flush=True)
        run_metrics, run_best_loss, run_best_path = run_training(
            args, train_recs, val_recs, vocabs, run_idx
        )
        all_run_metrics.append(run_metrics)
        if run_best_loss < best_overall_loss:
            best_overall_loss = run_best_loss
            best_overall_path = run_best_path

    shutil.copy2(best_overall_path, args.checkpoint_out)
    print(f"\nBest checkpoint (val_loss={best_overall_loss:.4f}) saved to: {args.checkpoint_out}", flush=True)

    if args.num_runs > 1:
        plot_mean_std_f1(all_run_metrics, vocabs, args.out_dir)
        plot_macro_f1_across_runs(all_run_metrics, args.out_dir)

    print("\nFinal Summary (macro F1 per head per run):", flush=True)
    header = f"{'Run':<6}" + "".join(f"{col:<22}" for col in LABEL_COLS)
    print(header, flush=True)
    for i, rm in enumerate(all_run_metrics):
        row = f"{i+1:<6}" + "".join(
            f"{rm[col]['report']['macro avg']['f1-score']:<22.4f}" for col in LABEL_COLS
        )
        print(row, flush=True)
    if args.num_runs > 1:
        means = {col: np.mean([rm[col]['report']['macro avg']['f1-score'] for rm in all_run_metrics])
                 for col in LABEL_COLS}
        stds  = {col: np.std([rm[col]['report']['macro avg']['f1-score'] for rm in all_run_metrics])
                 for col in LABEL_COLS}
        print("Mean  " + "".join(f"{means[col]:<22.4f}" for col in LABEL_COLS), flush=True)
        print("Std   " + "".join(f"{stds[col]:<22.4f}"  for col in LABEL_COLS), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-head histopathology tile classifier (Virchow2)')
    parser.add_argument('--slide_dirs',     required=True, nargs='+')
    parser.add_argument('--out_dir',        type=str,   default='results')
    parser.add_argument('--checkpoint_out', type=str,   default='multihead_classifier.pth')
    parser.add_argument('--sem_size',       type=int,   default=128)
    parser.add_argument('--num_epochs',     type=int,   default=100)
    parser.add_argument('--batch_size',     type=int,   default=256)
    parser.add_argument('--learning_rate',  type=float, default=1e-3)
    parser.add_argument('--val_split',      type=float, default=0.2)
    parser.add_argument('--eval_every',     type=int,   default=10)
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--num_runs',       type=int,   default=3)
    parser.add_argument('--seed',           type=int,   default=42)
    return parser.parse_args()


if __name__ == '__main__':
    main()
