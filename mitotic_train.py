#!/usr/bin/env python3
"""
Mitotic 3-class classifier.

Classes:
    0 — non-mitotic   (mitotic column == 'None')
    1 — Mitotic Figure
    2 — Bizarre Figure

Usage:
    python mitotic_train.py \
        --slide_dirs \
            /home/verma198/webserver1 \
            ... \
        --out_dir ./mitotic_results \
        --checkpoint_out mitotic_classifier.pth \
        --num_epochs 50 \
        --batch_size 128
"""
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, f1_score
)
import timm
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RAW_SUBDIR    = os.path.join('tiles1', 'nonbox')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES   = ['non-mitotic', 'Mitotic Figure', 'Bizarre Figure']
LABEL_MAP     = {'None': 0, 'Mitotic Figure': 1, 'Bizarre Figure': 2}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_tile_file(slide_dir: str, tile_num: int) -> str | None:
    d = os.path.join(slide_dir, RAW_SUBDIR)
    if not os.path.isdir(d):
        return None
    for fname in os.listdir(d):
        nums = re.findall(r'\d+', os.path.splitext(fname)[0])
        if nums and int(nums[-1]) == tile_num:
            return os.path.join(d, fname)
    return None


def load_records(slide_dirs: list[str]) -> list[dict]:
    records = []
    for slide_dir in slide_dirs:
        csvs = sorted(glob.glob(os.path.join(slide_dir, 'annotation_data_*.csv')))
        if not csvs:
            print(f"  No CSV in {slide_dir}, skipping", flush=True)
            continue
        df = pd.read_csv(csvs[-1], keep_default_na=False)
        df.columns = [c.strip().lower() for c in df.columns]
        df['tile_num'] = pd.to_numeric(
            df['desc'].str.extract(r'(\d+)')[0], errors='coerce'
        )
        df = df.dropna(subset=['tile_num'])
        df['tile_num'] = df['tile_num'].astype(int)

        counts = {0: 0, 1: 0, 2: 0}
        miss = 0
        for _, row in df.iterrows():
            path = find_tile_file(slide_dir, int(row['tile_num']))
            if path is None:
                miss += 1
                continue
            val   = str(row.get('mitotic', 'None')).strip()
            label = LABEL_MAP.get(val, 0)
            records.append({'path': path, 'label': label})
            counts[label] += 1

        print(
            f"  {os.path.basename(slide_dir)}: "
            f"{counts[1]} mitotic, {counts[2]} bizarre, {counts[0]} non-mitotic, {miss} missing",
            flush=True,
        )
    return records


def load_negatives_from_dirs(negative_dirs: list[str], max_negatives: int, seed: int) -> list[dict]:
    all_paths = []
    for slide_dir in negative_dirs:
        d = os.path.join(slide_dir, RAW_SUBDIR)
        if not os.path.isdir(d):
            print(f"  {os.path.basename(slide_dir)}: no tile dir, skipping", flush=True)
            continue
        paths = [
            os.path.join(d, f)
            for f in os.listdir(d)
            if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        ]
        print(f"  {os.path.basename(slide_dir)}: {len(paths)} tiles as negatives", flush=True)
        all_paths.extend(paths)

    if not all_paths:
        return []

    rng = np.random.default_rng(seed)
    if max_negatives > 0 and len(all_paths) > max_negatives:
        all_paths = rng.choice(all_paths, size=max_negatives, replace=False).tolist()

    print(f"  Total negatives sampled: {len(all_paths)}", flush=True)
    return [{'path': p, 'label': 0} for p in all_paths]


# ---------------------------------------------------------------------------
# Dataset and transforms
# ---------------------------------------------------------------------------

def build_transforms():
    train_tf = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(180),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


class TileDataset(Dataset):
    def __init__(self, records: list[dict], transform):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        img = Image.open(r['path']).convert('RGB')
        return self.transform(img), torch.tensor(r['label'], dtype=torch.long)


def build_weighted_sampler(records: list[dict]) -> WeightedRandomSampler:
    labels        = np.array([r['label'] for r in records])
    class_counts  = np.bincount(labels, minlength=3)
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(float)
    weights       = class_weights[labels]
    return WeightedRandomSampler(
        torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MitoticClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))  # (B, 3)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            all_logits.append(model(imgs.to(device)).cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    exp    = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs  = exp / exp.sum(axis=1, keepdims=True)
    preds  = np.argmax(probs, axis=1)
    try:
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0
    f1     = f1_score(labels, preds, average='macro', zero_division=0)
    report = classification_report(
        labels, preds,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    return auc, f1, report, cm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(train_losses, val_aucs, cm, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[1].plot(val_aucs)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Validation AUC')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mitotic_curves.png'), dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=ax, cmap='Blues', cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mitotic_confusion.png'), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading annotated records...", flush=True)
    records = load_records(args.slide_dirs)

    if args.negative_dirs:
        print("Loading negatives from raw tile dirs...", flush=True)
        neg_records = load_negatives_from_dirs(args.negative_dirs, args.max_negatives, args.seed)
        records.extend(neg_records)

    if not records:
        raise RuntimeError("No valid records found.")

    labels_arr = np.array([r['label'] for r in records])
    counts     = np.bincount(labels_arr, minlength=3)
    print(
        f"Total: {len(records)} | "
        f"{counts[0]} non-mitotic, {counts[1]} Mitotic Figure, {counts[2]} Bizarre Figure",
        flush=True,
    )

    if counts[1] == 0 and counts[2] == 0:
        raise RuntimeError("No positive records found.")

    train_recs, val_recs = train_test_split(
        records,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=labels_arr,
    )
    print(f"Train: {len(train_recs)}, Val: {len(val_recs)}", flush=True)

    train_tf, val_tf = build_transforms()
    train_loader = DataLoader(
        TileDataset(train_recs, train_tf),
        batch_size=args.batch_size,
        sampler=build_weighted_sampler(train_recs),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TileDataset(val_recs, val_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model     = MitoticClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    train_losses, val_aucs = [], []
    best_auc  = 0.0
    best_path = os.path.join(args.out_dir, 'best_mitotic.pth')

    for epoch in range(args.num_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        train_losses.append(loss)

        auc, f1, report, cm = evaluate(model, val_loader)
        val_aucs.append(auc)

        pct = (epoch + 1) / args.num_epochs * 100
        print(
            f"{pct:.1f}% | Epoch [{epoch+1}/{args.num_epochs}] | "
            f"Loss: {loss:.4f} | Val AUC: {auc:.4f} | Val F1: {f1:.4f}",
            flush=True,
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'best_auc':    float(best_auc),
                    'class_names': CLASS_NAMES,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    auc, f1, report, cm = evaluate(model, val_loader)
    print(f"\nFinal Val AUC: {auc:.4f}, F1: {f1:.4f}", flush=True)
    print(report, flush=True)

    save_plots(train_losses, val_aucs, cm, args.out_dir)

    torch.save(
        {
            'model_state': model.state_dict(),
            'best_auc':    float(auc),
            'class_names': CLASS_NAMES,
        },
        args.checkpoint_out,
    )
    print(f"Checkpoint saved -> {args.checkpoint_out}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_dirs',     nargs='+', required=True,
                        help='Dirs with annotation CSVs')
    parser.add_argument('--negative_dirs',  nargs='+', default=[],
                        help='Dirs where ALL tile images are treated as negatives (no CSV needed)')
    parser.add_argument('--max_negatives',  type=int,   default=20000,
                        help='Cap on negatives sampled from --negative_dirs (0 = no cap)')
    parser.add_argument('--out_dir',        default='./mitotic_results')
    parser.add_argument('--checkpoint_out', default='mitotic_classifier.pth')
    parser.add_argument('--num_epochs',     type=int,   default=50)
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--learning_rate',  type=float, default=1e-4)
    parser.add_argument('--val_split',      type=float, default=0.2)
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--seed',           type=int,   default=42)
    return parser.parse_args()


if __name__ == '__main__':
    main()