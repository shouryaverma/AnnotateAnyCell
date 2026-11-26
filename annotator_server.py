#!/usr/bin/env python3
"""
annotator_server.py - Multi-slide, multi-user histopathology annotation server.

Views served by templates/index.html + static/js/main.js:
  UMAP  - scatter of tile embeddings, coloured by annotation
  GRID  - tile grid (GT / PRED / ALL) with class filters and edit panel
  SLIDE - full-slide image with labelled cell boxes, lasso cell adder, and histograms

Storage backends (via storage.py):
  * Local default : SQLite (data/annotator.db) - zero setup required
  * Cloud / Atlas : MongoDB when MONGO_URI or MONGO_USERNAME/PASSWORD/HOST are set

Slide discovery & upload:
  * Auto-discovers preprocessed slide folders inside --slides-dir (default: ./slides)
  * Also accepts explicit --slides /path/a /path/b ...
  * Supports in-app upload & live preprocessing of raw histopathology slides
    (.png, .jpg, .tif, .tiff) up to --max-slide-dim (default: 7000x7000 px)

Quick local launch:
    python annotator_server.py

Cloud / MongoDB launch:
    export MONGO_URI="mongodb+srv://user:pass@cluster0.xxx.mongodb.net/"
    export MONGO_DB="annotator_db"
    export ALLOWED_USERS="verma,sola,english"
    python annotator_server.py --slides-dir ./slides --host 0.0.0.0 --port 5006
"""
import os
import re
import sys
import glob
import gzip
import json
import heapq
import atexit
import pickle
import secrets
import logging
import argparse
import threading
import subprocess
from collections import deque
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from urllib.parse import unquote

import numpy as np
import pandas as pd
from PIL import Image
from flask import (
    Flask, Response, jsonify, render_template, request,
    send_file, session, stream_with_context,
)
from werkzeug.exceptions import HTTPException

from storage import create_store, resolve_mongo_uri
from update_celladder import add_manual_cell, delete_manual_cell

Image.MAX_IMAGE_PIXELS = None

# -- Constants -----------------------------------------------------------------

LABEL_COLS = ['mitotic', 'chromatin', 'nucleoli', 'prominent', 'shape']
LABEL_OPTIONS = {
    'mitotic':   ['None', 'Mitotic Figure', 'Bizarre Figure'],
    'chromatin': ['None', 'Vesicular Chromatin'],
    'nucleoli':  ['None', 'Single Nucleoli', 'Multiple Nucleoli'],
    'prominent': ['None', 'Prominent Nucleoli'],
    'shape':     ['None', 'Circle', 'Oval', 'Irregular Shape'],
}
LABEL_INT = {col: {name: i for i, name in enumerate(opts)} for col, opts in LABEL_OPTIONS.items()}
LABEL_STR = {col: {v: k for k, v in LABEL_INT[col].items()} for col in LABEL_COLS}
TOP_N = 100

USERNAME_RE = re.compile(r'^[a-z0-9._-]{2,32}$')
UPLOAD_EXTS = frozenset({'png', 'jpg', 'jpeg', 'tif', 'tiff'})
IMAGE_MIME = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'tif': 'image/tiff'}
IMAGE_MAX_AGE = 3600
MAX_POLYGON_POINTS = 5000
MAX_VERSIONS_LISTED = 200
STREAM_PING_SECONDS = 15
PROGRESS_RE = re.compile(r'^Progress:\s*([0-9.]+)%\s*\|?\s*(.*)$')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
SECRET_KEY_FILE = os.path.join(BASE_DIR, '.annotator_secret_key')

log = logging.getLogger('annotator')

# -- Flask + globals -----------------------------------------------------------

app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=300 * 1024 * 1024,  # allow raw TIFF/PNG uploads up to 300 MB
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)

ARGS = None
ALLOWED_USERS: frozenset[str] | None = None  # None = open registration (any valid username)
SLIDE_DIRS: dict[str, str] = {}              # name -> abs path
SLIDE_ROOTS: list[str] = []                  # real paths, used by /img containment check
_registry_lock = threading.RLock()
_slide_cache: dict[str, dict] = {}
_slide_locks: dict[str, threading.RLock] = {}

_store = None
_store_lock = threading.Lock()

_file_cache: dict = {}
_file_cache_lock = threading.Lock()

_jobs: dict = {}
_jobs_lock = threading.Lock()
_preprocess_jobs: dict = {}
_preprocess_lock = threading.Lock()
_gpu_lock = threading.Lock()                 # one GPU job (recluster or preprocess) at a time
_active_procs: set = set()


# -- Helpers -------------------------------------------------------------------

def safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]', '_', value or '')


def get_store():
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = create_store(
                    sqlite_path=ARGS.sqlite_path,
                    mongo_uri=ARGS.mongo_uri,
                    mongo_username=ARGS.mongo_username,
                    mongo_password=ARGS.mongo_password,
                    mongo_host=ARGS.mongo_host,
                    mongo_db=ARGS.mongo_db,
                    backend=ARGS.storage,
                )
    return _store


def user_allowed(username: str | None) -> bool:
    if not username or not USERNAME_RE.match(username):
        return False
    if ALLOWED_USERS is None:
        return True
    return username in ALLOWED_USERS


# -- Slide tile index & registration -------------------------------------------

def scan_tiles(directory: str) -> dict[int, str]:
    tiles: dict[int, str] = {}
    if not os.path.isdir(directory):
        return tiles
    with os.scandir(directory) as it:
        for entry in it:
            stem, ext = os.path.splitext(entry.name)
            if ext.lower().lstrip('.') not in IMAGE_MIME:
                continue
            nums = re.findall(r'\d+', stem)
            if nums:
                tiles[int(nums[-1])] = entry.path
    return tiles


def build_slide_cache(slide_name: str) -> dict:
    slide_dir = SLIDE_DIRS[slide_name]
    tiles1 = scan_tiles(os.path.join(slide_dir, 'tiles1', 'nonbox'))
    tiles2 = scan_tiles(os.path.join(slide_dir, 'tiles2', 'nonbox'))
    tile_nums = sorted(tiles1)
    cache = {
        'tile_nums': tile_nums,
        'tile_to_idx': {tn: i for i, tn in enumerate(tile_nums)},
        'tiles1': tiles1,
        'tiles2': tiles2,
    }
    _slide_cache[slide_name] = cache
    return cache


def get_slide_cache(slide_name: str) -> dict:
    cache = _slide_cache.get(slide_name)
    return cache if cache is not None else build_slide_cache(slide_name)


def is_preprocessed_slide_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and (
            os.path.isdir(os.path.join(path, 'tiles1', 'nonbox'))
            or os.path.exists(os.path.join(path, 'compressed_image.jpg'))
        )
    )


def register_slide_dynamic(slide_dir: str, name: str | None = None) -> str:
    slide_dir = os.path.abspath(slide_dir)
    slide_name = name or os.path.basename(slide_dir.rstrip(os.sep))
    real_root = os.path.realpath(slide_dir)
    with _registry_lock:
        SLIDE_DIRS[slide_name] = slide_dir
        if real_root not in SLIDE_ROOTS:
            SLIDE_ROOTS.append(real_root)
        _slide_locks.setdefault(slide_name, threading.RLock())
        cache = build_slide_cache(slide_name)
    log.info('Registered slide: %s -> %s (%d tiles)', slide_name, slide_dir, len(cache['tile_nums']))
    return slide_name


# -- File helpers --------------------------------------------------------------

def read_cached(path: str, loader):
    """Load a file through an in-memory cache invalidated by (mtime_ns, size)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    sig = (st.st_mtime_ns, st.st_size)
    with _file_cache_lock:
        entry = _file_cache.get(path)
    if entry is not None and entry[0] == sig:
        return entry[1]
    value = loader(path)
    with _file_cache_lock:
        _file_cache[path] = (sig, value)
    return value


def _parse_annotation_csv(path: str) -> dict:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'desc' not in df.columns:
        df['desc'] = ''
    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = 'None'
        df[col] = df[col].astype(str).str.strip().replace({'': 'None', 'nan': 'None', 'NaN': 'None'})
    tile_num = pd.to_numeric(df['desc'].str.extract(r'(\d+)')[0], errors='coerce')
    labeled = (df[LABEL_COLS] != 'None').any(axis=1)

    gt: dict[int, dict] = {}
    desc: dict[int, str] = {}
    keep = (labeled & tile_num.notna()).to_numpy()
    cols = [df[c].to_numpy()[keep] for c in LABEL_COLS]
    for i, (tn, d) in enumerate(zip(tile_num.to_numpy()[keep], df['desc'].to_numpy()[keep])):
        tn = int(tn)
        gt[tn] = {c: str(cols[j][i]) for j, c in enumerate(LABEL_COLS)}
        desc[tn] = str(d)
    return {'path': path, 'df': df, 'tile_num': tile_num, 'gt': gt, 'desc': desc}


def load_baseline_annotations(slide_dir: str) -> dict | None:
    """Baseline annotation CSV shipped with the slide (excludes per-user *_user_<name>.csv)."""
    all_csvs = sorted(glob.glob(os.path.join(slide_dir, 'annotation_data_*.csv')))
    baseline = [p for p in all_csvs if not os.path.basename(p).startswith('annotation_data_user_')]
    if not baseline:
        return None
    return read_cached(baseline[-1], _parse_annotation_csv)


def load_user_csv_annotations(slide_dir: str, username: str) -> dict | None:
    path = os.path.join(slide_dir, f'annotation_data_user_{safe_name(username)}.csv')
    if not os.path.exists(path):
        return None
    return read_cached(path, _parse_annotation_csv)


def load_gt(slide_dir: str, slide_name: str | None = None, username: str | None = None) -> dict[int, dict]:
    """
    Effective ground-truth labels for a slide:
      baseline slide CSV < user CSV < user DB annotations.
    Tiles explicitly saved as all-None by the user are removed from GT.
    """
    gt: dict[int, dict] = {}
    base = load_baseline_annotations(slide_dir)
    if base:
        gt.update(base['gt'])
    if username:
        user_csv = load_user_csv_annotations(slide_dir, username)
        if user_csv:
            gt.update(user_csv['gt'])
        if slide_name:
            try:
                db_ann = get_store().get_user_annotations(username, slide_name)
                for tn, lbls in db_ann.items():
                    if any(lbls.get(c, 'None') != 'None' for c in LABEL_COLS):
                        gt[tn] = {c: lbls.get(c, 'None') for c in LABEL_COLS}
                    else:
                        gt.pop(tn, None)
            except Exception:
                log.exception('user_annotations lookup failed for %s/%s', username, slide_name)
    return gt


def load_centroids(slide_dir: str) -> dict:
    def _load(path):
        result = {}
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip().strip('[]')
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        result[i] = {'x': float(parts[0]), 'y': float(parts[1])}
                    except ValueError:
                        pass
        return result
    return read_cached(os.path.join(slide_dir, 'centroids.csv'), _load) or {}


def load_predictions(slide_dir: str, username: str | None = None) -> list:
    def _load(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return []
        return data if isinstance(data, list) else []

    if username:
        user_pred = os.path.join(slide_dir, f'predictions_{safe_name(username)}.json')
        if os.path.exists(user_pred):
            return read_cached(user_pred, _load) or []
    return read_cached(os.path.join(slide_dir, 'predictions.json'), _load) or []


def load_distributions(slide_dir: str) -> dict | None:
    def _load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return read_cached(os.path.join(slide_dir, 'areas_and_ratio.pkl'), _load)


def load_init_embeddings(slide_dir: str) -> np.ndarray | None:
    def _load(path):
        with open(path, 'rb') as f:
            return np.asarray(pickle.load(f), dtype=np.float64)
    return read_cached(os.path.join(slide_dir, 'init_umap_embeddings.pkl'), _load)


def load_manual_tile_nums(slide_dir: str) -> list:
    def _load(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return []
        return data if isinstance(data, list) else []
    return read_cached(os.path.join(slide_dir, 'manual_cells.json'), _load) or []


def save_labels_csv(slide_dir: str, username: str, tile_num: int, labels: dict) -> str:
    """Write labels for one tile into the user's per-slide CSV (caller holds the slide lock)."""
    path = os.path.join(slide_dir, f'annotation_data_user_{safe_name(username)}.csv')
    ann = load_user_csv_annotations(slide_dir, username)
    if ann is None:
        df = pd.DataFrame(columns=['x', 'y', 'desc'] + LABEL_COLS, dtype=str)
        mask = np.zeros(0, dtype=bool)
    else:
        df = ann['df'].copy()
        mask = (ann['tile_num'] == tile_num).to_numpy()

    if mask.any():
        for col, val in labels.items():
            df.loc[mask, col] = val
    else:
        centroid = load_centroids(slide_dir).get(tile_num, {})
        row = {col: '' for col in df.columns}
        row.update({col: 'None' for col in LABEL_COLS})
        row.update({'desc': f'Tile {tile_num}'})
        if 'x' in df.columns:
            row['x'] = str(centroid.get('x', ''))
        if 'y' in df.columns:
            row['y'] = str(centroid.get('y', ''))
        row.update(labels)
        df = pd.concat([df, pd.DataFrame([row], dtype=str)], ignore_index=True)

    tmp = f'{path}.tmp.{os.getpid()}.{threading.get_ident()}'
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


# -- Tile listings -------------------------------------------------------------

def build_tile_record(cache: dict, tile_num: int, desc: str, source: str,
                      labels: dict, probs: dict) -> dict | None:
    grid_img = cache['tiles1'].get(tile_num)
    if not grid_img:
        return None
    return {'tile_num': tile_num, 'desc': desc, 'source': source,
            'labels': labels, 'probs': probs, 'grid_path': grid_img,
            'raw_path': cache['tiles2'].get(tile_num)}


def parse_filters(raw: str) -> dict[str, set]:
    try:
        data = json.loads(raw or '{}')
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {h: {str(x) for x in v} for h, v in data.items()
            if h in LABEL_COLS and isinstance(v, list) and v}


def get_labeled_tiles(cache: dict, slide_dir: str, slide_name: str,
                      username: str, filters: dict[str, set]) -> list:
    gt = load_gt(slide_dir, slide_name, username)
    if not gt:
        return []
    no_probs = {c: None for c in LABEL_COLS}
    tiles = []
    for tn, labels in gt.items():
        if any(labels.get(h) not in allowed for h, allowed in filters.items()):
            continue
        rec = build_tile_record(cache, tn, f'Tile {tn}', 'gt', dict(labels), dict(no_probs))
        if rec:
            tiles.append(rec)
    return tiles


def get_prediction_tiles(cache: dict, slide_dir: str, slide_name: str,
                         username: str, filters: dict[str, set]) -> list:
    predictions = load_predictions(slide_dir, username)
    if not predictions:
        return []
    gt_nums = load_gt(slide_dir, slide_name, username).keys()
    rank_cols = list(filters) or LABEL_COLS

    def confidence(p):
        return max((p.get(f'{h}_prob') or 0.) for h in rank_cols)

    candidates = (p for p in predictions
                  if p.get('tile_num') not in gt_nums
                  and all(p.get(h) in allowed for h, allowed in filters.items()))
    tiles = []
    for p in heapq.nlargest(TOP_N, candidates, key=confidence):
        tn = p.get('tile_num')
        if tn is None:
            continue
        rec = build_tile_record(cache, tn, p.get('desc', f'Tile {tn}'), 'pred',
                                {c: p.get(c, 'None') for c in LABEL_COLS},
                                {c: p.get(f'{c}_prob') for c in LABEL_COLS})
        if rec:
            tiles.append(rec)
    return tiles


# -- UMAP / annotation state ---------------------------------------------------

def take_by_tile_num(values, tile_nums: list[int], width: int | None = None) -> np.ndarray:
    idx = np.asarray(tile_nums, dtype=np.int64)
    shape = (len(idx),) if width is None else (len(idx), width)
    out = np.zeros(shape, dtype=np.float64)
    if values is None:
        return out
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != (1 if width is None else 2) or len(arr) == 0:
        return out
    ok = idx < len(arr)
    out[ok] = arr[idx[ok]] if width is None else arr[idx[ok], :width]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def load_umap_embeddings(username: str, slide_name: str, slide_dir: str, cache: dict) -> np.ndarray:
    tile_nums = cache['tile_nums']
    n = len(tile_nums)
    xy = take_by_tile_num(load_init_embeddings(slide_dir), tile_nums, width=2)
    try:
        doc = get_store().get_umap_embeddings(username, slide_name)
    except Exception:
        log.exception('umap_embeddings lookup failed for %s/%s', username, slide_name)
        doc = None
    if not doc or not doc.get('embeddings'):
        return xy
    emb = np.asarray(doc['embeddings'], dtype=np.float64)
    if emb.ndim != 2 or emb.shape[1] < 2:
        return xy
    doc_tiles = doc.get('tile_nums')
    if doc_tiles and len(doc_tiles) == len(emb):
        tile_to_idx = cache['tile_to_idx']
        pairs = [(tile_to_idx[tn], j) for j, tn in enumerate(doc_tiles) if tn in tile_to_idx]
        if pairs:
            dst, src = map(list, zip(*pairs))
            xy[dst] = emb[src, :2]
    else:
        m = min(n, len(emb))
        xy[:m] = emb[:m, :2]
    return np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)


def to_int(value, upper: int | None = None) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    if v < 0 or (upper is not None and v > upper):
        return 0
    return v


def place_by_tile_num(values, doc_tiles, cache: dict, upper: int | None = None) -> list[int]:
    n = len(cache['tile_nums'])
    out = [0] * n
    values = list(values or [])
    if doc_tiles and len(doc_tiles) == len(values):
        tile_to_idx = cache['tile_to_idx']
        for tn, v in zip(doc_tiles, values):
            i = tile_to_idx.get(tn)
            if i is not None:
                out[i] = to_int(v, upper)
    else:
        for i, v in enumerate(values[:n]):
            out[i] = to_int(v, upper)
    return out


def unpack_version(version: dict, cache: dict) -> tuple[list, dict]:
    doc_tiles = version.get('tile_nums')
    emb = version.get('embeddingXY') or {}
    assoc = version.get('associated_data') or {}
    marked = [1 if m else 0 for m in place_by_tile_num(emb.get('marked'), doc_tiles, cache)]
    associated = {c: place_by_tile_num(assoc.get(c), doc_tiles, cache, len(LABEL_OPTIONS[c]) - 1)
                  for c in LABEL_COLS}
    return marked, associated


def load_annotation_state(username: str, slide_name: str, slide_dir: str,
                          cache: dict) -> tuple[list, dict]:
    n = len(cache['tile_nums'])
    marked = [0] * n
    associated = {c: [0] * n for c in LABEL_COLS}
    try:
        version = get_store().get_latest_version(username, slide_name)
        if version:
            marked, associated = unpack_version(version, cache)
    except Exception:
        log.exception('versions lookup failed for %s/%s', username, slide_name)
    tile_to_idx = cache['tile_to_idx']
    for tn, labels in load_gt(slide_dir, slide_name, username).items():
        idx = tile_to_idx.get(tn)
        if idx is None:
            continue
        marked[idx] = 1
        for c in LABEL_COLS:
            associated[c][idx] = LABEL_INT[c].get(labels.get(c, 'None'), 0)
    return marked, associated


def state_tile_nums(state: dict, length: int, cache: dict) -> list[int] | None:
    if length == 0:
        return []
    supplied = state.get('tile_nums')
    if isinstance(supplied, list) and len(supplied) == length:
        try:
            return [int(t) for t in supplied]
        except (TypeError, ValueError):
            return None
    if length == len(cache['tile_nums']):
        return list(cache['tile_nums'])
    return None


# -- Background GPU jobs (recluster + slide preprocessing) ---------------------

class ProgressJob:
    """State of one long-running subprocess; any number of SSE streams can follow it."""

    def __init__(self):
        self.cond = threading.Condition()
        self.seq = 0
        self.progress = 0.0
        self.message = 'Queued'
        self.done = False
        self.success = False
        self.error = ''
        self.extra: dict = {}

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        with self.cond:
            if progress is not None:
                self.progress = max(self.progress, min(float(progress), 100.0))
            if message is not None:
                self.message = message
            self.seq += 1
            self.cond.notify_all()

    def finish(self, success: bool, error: str = '', extra: dict | None = None) -> None:
        with self.cond:
            self.done, self.success, self.error = True, success, error
            if extra:
                self.extra.update(extra)
            self.seq += 1
            self.cond.notify_all()

    def wait(self, seen: int, timeout: float) -> tuple:
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seen, timeout)
            return (self.seq, self.progress, self.message,
                    self.done, self.success, self.error, dict(self.extra))


def _stream_subprocess(job: ProgressJob, cmd: list[str], env: dict, log_path: str, header: str) -> tuple[int, str]:
    tail: deque = deque(maxlen=20)
    with open(log_path, 'a') as logf:
        logf.write(f'\n===== {datetime.now().isoformat(timespec="seconds")} {header} =====\n')
        logf.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=BASE_DIR, env=env,
        )
        _active_procs.add(proc)
        try:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                stripped = line.strip()
                match = PROGRESS_RE.match(stripped)
                if match:
                    job.update(float(match.group(1)), match.group(2))
                elif stripped:
                    tail.append(stripped)
            rc = proc.wait()
        finally:
            _active_procs.discard(proc)
    return rc, '\n'.join(tail)[-600:]


def _storage_env() -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED='1',
               STORAGE_BACKEND=ARGS.storage,
               SQLITE_PATH=ARGS.sqlite_path,
               MONGO_DB=ARGS.mongo_db)
    uri = resolve_mongo_uri(ARGS.mongo_uri, ARGS.mongo_username, ARGS.mongo_password, ARGS.mongo_host)
    if uri:
        env['MONGO_URI'] = uri
    return env


def run_recluster(job: ProgressJob, username: str, slide_name: str, slide_dir: str) -> None:
    try:
        with _gpu_lock:
            job.update(message='Starting model update')
            cmd = [
                sys.executable, '-u', os.path.join(BASE_DIR, 'update_clusters.py'),
                '--slide_path', slide_dir,
                '--slide_name', slide_name,
                '--username', username,
                '--checkpoint', ARGS.checkpoint,
                '--storage', ARGS.storage,
                '--sqlite_path', ARGS.sqlite_path,
                '--mongo_db', ARGS.mongo_db,
            ]
            if ARGS.mitotic_checkpoint and os.path.exists(ARGS.mitotic_checkpoint):
                cmd += ['--mitotic_checkpoint', ARGS.mitotic_checkpoint,
                        '--mitotic_threshold', str(ARGS.mitotic_threshold)]
            log_path = os.path.join(
                LOG_DIR, f'recluster_{safe_name(slide_name)}_{safe_name(username)}.log'
            )
            log.info('Recluster start: user=%s slide=%s', username, slide_name)
            rc, tail = _stream_subprocess(job, cmd, _storage_env(), log_path,
                                          f'recluster user={username} slide={slide_name}')
        if rc == 0:
            log.info('Recluster done: user=%s slide=%s', username, slide_name)
            job.finish(True)
        else:
            log.error('Recluster failed (rc=%s): user=%s slide=%s, see %s',
                      rc, username, slide_name, log_path)
            job.finish(False, tail)
    except Exception as e:
        log.exception('Recluster crashed: user=%s slide=%s', username, slide_name)
        job.finish(False, str(e))


def run_preprocess(job: ProgressJob, slide_name: str, image_path: str, work_dir: str) -> None:
    try:
        with _gpu_lock:
            job.update(progress=1.0, message='Starting slide preprocessing')
            cmd = [
                sys.executable, '-u', os.path.join(BASE_DIR, 'preprocess.py'),
                '--slide-name', slide_name,
                '--work-dir', work_dir,
                '--image-path', image_path,
                '--max-dim', str(ARGS.max_slide_dim),
            ]
            if ARGS.checkpoint and os.path.exists(ARGS.checkpoint):
                cmd += ['--checkpoint', ARGS.checkpoint]
            if ARGS.mitotic_checkpoint and os.path.exists(ARGS.mitotic_checkpoint):
                cmd += ['--mitotic-checkpoint', ARGS.mitotic_checkpoint,
                        '--mitotic-threshold', str(ARGS.mitotic_threshold)]
            log_path = os.path.join(LOG_DIR, f'preprocess_{safe_name(slide_name)}.log')
            log.info('Preprocess start: slide=%s image=%s', slide_name, image_path)
            rc, tail = _stream_subprocess(job, cmd, _storage_env(), log_path,
                                          f'preprocess slide={slide_name}')
        if rc == 0:
            out_dir = os.path.join(work_dir, slide_name)
            register_slide_dynamic(out_dir, slide_name)
            log.info('Preprocess done: slide=%s', slide_name)
            job.finish(True, extra={'slide': slide_name})
        else:
            log.error('Preprocess failed (rc=%s): slide=%s, see %s', rc, slide_name, log_path)
            job.finish(False, tail)
    except Exception as e:
        log.exception('Preprocess crashed: slide=%s', slide_name)
        job.finish(False, str(e))


def terminate_children() -> None:
    for proc in list(_active_procs):
        if proc.poll() is None:
            proc.terminate()


def _sse_stream(job: ProgressJob | None):
    def generate():
        if job is None:
            yield f'data: {json.dumps({"type": "error", "error": "No active job"})}\n\n'
            return
        seen = -1
        while True:
            seq, progress, message, done, success, error, extra = job.wait(seen, STREAM_PING_SECONDS)
            if seq == seen:
                yield 'data: {"type":"ping"}\n\n'
                continue
            seen = seq
            if done:
                msg = {'type': 'done', 'success': success, **extra}
                if not success:
                    msg['error'] = error
                yield f'data: {json.dumps(msg)}\n\n'
                return
            yield f'data: {json.dumps({"type": "progress", "progress": progress, "message": message})}\n\n'

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


# -- Request plumbing ----------------------------------------------------------

PUBLIC_ENDPOINTS = {
    'index', 'static', 'healthz', 'api_config',
    'api_login', 'api_logout', 'api_session', 'api_slides',
}


def current_user() -> str:
    return session['user']


def json_body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


@app.before_request
def require_login():
    if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
        return None
    if not user_allowed(session.get('user')):
        return jsonify({'error': 'Not logged in'}), 401
    return None


@app.after_request
def finalize_response(resp):
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    if (resp.status_code == 200 and resp.mimetype == 'application/json'
            and not resp.direct_passthrough and not resp.is_streamed
            and 'Content-Encoding' not in resp.headers
            and 'gzip' in request.headers.get('Accept-Encoding', '')):
        data = resp.get_data()
        if len(data) >= 1024:
            resp.set_data(gzip.compress(data, compresslevel=5))
            resp.headers['Content-Encoding'] = 'gzip'
            resp.headers['Vary'] = 'Accept-Encoding'
    return resp


@app.errorhandler(Exception)
def handle_error(e):
    if isinstance(e, HTTPException):
        if request.path.startswith('/api/'):
            return jsonify({'error': e.description}), e.code
        return e
    log.exception('Unhandled error on %s %s', request.method, request.path)
    return jsonify({'error': 'Internal server error'}), 500


def unknown_slide():
    return jsonify({'error': 'Unknown slide'}), 404


# -- Routes --------------------------------------------------------------------

@app.route('/')
def index():
    return render_template(
        'index.html',
        label_options=json.dumps(LABEL_OPTIONS),
        label_cols=json.dumps(LABEL_COLS),
        label_int=json.dumps(LABEL_INT),
        label_str=json.dumps(LABEL_STR),
        max_slide_dim=ARGS.max_slide_dim,
    )


@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok', 'slides': len(SLIDE_DIRS), 'storage': get_store().backend})


@app.route('/api/config')
def api_config():
    return jsonify({
        'storage': get_store().backend,
        'max_slide_dim': ARGS.max_slide_dim,
        'has_checkpoint': bool(ARGS.checkpoint and os.path.exists(ARGS.checkpoint)),
        'has_mitotic_checkpoint': bool(ARGS.mitotic_checkpoint and os.path.exists(ARGS.mitotic_checkpoint)),
        'restricted_users': ALLOWED_USERS is not None,
    })


@app.route('/api/login', methods=['POST'])
def api_login():
    username = str(json_body().get('username', '')).strip().lower()
    if not username:
        return jsonify({'error': 'Empty username'}), 400
    if not USERNAME_RE.match(username):
        return jsonify({'error': 'Use 2-32 lowercase letters, numbers, ., _, or -'}), 400
    if ALLOWED_USERS is not None and username not in ALLOWED_USERS:
        return jsonify({'error': 'Unauthorized user. Access denied.'}), 403
    created = get_store().ensure_user(username)
    session.clear()
    session['user'] = username
    session.permanent = True
    return jsonify({'username': username, 'created': created})


@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'status': 'logged_out'})


@app.route('/api/session')
def api_session():
    user = session.get('user')
    return jsonify({'username': user if user_allowed(user) else None})


@app.route('/api/slides')
def api_slides():
    username = session.get('user') if user_allowed(session.get('user')) else None
    result = []
    with _registry_lock:
        items = sorted(SLIDE_DIRS.items())
    for name, sd in items:
        pred_count = len(load_predictions(sd, username))
        result.append({
            'name': name,
            'labeled': len(load_gt(sd, name, username)),
            'predictions': pred_count,
            'has_pred': pred_count > 0,
            'has_umap': os.path.exists(os.path.join(sd, 'init_umap_embeddings.pkl')),
            'has_dist': os.path.exists(os.path.join(sd, 'areas_and_ratio.pkl')),
            'has_slide_img': any(os.path.exists(os.path.join(sd, n))
                                 for n in ('compressed_image.jpg', 'compressed_image.png')),
        })
    return jsonify(result)


@app.route('/api/upload_slide', methods=['POST'])
def api_upload_slide():
    """Accept a raw slide image (<= max_slide_dim), validate it, and launch preprocessing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    orig_name = (f.filename or '').strip()
    if not orig_name:
        return jsonify({'error': 'Empty filename'}), 400

    stem, ext = os.path.splitext(orig_name)
    ext_clean = ext.lower().lstrip('.')
    if ext_clean not in UPLOAD_EXTS:
        return jsonify({'error': f'Unsupported file type .{ext_clean}. Use PNG, JPG, or TIFF.'}), 400

    custom_name = (request.form.get('slide_name') or '').strip()
    raw_name = safe_name(custom_name or stem).strip('._-') or 'uploaded_slide'

    work_dir = os.path.abspath(ARGS.slides_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Ensure unique slide name
    slide_name = raw_name
    suffix = 2
    with _registry_lock:
        while slide_name in SLIDE_DIRS or os.path.exists(os.path.join(work_dir, slide_name)):
            slide_name = f'{raw_name}_{suffix}'
            suffix += 1

    raw_img_path = os.path.join(work_dir, f'{slide_name}.{ext_clean}')
    f.save(raw_img_path)

    # Validate image integrity and dimensions before queuing preprocessing
    try:
        with Image.open(raw_img_path) as im:
            w, h = im.size
    except Exception:
        if os.path.exists(raw_img_path):
            os.remove(raw_img_path)
        return jsonify({'error': 'Uploaded file is not a valid image'}), 400

    if w > ARGS.max_slide_dim or h > ARGS.max_slide_dim:
        os.remove(raw_img_path)
        return jsonify({
            'error': f'Image is {w}x{h} px, exceeding the {ARGS.max_slide_dim}x{ARGS.max_slide_dim} px limit.'
        }), 400

    job_id = secrets.token_hex(8)
    job = ProgressJob()
    with _preprocess_lock:
        _preprocess_jobs[job_id] = job
    threading.Thread(
        target=run_preprocess,
        args=(job, slide_name, raw_img_path, work_dir),
        daemon=True,
    ).start()
    return jsonify({'job_id': job_id, 'slide': slide_name, 'width': w, 'height': h})


@app.route('/api/preprocess_stream')
def api_preprocess_stream():
    job_id = request.args.get('job_id', '')
    with _preprocess_lock:
        job = _preprocess_jobs.get(job_id)
    return _sse_stream(job)


@app.route('/api/tiles')
def api_tiles():
    slide_name = request.args.get('slide', '')
    view = request.args.get('view', 'labeled')
    if slide_name not in SLIDE_DIRS:
        return unknown_slide()
    filters = parse_filters(request.args.get('filters', '{}'))
    username = current_user()
    sd = SLIDE_DIRS[slide_name]
    cache = get_slide_cache(slide_name)
    by_tile: dict[int, dict] = {}
    if view in ('pred', 'all'):
        for t in get_prediction_tiles(cache, sd, slide_name, username, filters):
            by_tile[t['tile_num']] = t
    if view in ('labeled', 'all'):
        for t in get_labeled_tiles(cache, sd, slide_name, username, filters):
            by_tile[t['tile_num']] = t
    tiles = sorted(by_tile.values(),
                   key=lambda t: (0 if t['source'] == 'gt' else 1, t['tile_num']))
    return jsonify(tiles)


@app.route('/api/umap')
def api_umap():
    slide_name = request.args.get('slide', '')
    if slide_name not in SLIDE_DIRS:
        return unknown_slide()
    username = current_user()
    sd = SLIDE_DIRS[slide_name]
    cache = get_slide_cache(slide_name)
    tile_nums = cache['tile_nums']
    xy = load_umap_embeddings(username, slide_name, sd, cache)
    marked, associated = load_annotation_state(username, slide_name, sd, cache)
    dist = load_distributions(sd) or {}

    def dist_values(key):
        return take_by_tile_num(dist[key], tile_nums).tolist() if key in dist else []

    return jsonify({
        'x': xy[:, 0].tolist(), 'y': xy[:, 1].tolist(),
        'marked': marked,
        'colors': ['orange' if m else 'royalblue' for m in marked],
        'associated_data': associated,
        'cell_sizes':    dist_values('C'),
        'nuclear_sizes': dist_values('N'),
        'nc_ratios':     dist_values('N/C'),
    })


@app.route('/api/umap_static')
def api_umap_static():
    slide_name = request.args.get('slide', '')
    if slide_name not in SLIDE_DIRS:
        return unknown_slide()
    cache = get_slide_cache(slide_name)
    tile_nums = cache['tile_nums']
    return jsonify({
        'tile_nums':        tile_nums,
        'imgs':             [cache['tiles1'].get(tn, '') for tn in tile_nums],
        'imgs_pane':        [cache['tiles2'].get(tn, '') for tn in tile_nums],
        'desc':             [f'Tile {tn}' for tn in tile_nums],
        'manual_tile_nums': load_manual_tile_nums(SLIDE_DIRS[slide_name]),
    })


@app.route('/api/distributions')
def api_distributions():
    slide_name = request.args.get('slide', '')
    if slide_name not in SLIDE_DIRS:
        return unknown_slide()
    dist = load_distributions(SLIDE_DIRS[slide_name])
    if not dist:
        return jsonify({'available': False})

    def hist(arr):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) == 0:
            return {'hist': [], 'edges': []}
        h, e = np.histogram(arr, bins=50, density=True)
        return {'hist': h.tolist(), 'edges': e.tolist()}

    return jsonify({
        'available': True,
        'cell':     hist(dist.get('C', [])),
        'nuclear':  hist(dist.get('N', [])),
        'nc_ratio': hist(dist.get('N/C', [])),
        'image_dimensions': dist.get('image_dimensions', {}),
    })


@app.route('/api/centroids')
def api_centroids():
    slide_name = request.args.get('slide', '')
    if slide_name not in SLIDE_DIRS:
        return unknown_slide()
    return jsonify(load_centroids(SLIDE_DIRS[slide_name]))


@app.route('/api/versions')
def api_versions():
    slide_name = request.args.get('slide', '')
    if slide_name not in SLIDE_DIRS:
        return jsonify([])
    return jsonify(get_store().list_versions(current_user(), slide_name, MAX_VERSIONS_LISTED))


@app.route('/api/upload', methods=['POST'])
def api_upload():
    data = json_body()
    slide_name = data.get('slide', '')
    if slide_name not in SLIDE_DIRS:
        return unknown_slide()
    state = data.get('state') if isinstance(data.get('state'), dict) else {}
    emb = state.get('embeddingXY') if isinstance(state.get('embeddingXY'), dict) else {}
    marked = emb.get('marked') if isinstance(emb.get('marked'), list) else []
    tile_nums = state_tile_nums(state, len(marked), get_slide_cache(slide_name))
    if tile_nums is None:
        return jsonify({'error': 'Slide tiles changed since this page was loaded. '
                                 'Reload the slide and try again.'}), 409
    marked_count = sum(1 for m in marked if m)
    saved = get_store().save_version(
        current_user(), slide_name, tile_nums, marked_count,
        emb, state.get('associated_data', {}), state.get('actions_log', []),
    )
    return jsonify({'status': 'uploaded', **saved})


@app.route('/api/load_version', methods=['POST'])
def api_load_version():
    data = json_body()
    slide_name = data.get('slide', '')
    if slide_name not in SLIDE_DIRS:
        return unknown_slide()
    v = get_store().get_version(str(data.get('version_id', '')), current_user(), slide_name)
    if not v:
        return jsonify({'error': 'Not found'}), 404
    marked, associated = unpack_version(v, get_slide_cache(slide_name))
    return jsonify({
        'marked': marked,
        'associated_data': associated,
        'colors': ['orange' if m else 'royalblue' for m in marked],
        'timestamp': v['timestamp'],
    })


@app.route('/api/save', methods=['POST'])
def api_save():
    data = json_body()
    slide_name = data.get('slide', '')
    if slide_name not in SLIDE_DIRS:
        return unknown_slide()
    try:
        tile_num = int(data.get('tile_num'))
    except (TypeError, ValueError):
        return jsonify({'error': 'tile_num required'}), 400
    labels = data.get('labels')
    if not isinstance(labels, dict):
        return jsonify({'error': 'labels required'}), 400
    clean_labels = {}
    for col in LABEL_COLS:
        val = labels.get(col, 'None')
        if val not in LABEL_OPTIONS[col]:
            return jsonify({'error': f'Invalid label {col}={val!r}'}), 400
        clean_labels[col] = val

    username = current_user()
    with _slide_locks[slide_name]:
        if tile_num not in get_slide_cache(slide_name)['tile_to_idx']:
            return jsonify({'error': f'Unknown tile {tile_num}'}), 404
        path = save_labels_csv(SLIDE_DIRS[slide_name], username, tile_num, clean_labels)
    get_store().save_user_annotation(username, slide_name, tile_num, clean_labels)
    return jsonify({'status': 'saved', 'csv': path})


@app.route('/api/add_cell', methods=['POST'])
def api_add_cell():
    data = json_body()
    slide_name = data.get('slide', '')
    if slide_name not in SLIDE_DIRS:
        return unknown_slide()
    polygon = data.get('polygon')
    if not isinstance(polygon, list) or len(polygon) < 3:
        return jsonify({'error': 'Polygon requires at least 3 points'}), 400
    if len(polygon) > MAX_POLYGON_POINTS:
        return jsonify({'error': 'Polygon has too many points'}), 400
    try:
        points = [[float(p[0]), float(p[1])] for p in polygon]
    except (TypeError, ValueError, IndexError, KeyError):
        return jsonify({'error': 'Polygon points must be [x, y] numbers'}), 400
    if not np.isfinite(np.asarray(points)).all():
        return jsonify({'error': 'Polygon points must be finite'}), 400
    with _slide_locks[slide_name]:
        result = add_manual_cell(SLIDE_DIRS[slide_name], points)
        if 'error' in result:
            return jsonify(result), 500
        build_slide_cache(slide_name)
    log.info('Manual cell added: user=%s slide=%s tile=%s',
             current_user(), slide_name, result.get('tile_num'))
    return jsonify(result)


@app.route('/api/delete_cell', methods=['POST'])
def api_delete_cell():
    data = json_body()
    slide_name = data.get('slide', '')
    if slide_name not in SLIDE_DIRS:
        return unknown_slide()
    try:
        tile_num = int(data.get('tile_num'))
    except (TypeError, ValueError):
        return jsonify({'error': 'tile_num required'}), 400
    with _slide_locks[slide_name]:
        result = delete_manual_cell(SLIDE_DIRS[slide_name], tile_num)
        if 'error' in result:
            return jsonify(result), 400
        build_slide_cache(slide_name)
    get_store().delete_user_annotation(slide_name, tile_num)
    log.info('Manual cell deleted: user=%s slide=%s tile=%s', current_user(), slide_name, tile_num)
    return jsonify(result)


@app.route('/api/recluster', methods=['POST'])
def api_recluster():
    data = json_body()
    slide_name = data.get('slide', '')
    if slide_name not in SLIDE_DIRS:
        return unknown_slide()
    if not ARGS.checkpoint or not os.path.exists(ARGS.checkpoint):
        return jsonify({'error': 'Base model checkpoint (multihead_classifier.pth) not found on server.'}), 503
    username = current_user()
    key = (username, slide_name)
    with _jobs_lock:
        running = _jobs.get(key)
        if running is not None and not running.done:
            return jsonify({'status': 'started'})

    state = data.get('state') if isinstance(data.get('state'), dict) else {}
    cache = get_slide_cache(slide_name)
    n = len(cache['tile_nums'])
    marked = state.get('marked') if isinstance(state.get('marked'), list) else []
    assoc = state.get('associated_data') if isinstance(state.get('associated_data'), dict) else {}
    if not marked:
        marked = [0] * n
    tiles_of_state = state_tile_nums(state, len(marked), cache)
    if tiles_of_state is None:
        return jsonify({'error': 'Slide tiles changed since this page was loaded. '
                                 'Reload the slide and try again.'}), 409

    marked_now = [1 if m else 0 for m in place_by_tile_num(marked, tiles_of_state, cache)]
    codes = {}
    for c in LABEL_COLS:
        values = (list(assoc.get(c) or []) + [0] * len(marked))[:len(marked)]
        codes[c] = place_by_tile_num(values, tiles_of_state, cache, len(LABEL_OPTIONS[c]) - 1)
    labels_list = [''.join(str(codes[c][i]) for c in LABEL_COLS) for i in range(n)]
    probs_list = ['100' if m else '0' for m in marked_now]
    get_store().save_features_assigned(
        username, slide_name,
        list(cache['tile_nums']),
        list(range(n)),
        labels_list,
        probs_list,
    )

    with _jobs_lock:
        running = _jobs.get(key)
        if running is not None and not running.done:
            return jsonify({'status': 'started'})
        job = ProgressJob()
        _jobs[key] = job
    threading.Thread(
        target=run_recluster,
        args=(job, username, slide_name, SLIDE_DIRS[slide_name]),
        daemon=True,
    ).start()
    return jsonify({'status': 'started'})


@app.route('/api/recluster_stream')
def api_recluster_stream():
    slide_name = request.args.get('slide', '')
    with _jobs_lock:
        job = _jobs.get((current_user(), slide_name))
    return _sse_stream(job)


@app.route('/img')
def serve_img():
    path = unquote(request.args.get('path', ''))
    if not path:
        return Response('Not found', status=404)
    real = os.path.realpath(path)
    with _registry_lock:
        roots = list(SLIDE_ROOTS)
    if not any(real.startswith(root + os.sep) for root in roots):
        return Response('Forbidden', status=403)
    mime = IMAGE_MIME.get(os.path.splitext(real)[1].lower().lstrip('.'))
    if mime is None:
        return Response('Forbidden', status=403)
    if not os.path.isfile(real):
        return Response('Not found', status=404)
    return send_file(real, mimetype=mime, conditional=True, max_age=IMAGE_MAX_AGE)


@app.route('/api/slide_image')
def api_slide_image():
    slide_name = request.args.get('slide', '')
    if slide_name not in SLIDE_DIRS:
        return unknown_slide()
    sd = SLIDE_DIRS[slide_name]
    candidates = [
        ('compressed_image.jpg', 'RAW'),
        ('extracted_cells.jpg',  'CELLS'),
        ('overlay_cells.jpg',    'OVERLAY'),
        ('semantic_cells.jpg',   'MASK'),
    ]
    images = [{'label': label, 'path': os.path.join(sd, name)}
              for name, label in candidates if os.path.exists(os.path.join(sd, name))]
    if not images:
        return jsonify({'available': False})
    return jsonify({'available': True, 'images': images})


# -- Entry point ---------------------------------------------------------------

def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, 'annotator_server.log'),
        maxBytes=10 * 1024 * 1024, backupCount=5,
    )
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def load_secret_key() -> str:
    key = os.environ.get('ANNOTATOR_SECRET_KEY')
    if key:
        return key
    if os.path.exists(SECRET_KEY_FILE):
        with open(SECRET_KEY_FILE) as f:
            key = f.read().strip()
        if key:
            return key
    key = secrets.token_hex(32)
    fd = os.open(SECRET_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(key)
    return key


def register_initial_slides(explicit_paths: list[str], slides_dir: str) -> None:
    os.makedirs(slides_dir, exist_ok=True)
    seen_dirs = set()

    for path in explicit_paths:
        slide_dir = os.path.abspath(path)
        if not os.path.isdir(slide_dir):
            raise SystemExit(f'Slide directory not found: {slide_dir}')
        name = os.path.basename(slide_dir.rstrip(os.sep))
        if name in SLIDE_DIRS:
            raise SystemExit(f'Duplicate slide name "{name}": {SLIDE_DIRS[name]} and {slide_dir}')
        register_slide_dynamic(slide_dir, name)
        seen_dirs.add(os.path.realpath(slide_dir))

    for entry in sorted(os.scandir(slides_dir), key=lambda e: e.name):
        if not entry.is_dir() or entry.name.startswith('.'):
            continue
        real = os.path.realpath(entry.path)
        if real in seen_dirs:
            continue
        if is_preprocessed_slide_dir(entry.path):
            if entry.name not in SLIDE_DIRS:
                register_slide_dynamic(entry.path, entry.name)
                seen_dirs.add(real)


def parse_args(argv=None):
    env = os.environ.get
    p = argparse.ArgumentParser(description='Annotate Any Cell - Histopathology annotation server')
    p.add_argument('--slides', nargs='*', default=[],
                   help='Optional explicit paths to preprocessed slide directories')
    p.add_argument('--slides-dir', default=env('SLIDES_DIR', os.path.join(BASE_DIR, 'slides')),
                   help='Directory auto-scanned for preprocessed slides and used for uploads (default: ./slides)')
    p.add_argument('--storage', default=env('STORAGE_BACKEND', 'auto'),
                   choices=['auto', 'sqlite', 'mongo'],
                   help='Storage backend: auto (MongoDB if configured, else SQLite), sqlite, or mongo')
    p.add_argument('--sqlite-path', dest='sqlite_path',
                   default=env('SQLITE_PATH', os.path.join(BASE_DIR, 'data', 'annotator.db')),
                   help='SQLite database file path (used when storage=sqlite/auto without MongoDB)')
    p.add_argument('--mongo_uri',      default=env('MONGO_URI', ''))
    p.add_argument('--mongo_username', default=env('MONGO_USERNAME', ''))
    p.add_argument('--mongo_password', default=env('MONGO_PASSWORD', ''))
    p.add_argument('--mongo_host',     default=env('MONGO_HOST', ''))
    p.add_argument('--mongo_db',       default=env('MONGO_DB', 'annotator_db'))
    p.add_argument('--allowed-users',  default=env('ALLOWED_USERS', ''),
                   help='Comma-separated username allowlist (leave empty for open login)')
    p.add_argument('--checkpoint',         default=env('CHECKPOINT', os.path.join(BASE_DIR, 'weights', 'multihead_classifier.pth')))
    p.add_argument('--mitotic_checkpoint', default=env('MITOTIC_CHECKPOINT', os.path.join(BASE_DIR, 'weights', 'mitotic_classifier.pth')))
    p.add_argument('--mitotic_threshold',  type=float, default=float(env('MITOTIC_THRESHOLD', '0.9')))
    p.add_argument('--max-slide-dim',      type=int,   default=int(env('MAX_SLIDE_DIM', '7000')))
    p.add_argument('--port', type=int, default=int(env('PORT', '5006')))
    p.add_argument('--host', default=env('HOST', '127.0.0.1'))
    p.add_argument('--threads', type=int, default=16)
    args = p.parse_args(argv)
    args.slides_dir = os.path.abspath(args.slides_dir)
    args.sqlite_path = os.path.abspath(args.sqlite_path)
    if args.checkpoint:
        args.checkpoint = os.path.abspath(args.checkpoint)
    if args.mitotic_checkpoint:
        args.mitotic_checkpoint = os.path.abspath(args.mitotic_checkpoint)
    return args


def main():
    global ARGS, ALLOWED_USERS
    ARGS = parse_args()
    setup_logging()
    app.secret_key = load_secret_key()

    raw_allowed = [u.strip().lower() for u in ARGS.allowed_users.split(',') if u.strip()]
    ALLOWED_USERS = frozenset(raw_allowed) if raw_allowed else None

    register_initial_slides(ARGS.slides, ARGS.slides_dir)

    if not os.path.exists(os.path.join(BASE_DIR, 'update_clusters.py')):
        raise SystemExit('update_clusters.py must sit next to annotator_server.py')
    if not os.path.exists(ARGS.checkpoint):
        log.warning('Base checkpoint not found: %s ("Update Clusters" and initial AI predictions disabled)',
                    ARGS.checkpoint)
    if ARGS.mitotic_checkpoint and not os.path.exists(ARGS.mitotic_checkpoint):
        log.warning('Mitotic checkpoint not found: %s (mitotic refinement skipped)',
                    ARGS.mitotic_checkpoint)

    store = get_store()
    store.ping()
    atexit.register(terminate_children)
    atexit.register(store.close)

    from waitress import serve
    log.info(
        'Annotation server on http://%s:%d | storage=%s | slides=%d | auth=%s',
        ARGS.host, ARGS.port, store.backend, len(SLIDE_DIRS),
        'allowlist' if ALLOWED_USERS is not None else 'open',
    )
    serve(app, host=ARGS.host, port=ARGS.port, threads=ARGS.threads)


if __name__ == '__main__':
    main()
