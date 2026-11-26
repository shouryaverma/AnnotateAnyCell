"""
update_celladder.py - add / delete manually outlined cells for a preprocessed slide.

Invariants kept for the rest of the pipeline:
  * line i of centroids.csv, element i of every array in areas_and_ratio.pkl and
    row i of init_umap_embeddings.pkl all describe tile i (image_tile_i.*)
  * a tile number is never reused: deleting a cell blanks its slots, and the next
    manual cell always gets a number above every number ever issued. Cached
    features, browser-cached tile images and stored annotations therefore never
    end up pointing at a different cell.

Callers serialise access per slide (annotator_server.py holds the slide lock).
"""
import os
import re
import csv
import glob
import json
import pickle
import threading

import cv2
import numpy as np

ROTATIONS = (90, 180, 270)

_image_cache = {'key': None, 'image': None}
_image_cache_lock = threading.Lock()


def _tmp_path(path):
    return f'{path}.tmp.{os.getpid()}.{threading.get_ident()}'


def _atomic_write(path, write_fn, mode='w', **open_kwargs):
    tmp = _tmp_path(path)
    try:
        with open(tmp, mode, **open_kwargs) as f:
            write_fn(f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _atomic_pickle(path, obj):
    _atomic_write(path, lambda f: pickle.dump(obj, f), mode='wb')


def _load_slide_image(path):
    """Decoded slide image, cached for the most recently used slide. Treat as read-only."""
    key = (path, os.stat(path).st_mtime_ns)
    with _image_cache_lock:
        if _image_cache['key'] == key:
            return _image_cache['image']
    image = cv2.imread(path)
    if image is not None:
        with _image_cache_lock:
            _image_cache['key'], _image_cache['image'] = key, image
    return image


def _read_centroid_lines(slide_dir):
    path = os.path.join(slide_dir, 'centroids.csv')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line if line.endswith('\n') else line + '\n' for line in f.readlines()]


def _write_centroid_lines(slide_dir, lines):
    _atomic_write(os.path.join(slide_dir, 'centroids.csv'), lambda f: f.writelines(lines))


def _max_tile_num(slide_dir):
    d = os.path.join(slide_dir, 'tiles1', 'nonbox')
    best = -1
    if os.path.isdir(d):
        for name in os.listdir(d):
            nums = re.findall(r'\d+', os.path.splitext(name)[0])
            if nums:
                best = max(best, int(nums[-1]))
    return best


def _next_tile_num(slide_dir):
    return max(len(_read_centroid_lines(slide_dir)), _max_tile_num(slide_dir) + 1)


def _load_manual_nums(slide_dir):
    path = os.path.join(slide_dir, 'manual_cells.json')
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _save_manual_nums(slide_dir, nums):
    _atomic_write(os.path.join(slide_dir, 'manual_cells.json'), lambda f: json.dump(nums, f))


def _remove_from_annotation_csvs(slide_dir, tile_num):
    """Remove deleted manual tile from every annotation_data_*.csv in slide_dir."""
    for path in sorted(glob.glob(os.path.join(slide_dir, 'annotation_data_*.csv'))):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            desc_key = next((k for k in fieldnames if k.strip().lower() == 'desc'), 'desc')
            rows, removed = [], False
            for row in reader:
                m = re.search(r'(\d+)', str(row.get(desc_key, '')))
                if m and int(m.group(1)) == tile_num:
                    removed = True
                    continue
                rows.append(row)
        if not removed:
            continue

        def write(f, _fieldnames=fieldnames, _rows=rows):
            writer = csv.DictWriter(f, fieldnames=_fieldnames)
            writer.writeheader()
            writer.writerows(_rows)

        _atomic_write(path, write, newline='')


def _crop_with_padding(image, cx, cy, half_size):
    h, w = image.shape[:2]
    x1, y1 = cx - half_size, cy - half_size
    x2, y2 = cx + half_size, cy + half_size
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    full = half_size * 2
    out = np.zeros((full, full) + image.shape[2:], dtype=image.dtype)
    if x2c > x1c and y2c > y1c:
        out[y1c - y1: y2c - y1, x1c - x1: x2c - x1] = image[y1c:y2c, x1c:x2c]
    return out


def _polygon_mask(polygon_local, size):
    mask = np.zeros((size, size), dtype=np.uint8)
    pts = np.round(np.asarray(polygon_local, dtype=np.float64)).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _masked_on_white(crop, mask):
    out = np.full_like(crop, 255, dtype=np.uint8)
    keep = mask == 255
    out[keep] = crop[keep]
    return out


def _write_image(path, image, params):
    if not cv2.imwrite(path, image, params):
        raise IOError(f'Failed to write {path}')


def add_manual_cell(slide_dir, polygon_coords):
    img_path = os.path.join(slide_dir, 'compressed_image.jpg')
    if not os.path.exists(img_path):
        img_path = os.path.join(slide_dir, 'compressed_image.png')
    if not os.path.exists(img_path):
        return {'error': 'compressed_image.jpg not found'}
    image = _load_slide_image(img_path)
    if image is None:
        return {'error': 'Failed to load slide image'}

    pts = np.asarray(polygon_coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2 or not np.isfinite(pts).all():
        return {'error': 'Polygon requires at least 3 finite [x, y] points'}
    pts = pts[:, :2]
    cx = int(round(pts[:, 0].mean()))
    cy = int(round(pts[:, 1].mean()))
    height, width = image.shape[:2]
    if not (0 <= cx < width and 0 <= cy < height):
        return {'error': 'Polygon lies outside the slide image'}

    tile_num = _next_tile_num(slide_dir)
    jpg = [cv2.IMWRITE_JPEG_QUALITY, 90]
    png = [cv2.IMWRITE_PNG_COMPRESSION, 9]

    crop128 = _crop_with_padding(image, cx, cy, 64)
    local128 = pts - np.array([cx - 64, cy - 64], dtype=np.float64)
    cell_mask128 = _polygon_mask(local128, 128)

    cell_area = float(cv2.contourArea(local128.astype(np.float32)))
    if cell_area <= 0:
        cell_area = float(np.sum(cell_mask128 > 0))
    if cell_area <= 0:
        return {'error': 'Polygon has zero area'}
    nuclei_area = float(np.sum(cell_mask128 > 0)) or cell_area
    nc_ratio = nuclei_area / cell_area

    t1 = {name: os.path.join(slide_dir, 'tiles1', name)
          for name in ('nonbox', 'raw', 'semantic', 'nuclei')}
    t2 = {name: os.path.join(slide_dir, 'tiles2', name)
          for name in ('nonbox', 'semantic', 'nuclei')}
    for d in list(t1.values()) + list(t2.values()):
        os.makedirs(d, exist_ok=True)

    tile1_path = os.path.join(t1['nonbox'], f'image_tile_{tile_num}.jpg')
    _write_image(tile1_path, crop128, jpg)
    _write_image(os.path.join(t1['raw'], f'image_tile_{tile_num}.png'), crop128, png)
    _write_image(os.path.join(t1['semantic'], f'image_tile_{tile_num}.png'), cell_mask128, png)
    _write_image(os.path.join(t1['nuclei'], f'image_tile_{tile_num}.png'),
                 _masked_on_white(crop128, cell_mask128), png)

    crop512 = _crop_with_padding(image, cx, cy, 256)
    local512 = pts - np.array([cx - 256, cy - 256], dtype=np.float64)
    cell_mask512 = _polygon_mask(local512, 512)

    tile2_path = os.path.join(t2['nonbox'], f'image_tile_{tile_num}.jpg')
    _write_image(tile2_path, crop512, jpg)
    _write_image(os.path.join(t2['semantic'], f'image_tile_{tile_num}.png'), cell_mask512, png)
    _write_image(os.path.join(t2['nuclei'], f'image_tile_{tile_num}.png'),
                 _masked_on_white(crop512, cell_mask512), png)

    lines = _read_centroid_lines(slide_dir)
    lines.extend('\n' for _ in range(tile_num - len(lines)))
    lines.append(f'[{float(cx)}, {float(cy)}]\n')
    _write_centroid_lines(slide_dir, lines)

    pkl_path = os.path.join(slide_dir, 'areas_and_ratio.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            dist = pickle.load(f)
        for key, value in (('C', cell_area), ('N', nuclei_area), ('N/C', nc_ratio)):
            arr = np.array(dist.get(key, []), dtype=np.float64)
            if tile_num >= len(arr):
                arr = np.concatenate([arr, np.zeros(tile_num + 1 - len(arr))])
            arr[tile_num] = value
            dist[key] = arr
        _atomic_pickle(pkl_path, dist)

    emb_path = os.path.join(slide_dir, 'init_umap_embeddings.pkl')
    umap_xy = [0.0, 0.0]
    if os.path.exists(emb_path):
        with open(emb_path, 'rb') as f:
            embeddings = np.array(pickle.load(f), dtype=np.float64).reshape(-1, 2)
        valid = embeddings[np.isfinite(embeddings).all(axis=1)]
        if len(valid) > 0:
            umap_xy = [float(v) for v in valid.mean(axis=0)]
        if tile_num >= len(embeddings):
            pad = np.full((tile_num + 1 - len(embeddings), 2), np.nan)
            embeddings = np.vstack([embeddings, pad])
        embeddings[tile_num] = umap_xy
        _atomic_pickle(emb_path, embeddings)

    manual_nums = _load_manual_nums(slide_dir)
    if tile_num not in manual_nums:
        manual_nums.append(tile_num)
    _save_manual_nums(slide_dir, manual_nums)

    return {
        'tile_num':      tile_num,
        'centroid':      {'x': float(cx), 'y': float(cy)},
        'umap_xy':       umap_xy,
        'cell_area':     cell_area,
        'nuclei_area':   nuclei_area,
        'nc_ratio':      nc_ratio,
        'img_path':      tile1_path,
        'img_pane_path': tile2_path,
    }


def delete_manual_cell(slide_dir, tile_num):
    """
    Delete a manually added cell. Only tiles listed in manual_cells.json can be
    deleted. Index-aligned files keep a blank slot so every other tile keeps its number.
    """
    manual_nums = _load_manual_nums(slide_dir)
    if tile_num not in manual_nums:
        return {'error': f'Tile {tile_num} is not a manually added cell'}

    lines = _read_centroid_lines(slide_dir)
    if tile_num < len(lines):
        lines[tile_num] = '\n'
        _write_centroid_lines(slide_dir, lines)

    pkl_path = os.path.join(slide_dir, 'areas_and_ratio.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            dist = pickle.load(f)
        for key in ('C', 'N', 'N/C'):
            if key in dist and tile_num < len(dist[key]):
                arr = np.array(dist[key], dtype=np.float64)
                arr[tile_num] = 0.0
                dist[key] = arr
        _atomic_pickle(pkl_path, dist)

    emb_path = os.path.join(slide_dir, 'init_umap_embeddings.pkl')
    if os.path.exists(emb_path):
        with open(emb_path, 'rb') as f:
            embeddings = np.array(pickle.load(f), dtype=np.float64)
        if tile_num < len(embeddings):
            embeddings[tile_num] = np.nan
            _atomic_pickle(emb_path, embeddings)

    stale = [os.path.join(slide_dir, tier, subdir, f'image_tile_{tile_num}.{ext}')
             for tier, subdir, ext in (
                 ('tiles1', 'nonbox',   'jpg'),
                 ('tiles1', 'raw',      'png'),
                 ('tiles1', 'semantic', 'png'),
                 ('tiles1', 'nuclei',   'png'),
                 ('tiles2', 'nonbox',   'jpg'),
                 ('tiles2', 'semantic', 'png'),
                 ('tiles2', 'nuclei',   'png'),
             )]
    cache_dir = os.path.join(slide_dir, 'tiles1', 'virchow2_cache')
    stale.append(os.path.join(cache_dir, f'{tile_num}.npy'))
    stale.extend(os.path.join(cache_dir, f'{tile_num}_r{rot}.npy') for rot in ROTATIONS)
    for p in stale:
        if os.path.exists(p):
            os.remove(p)

    _remove_from_annotation_csvs(slide_dir, tile_num)

    manual_nums.remove(tile_num)
    _save_manual_nums(slide_dir, manual_nums)

    return {'status': 'deleted', 'tile_num': tile_num}
