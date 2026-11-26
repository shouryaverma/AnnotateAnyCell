#!/usr/bin/env python3
"""
preprocess.py - Segment a histopathology slide image (up to 7000x7000 by default)
and build all files required by the Annotate Any Cell web application.

Can be run standalone from the CLI or spawned by annotator_server.py when a user
uploads a new slide directly through the web UI. Emits progress lines in the format:
    "Progress: 42.50% | message"

Input : <work_dir>/<slide_name>.png | .jpg | .jpeg | .tif | .tiff  (or --image-path)
Output: <work_dir>/<slide_name>/
    compressed_image.jpg, semantic_*.jpg, extracted_*.jpg, overlay_*.jpg
    centroids.csv              line i  = centroid of tile i
    areas_and_ratio.pkl        index i = cell area, nuclear area, N/C of tile i
    tiles1/{nonbox,raw,semantic,nuclei}/image_tile_i.*   128x128
    tiles2/nonbox/image_tile_i.jpg                       512x512
    init_umap_embeddings.pkl   row i   = initial 2-D embedding of tile i
    predictions.json           (optional, when --checkpoint is provided)
"""
import os
import sys
import pickle
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.utils.model_zoo
import umap
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_load_url = torch.utils.model_zoo.load_url
torch.utils.model_zoo.load_url = lambda *a, **k: _load_url(*a, **{**k, 'weights_only': True})

Image.MAX_IMAGE_PIXELS = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, 'logs')

MIN_CELL_AREA = 1000
MAX_CELL_AREA = 10000
MAX_SLIDE_DIM = 7000
UMAP_THUMB    = 32
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 90]
PNG  = [cv2.IMWRITE_PNG_COMPRESSION, 9]

logger = logging.getLogger('preprocess')


def progress(pct: float, msg: str = ''):
    print(f'Progress: {pct:.2f}% | {msg}', flush=True)


def scaled_progress(start: float, end: float, label: str):
    def report(done: int, total: int):
        progress(start + done / max(total, 1) * (end - start), f'{label} {done}/{total}')
    return report


def setup_logging(slide_name: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, f'preprocess_{slide_name}.log')),
    ]
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(fmt)
        root.addHandler(handler)


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def log_gpu_memory():
    if torch.cuda.is_available():
        logger.info(
            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
            f"reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
        )


def _import_cellpose():
    import cellpose
    from cellpose import models, utils
    ver = getattr(cellpose, '__version__', '')
    major = int(ver.split('.')[0]) if ver and ver.split('.')[0].isdigit() else 3
    if major >= 4:
        raise RuntimeError(
            f"Detected cellpose {ver}, which removed the Cellpose 3 'cyto3' and 'nuclei' models.\n"
            "Install Cellpose 3 with: pip install 'cellpose>=3.0.0,<4.0.0'"
        )
    return models, utils


def find_slide_image(base_path: str) -> str | None:
    for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
        cand = base_path + ext
        if os.path.exists(cand):
            return cand
    return None


def load_rgb_bgr_uint8(image_path: str, max_dim: int = MAX_SLIDE_DIM) -> np.ndarray:
    """Load PNG/JPG/TIFF safely as 3-channel uint8 BGR and enforce max_dim."""
    with Image.open(image_path) as im:
        w, h = im.size
        if max_dim > 0 and (w > max_dim or h > max_dim):
            raise ValueError(
                f"Slide image is {w}x{h} px, which exceeds the maximum supported size of "
                f"{max_dim}x{max_dim} px. Please crop or downsample to <= {max_dim}x{max_dim} px."
            )
        if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):
            arr = np.asarray(im, dtype=np.float32)
            lo, hi = np.percentile(arr, (0.5, 99.5))
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
            im = Image.fromarray(arr)
        rgb = np.asarray(im.convert('RGB'), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def run_cellpose(img, model_type: str, **eval_kwargs):
    """Cellpose instance masks; retries on CPU if the GPU run fails."""
    models, _ = _import_cellpose()
    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(
        model_type=model_type, gpu=use_gpu,
        device=torch.device('cuda' if use_gpu else 'cpu'),
    )
    try:
        masks, *_ = model.eval(img, **eval_kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not use_gpu:
            raise
        logger.warning(f"GPU processing failed: {e}. Falling back to CPU.")
        del model
        clear_gpu_memory()
        model = models.CellposeModel(
            model_type=model_type, gpu=False, device=torch.device('cpu')
        )
        masks, *_ = model.eval(img, **eval_kwargs)
    return masks


def segment_image(img, model_type: str, **eval_kwargs) -> dict:
    _, utils = _import_cellpose()
    masks = np.asarray(run_cellpose(img, model_type, **eval_kwargs)).astype(np.uint32)
    return {'masks': masks, 'bounds': utils.masks_to_outlines(masks)}


def filter_segments(nuclei, bounds_nucl, cells, bounds_cell):
    cells_f = cells.copy()
    cells_f[bounds_cell.astype(bool)] = 0
    nuclei_f = nuclei.copy()
    nuclei_f[bounds_nucl.astype(bool)] = 0
    nuclei_f[cells_f == 0] = 0

    max_id       = int(cells_f.max()) if cells_f.size else 0
    has_nucleus  = np.zeros(max_id + 1, dtype=bool)
    has_nucleus[np.unique(cells_f[nuclei_f != 0])] = True
    has_nucleus[0] = True
    without_nucleus = ~has_nucleus[cells_f]

    semantic_cells  = np.where(cells_f != 0, 255, 0).astype(np.uint8)
    nucleus_pixels  = (nuclei_f != 0) | without_nucleus | ((cells_f != 0) & (nuclei != 0))
    semantic_nuclei = np.where(nucleus_pixels, 255, 0).astype(np.uint8)
    return semantic_nuclei, semantic_cells


def extract_masked(image, semantic):
    out = np.full_like(image, 255, dtype=np.uint8)
    keep = semantic == 255
    out[keep] = image[keep]
    return out


def measure_cells(semantic_cells, semantic_nuclei, min_area: int, max_area: int):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(semantic_cells, connectivity=4)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = (areas >= min_area) & (areas <= max_area)
    keep[0] = False
    nuclear = np.bincount(labels[semantic_nuclei == 255], minlength=num)
    return centroids[keep].tolist(), areas[keep].astype(np.float64), nuclear[keep].astype(np.float64)


def save_distributions(cell_area, nuclei_area, image, out_dir):
    ratio = np.divide(nuclei_area, cell_area, out=np.zeros_like(nuclei_area), where=cell_area > 0)
    height, width = image.shape[:2]
    with open(os.path.join(out_dir, 'areas_and_ratio.pkl'), 'wb') as f:
        pickle.dump({'N/C': ratio, 'N': nuclei_area, 'C': cell_area,
                     'image_dimensions': {'width': width, 'height': height}}, f)


def save_centroids(centroids, out_dir):
    with open(os.path.join(out_dir, 'centroids.csv'), 'w') as f:
        for item in centroids:
            f.write("%s\n" % str(item))


def crop_centered(arr, cx: int, cy: int, size: int, pad_value: int):
    half = size // 2
    x1, y1 = cx - half, cy - half
    h, w = arr.shape[:2]
    xs, ys = max(0, x1), max(0, y1)
    xe, ye = min(w, x1 + size), min(h, y1 + size)
    out = np.full((size, size) + arr.shape[2:], pad_value, dtype=arr.dtype)
    if xe > xs and ye > ys:
        out[ys - y1:ye - y1, xs - x1:xe - x1] = arr[ys:ye, xs:xe]
    return out


def find_nearest_mask(mask_tile):
    contours, _ = cv2.findContours(mask_tile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_y, center_x = mask_tile.shape[0] // 2, mask_tile.shape[1] // 2
    best, best_dist = None, float('inf')
    for contour in contours:
        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        dist = np.hypot(m["m10"] / m["m00"] - center_x, m["m01"] / m["m00"] - center_y)
        if dist < best_dist:
            best, best_dist = contour, dist
    return best


def write_tiles(image, semantic_cells, extracted_cells, centroids, out_dir,
                folder_num: int, window_size: int):
    tile_path = os.path.join(out_dir, f'tiles{folder_num}')
    full_tier = window_size == 128
    subdirs = ('nonbox', 'raw', 'semantic', 'nuclei') if full_tier else ('nonbox',)
    paths = {name: os.path.join(tile_path, name) for name in subdirs}
    for d in paths.values():
        os.makedirs(d, exist_ok=True)

    half = window_size // 2
    h, w = image.shape[:2]

    def process(i):
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        image_tile = crop_centered(image, cx, cy, window_size, 0)
        cv2.imwrite(os.path.join(paths['nonbox'], f'image_tile_{i}.jpg'), image_tile, JPEG)
        if not full_tier:
            return None
        semantic_tile = crop_centered(semantic_cells, cx, cy, window_size, 0)
        cell_tile     = crop_centered(extracted_cells, cx, cy, window_size, 255)
        contour = find_nearest_mask(semantic_tile)
        if contour is not None:
            mask = np.zeros_like(semantic_tile)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            semantic_tile = cv2.bitwise_and(semantic_tile, mask)
            cell_tile[mask == 0] = 255
        cv2.imwrite(os.path.join(paths['raw'], f'image_tile_{i}.png'), image_tile, PNG)
        cv2.imwrite(os.path.join(paths['semantic'], f'image_tile_{i}.png'), semantic_tile, PNG)
        cv2.imwrite(os.path.join(paths['nuclei'], f'image_tile_{i}.png'), cell_tile, PNG)
        thumb = cv2.resize(image_tile, (UMAP_THUMB, UMAP_THUMB), interpolation=cv2.INTER_AREA)
        return thumb.reshape(-1)

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
        thumbs = list(pool.map(process, range(len(centroids))))

    with open(os.path.join(tile_path, 'leftright_coords.csv'), 'w') as f:
        for centroid in centroids:
            cx, cy = int(centroid[0]), int(centroid[1])
            f.write("%s\n" % str(((max(0, cx - half), max(0, cy - half)),
                                  (min(w, cx + half), min(h, cy + half)))))

    return np.stack(thumbs) if full_tier and thumbs else None


def initial_umap(thumbs: np.ndarray) -> np.ndarray:
    n = len(thumbs)
    if n < 4:
        return np.zeros((n, 2), dtype=np.float64)
    features = StandardScaler().fit_transform(thumbs.astype(np.float32))
    n_components = min(50, n - 1, features.shape[1])
    features = PCA(n_components=n_components, random_state=42).fit_transform(features)
    return umap.UMAP(
        n_neighbors=max(2, min(20, n - 1)),
        min_dist=1.0,
        n_components=2,
        random_state=42,
        n_jobs=1,
    ).fit_transform(features)


def main(args):
    setup_logging(args.slide_name)
    base_path  = os.path.join(os.path.abspath(args.work_dir), args.slide_name)
    output_dir = base_path

    image_path = os.path.abspath(args.image_path) if args.image_path else find_slide_image(base_path)
    if image_path is None or not os.path.exists(image_path):
        raise SystemExit(f"No .png, .jpg, .tif, or .tiff file found for: {base_path}")

    progress(2., f'Loading image {os.path.basename(image_path)}')
    image = load_rgb_bgr_uint8(image_path, max_dim=args.max_dim)
    os.makedirs(output_dir, exist_ok=True)
    h, w = image.shape[:2]
    logger.info(f'Loaded image {w}x{h}')
    log_gpu_memory()

    inverted = 255 - image
    progress(8., 'Segmenting cells (Cellpose cyto3)...')
    cell_data = segment_image(inverted, 'cyto3', batch_size=1, diameter=None,
                              flow_threshold=0.8, cellprob_threshold=-4, channels=[3, 0])
    clear_gpu_memory()
    log_gpu_memory()

    progress(32., 'Segmenting nuclei (Cellpose nuclei)...')
    nuclei_data = segment_image(inverted, 'nuclei', min_size=15, batch_size=1, diameter=None,
                                flow_threshold=0.8, channels=[3, 0])
    clear_gpu_memory()
    log_gpu_memory()
    del inverted

    progress(52., 'Filtering masks and measuring cells...')
    semantic_nuclei, semantic_cells = filter_segments(
        nuclei_data['masks'], nuclei_data['bounds'], cell_data['masks'], cell_data['bounds'])
    del nuclei_data, cell_data

    extracted_nuclei = extract_masked(image, semantic_nuclei)
    extracted_cells  = extract_masked(image, semantic_cells)

    centroids, cell_area, nuclei_area = measure_cells(
        semantic_cells, semantic_nuclei, args.min_cell_area, args.max_cell_area)
    logger.info(f'{len(centroids)} cells within [{args.min_cell_area}, {args.max_cell_area}] px')

    progress(58., f'Saving slide overlays ({len(centroids)} cells detected)...')
    cv2.imwrite(os.path.join(output_dir, 'compressed_image.jpg'), image, JPEG)
    cv2.imwrite(os.path.join(output_dir, 'semantic_nuclei.jpg'), semantic_nuclei, JPEG)
    cv2.imwrite(os.path.join(output_dir, 'semantic_cells.jpg'), semantic_cells, JPEG)
    cv2.imwrite(os.path.join(output_dir, 'extracted_cells.jpg'), extracted_cells, JPEG)
    cv2.imwrite(os.path.join(output_dir, 'extracted_nuclei.jpg'), extracted_nuclei, JPEG)
    del extracted_nuclei

    overlay = image.copy()
    overlay[semantic_nuclei == 255] = [0, 0, 255]
    cv2.imwrite(os.path.join(output_dir, 'overlay_nuclei.jpg'), overlay, JPEG)
    overlay[semantic_cells == 255] = [0, 255, 0]
    cv2.imwrite(os.path.join(output_dir, 'overlay_cells.jpg'), overlay, JPEG)
    del overlay

    save_distributions(cell_area, nuclei_area, image, output_dir)
    save_centroids(centroids, output_dir)

    progress(64., 'Writing 128x128 cell tiles...')
    thumbs = write_tiles(image, semantic_cells, extracted_cells, centroids, output_dir, 1, 128)
    progress(72., 'Writing 512x512 context tiles...')
    write_tiles(image, semantic_cells, extracted_cells, centroids, output_dir, 2, 512)

    run_pred = bool(args.checkpoint and os.path.exists(args.checkpoint) and len(centroids) > 0)
    umap_pct = 78. if run_pred else 90.
    progress(umap_pct, 'Computing initial UMAP embedding...')
    if thumbs is not None and len(thumbs) > 0:
        embeddings = initial_umap(thumbs)
    else:
        embeddings = np.zeros((0, 2), dtype=np.float64)
    with open(os.path.join(output_dir, 'init_umap_embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)

    if run_pred:
        sys.path.insert(0, BASE_DIR)
        from model_predict import predict_slide
        progress(80., 'Running initial model predictions (Virchow2 + MultiHead)...')
        try:
            n_pred = predict_slide(
                output_dir,
                checkpoint=args.checkpoint,
                batch_size=args.batch_size,
                mitotic_checkpoint=args.mitotic_checkpoint or '',
                mitotic_threshold=args.mitotic_threshold,
                on_extract_progress=scaled_progress(80., 92., 'Virchow2 features'),
                on_infer_progress=scaled_progress(92., 98., 'Inference batch'),
            )
            logger.info(f'Generated {n_pred} initial predictions')
        except Exception as e:
            logger.warning(f'Initial prediction step skipped due to error: {e}')
            progress(98., f'Slide ready (predictions skipped: {e})')

    progress(100., f'Done ({len(centroids)} cells)')
    print('completed', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess histopathology slide image.')
    parser.add_argument('--slide-name', type=str, required=True,
                        help='Output slide folder name (e.g. ExampleSlide1)')
    parser.add_argument('--work-dir', type=str, default=os.getcwd(),
                        help='Directory holding slide images and output folders')
    parser.add_argument('--image-path', type=str, default='',
                        help='Optional explicit path to the raw slide image')
    parser.add_argument('--max-dim', type=int, default=MAX_SLIDE_DIM,
                        help='Maximum allowed width/height in pixels (default: 7000)')
    parser.add_argument('--min-cell-area', type=int, default=MIN_CELL_AREA)
    parser.add_argument('--max-cell-area', type=int, default=MAX_CELL_AREA)
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Optional multihead_classifier.pth to run initial predictions')
    parser.add_argument('--mitotic-checkpoint', type=str, default='',
                        help='Optional mitotic_classifier.pth for mitotic refinement')
    parser.add_argument('--mitotic-threshold', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
