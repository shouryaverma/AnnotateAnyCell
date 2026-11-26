// ── State ──────────────────────────────────────────────────────────────────────
const S = {
  user: localStorage.getItem('ann_user') || null,
  slide: null,
  view: 'umap',
  gridMode: 'labeled',
  activeFilters: {},
  tiles: [],
  selGridIdx: null,
  selectedUmapIdx: null,
  tileStatus: new Map(),
  umap: null,
  umapStatic: null,
  tileNumToIdx: {},
  centroids: {},
  origW: 1, origH: 1,
  dist: null,
  versions: [],
  colorFeature: null,
  reclustering: false,
  uploading: false,
  slideColorFeature: null,
  slideLineThickness: 2,
  slideDimAlpha: 0,
  slideImages: [],
  slideImageIdx: 0,
  selectedSlideTileNum: null,
  pointSize: 6,
  drawingMode: false,
  polygonPoints: [],
  polygonMousePos: null,
  manualTileNums: new Set(),
  pendingUploadFile: null,
};
let currentTile = null;
let _slideRafId = null;
let _lastXRange = null;
let _lastYRange = null;

const $ = id => document.getElementById(id);
const loading = on => $('overlay').classList.toggle('on', on);

// ── API ────────────────────────────────────────────────────────────────────────
class AuthError extends Error {}

async function api(url, opts) {
  const r = await fetch(url, opts);
  if (r.status === 401) {
    setLoggedOut('session expired - log in again');
    throw new AuthError('Not logged in');
  }
  return r;
}

window.addEventListener('unhandledrejection', e => {
  if (e.reason instanceof AuthError) e.preventDefault();
});

const SLIDE_CLASS_COLORS = {
  mitotic: '#22c55e', chromatin: '#22c55e', nucleoli: '#22c55e',
  prominent: '#22c55e', shape: '#22c55e',
};
const TILE2_CROP_HALF = 256;

// ── Theme ──────────────────────────────────────────────────────────────────────
const THEMES = ['dark', 'light', 'contrast'];
const THEME_LABELS = {dark: 'DARK', light: 'LIGHT', contrast: 'HICON'};
let themeIdx = 0;

function cycleTheme() {
  themeIdx = (themeIdx + 1) % 3;
  const t = THEMES[themeIdx];
  document.documentElement.dataset.theme = t === 'dark' ? '' : t;
  $('theme-btn').textContent = THEME_LABELS[t];
  refreshPlotlyTheme();
}

function plotlyPaperBg() { return 'rgba(0,0,0,0)'; }
function plotlyFont() {
  const c = ['#e8edf5', '#000000', '#ffffff'];
  return {color: c[themeIdx], size: 10};
}

function refreshPlotlyTheme() {
  const upd = {paper_bgcolor: plotlyPaperBg(), plot_bgcolor: plotlyPaperBg(), font: plotlyFont()};
  ['umap-scatter',
   'dist-cell-umap', 'dist-nc-umap',
   'dist-cell-slide', 'dist-nc-slide',
   'dist-cell-grid', 'dist-nc-grid'].forEach(id => {
    const el = $(id);
    if (el && el.data) Plotly.relayout(id, upd);
  });
}

// ── Init ───────────────────────────────────────────────────────────────────────
async function init() {
  buildFilterBoxes();
  buildColorButtons();
  buildSlideColorButtons();
  setupUploadDropZone();
  $('umap-controls').style.display  = '';
  $('slide-controls').style.display = 'none';
  $('user-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.stopPropagation(); doLoginClick(); }
  });

  try {
    const cfg = await (await fetch('/api/config')).json();
    $('storage-badge').textContent = (cfg.storage || 'sqlite').toUpperCase();
    if (!cfg.has_checkpoint) {
      $('recluster-btn').title = 'Place multihead_classifier.pth in the app folder to enable Update Clusters';
    }
  } catch (_) {}

  await refreshSlideList();
  const r = await fetch('/api/session');
  const j = await r.json();
  if (j.username) {
    setLoggedIn(j.username, 'logged in');
    await refreshSlideList();
    await pickFirstSlide();
  } else {
    setLoggedOut();
  }
}

function updateLabeledCount() {
  const row = $('labeled-count-row');
  if (!row) return;
  if (!S.umap || !S.slide) {
    row.style.display = 'none';
    return;
  }
  const count = S.umap.marked.reduce((a, b) => a + b, 0);
  const total = S.umap.marked.length;
  row.style.display = '';
  $('labeled-count').textContent = count;
  $('total-count').textContent = total;
  $('labeled-bar').style.width = total > 0 ? `${(count / total * 100).toFixed(1)}%` : '0%';
}

// ── Login ──────────────────────────────────────────────────────────────────────
function setUserStatus(text, color) {
  $('user-status').textContent = text;
  $('user-status').style.color = color;
}

function setLoggedIn(username, status) {
  S.user = username;
  localStorage.setItem('ann_user', username);
  $('user-input').value = username;
  $('user-input').disabled = true;
  $('login-btn').textContent = 'LOGOUT';
  $('login-btn').onclick = doLogout;
  $('upload-slide-btn').disabled = S.uploading;
  setUserStatus(status, 'var(--grn)');
}

function setLoggedOut(status) {
  S.user = null;
  S.slide = null;
  S.umap = null;
  S.umapStatic = null;
  S.tiles = [];
  S.versions = [];
  S.reclustering = false;
  localStorage.removeItem('ann_user');
  loading(false);
  closePanel();
  stopSlideBgLoop();
  $('user-input').disabled = false;
  $('login-btn').textContent = 'LOGIN';
  $('login-btn').onclick = doLoginClick;
  $('recluster-btn').disabled = false;
  $('upload-slide-btn').disabled = true;
  setUserStatus(status || 'log in to load or upload a slide', 'var(--mut)');
  document.querySelectorAll('.slide-item').forEach(e => e.classList.remove('active'));
  if ($('umap-scatter').data) Plotly.purge('umap-scatter');
  if ($('slide-plotly').data) Plotly.purge('slide-plotly');
  $('slide-bg-img').style.display = 'none';
  $('grid').innerHTML = '<div class="empty"><div class="sym">◈</div>Log in to load a slide</div>';
  $('version-select').innerHTML = '<option value="">-- select version --</option>';
  updateLabeledCount();
}

async function doLoginClick() {
  const u = $('user-input').value.trim().toLowerCase();
  if (!u) return;
  const r = await fetch('/api/login', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({username: u}),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    setUserStatus(j.error || 'login failed', 'var(--red)');
    return;
  }
  setLoggedIn(j.username, j.created ? 'new user' : 'logged in');
  await refreshSlideList();
  await pickFirstSlide();
}

async function doLogout() {
  await fetch('/api/logout', {method: 'POST'});
  setLoggedOut();
  $('user-input').value = '';
}

// ── Slides ─────────────────────────────────────────────────────────────────────
async function refreshSlideList() {
  const resp = await fetch('/api/slides');
  const slides = await resp.json();
  const list = $('slide-list');
  list.innerHTML = '';
  if (!slides.length) {
    list.innerHTML = '<div style="padding:8px 14px;color:var(--mut);font-size:12px">No slides yet. Upload one below or drop a preprocessed folder into slides/.</div>';
    return;
  }
  slides.forEach(s => {
    const el = document.createElement('div');
    el.className = 'slide-item' + (S.slide === s.name ? ' active' : '');
    el.dataset.name = s.name;
    const dot = s.has_pred ? '<span class="has-pred-dot" title="AI predictions available"></span>' : '';
    el.innerHTML = `<span class="slide-name-txt" title="${s.name}">${s.name}${dot}</span><span class="s-badge">${s.labeled}</span>`;
    el.onclick = () => pickSlide(s.name);
    list.appendChild(el);
  });
}

async function pickFirstSlide() {
  const first = document.querySelector('.slide-item');
  if (first) await pickSlide(first.dataset.name);
}

async function pickSlide(name) {
  if (!S.user) {
    setUserStatus('log in to load a slide', 'var(--amb)');
    return;
  }
  S.slide = name;
  S.tiles = [];
  S.umap = null;
  S.umapStatic = null;
  S.tileNumToIdx = {};
  S.centroids = {};
  S.tileStatus = new Map();
  S.versions = [];
  closePanel();
  document.querySelectorAll('.slide-item').forEach(e =>
    e.classList.toggle('active', e.dataset.name === name));
  $('slide-title').innerHTML = `<b>${name.toUpperCase()}</b>`;

  const [distResp, centResp] = await Promise.all([
    api(`/api/distributions?slide=${encodeURIComponent(name)}`),
    api(`/api/centroids?slide=${encodeURIComponent(name)}`),
  ]);
  S.dist = await distResp.json();
  S.centroids = await centResp.json();
  if (S.dist.available && S.dist.image_dimensions) {
    S.origW = S.dist.image_dimensions.width  || 1;
    S.origH = S.dist.image_dimensions.height || 1;
  }
  if (S.dist.available) renderDistPlots();

  if (S.user) await loadVersions();
  await loadUmapData();
  if (S.view === 'grid')  await loadTiles();
  else if (S.view === 'slide') await loadSlideView();
}

// ── Upload & Preprocess Slide ──────────────────────────────────────────────────
function setupUploadDropZone() {
  const dz = $('drop-zone');
  if (!dz) return;
  ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => {
    e.preventDefault(); e.stopPropagation(); dz.classList.add('drag');
  }));
  ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => {
    e.preventDefault(); e.stopPropagation(); dz.classList.remove('drag');
  }));
  dz.addEventListener('drop', e => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onSlideFileSelected(e.dataTransfer.files[0]);
    }
  });
}

function openUploadModal() {
  if (!S.user || S.uploading) return;
  S.pendingUploadFile = null;
  $('slide-file-input').value = '';
  $('upload-slide-name').value = '';
  $('upload-modal-err').textContent = '';
  $('drop-zone-label').innerHTML = `Click or drag &amp; drop a slide image here<br>
    <span style="font-size:11px;opacity:.8">.PNG, .JPG, .TIF, .TIFF (max ${MAX_SLIDE_DIM}×${MAX_SLIDE_DIM} px)</span>`;
  $('upload-modal').classList.add('open');
}

function closeUploadModal() {
  $('upload-modal').classList.remove('open');
}

function onSlideFileSelected(file) {
  if (!file) return;
  S.pendingUploadFile = file;
  $('upload-modal-err').textContent = '';
  const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
  $('drop-zone-label').innerHTML = `<b>${file.name}</b><br><span style="font-size:11px;opacity:.8">${sizeMB} MB selected</span>`;
  if (!$('upload-slide-name').value) {
    $('upload-slide-name').value = file.name.replace(/\.[^.]+$/, '');
  }
}

async function submitSlideUpload() {
  if (!S.pendingUploadFile) {
    $('upload-modal-err').textContent = 'Please select a slide image first.';
    return;
  }
  const fd = new FormData();
  fd.append('file', S.pendingUploadFile);
  const customName = $('upload-slide-name').value.trim();
  if (customName) fd.append('slide_name', customName);

  $('upload-submit-btn').disabled = true;
  $('upload-modal-err').textContent = 'Uploading image...';

  let j = {};
  let r;
  try {
    r = await api('/api/upload_slide', {method: 'POST', body: fd});
    j = await r.json().catch(() => ({}));
  } catch (err) {
    $('upload-submit-btn').disabled = false;
    $('upload-modal-err').textContent = err.message || 'Upload failed';
    return;
  }
  $('upload-submit-btn').disabled = false;

  if (!r.ok) {
    $('upload-modal-err').textContent = j.error || `Upload failed (HTTP ${r.status})`;
    return;
  }

  closeUploadModal();
  S.uploading = true;
  $('upload-slide-btn').disabled = true;
  $('upload-prog-box').style.display = 'block';
  $('upload-prog-fill').style.width = '2%';
  $('upload-prog-fill').style.background = 'var(--acc)';
  $('upload-prog-lbl').textContent = `Preprocessing ${j.slide} (${j.width}×${j.height})...`;

  const src = new EventSource(`/api/preprocess_stream?job_id=${encodeURIComponent(j.job_id)}`);
  src.onmessage = async e => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'progress') {
      $('upload-prog-fill').style.width = msg.progress + '%';
      $('upload-prog-fill').style.background = 'var(--grn)';
      $('upload-prog-lbl').textContent = msg.progress.toFixed(0) + '%' + (msg.message ? ' - ' + msg.message : '');
    } else if (msg.type === 'done') {
      src.close();
      S.uploading = false;
      $('upload-slide-btn').disabled = !S.user;
      if (msg.success) {
        $('upload-prog-fill').style.width = '100%';
        $('upload-prog-lbl').textContent = `Ready: ${msg.slide || j.slide}`;
        setTimeout(() => { $('upload-prog-box').style.display = 'none'; }, 3500);
        await refreshSlideList();
        await pickSlide(msg.slide || j.slide);
      } else {
        $('upload-prog-fill').style.width = '100%';
        $('upload-prog-fill').style.background = 'var(--red)';
        $('upload-prog-lbl').textContent = 'Failed: ' + (msg.error || 'see server logs');
      }
    }
  };
  src.onerror = () => {
    src.close();
    S.uploading = false;
    $('upload-slide-btn').disabled = !S.user;
    $('upload-prog-fill').style.background = 'var(--red)';
    $('upload-prog-lbl').textContent = 'Preprocess stream error';
  };
}

// ── View switching ─────────────────────────────────────────────────────────────
async function setView(v, btn) {
  S.view = v;
  if (v !== 'slide') stopSlideBgLoop();
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('on'));
  if (btn) btn.classList.add('on');
  document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
  $(`${v}-view`).classList.add('active');
  $('umap-controls').style.display  = v === 'umap'  ? '' : 'none';
  $('slide-controls').style.display = v === 'slide' ? '' : 'none';

  const distSuffix = v === 'umap' ? 'umap' : v === 'grid' ? 'grid' : 'slide';
  ['dist-cell-', 'dist-nc-'].forEach(prefix => {
    const el = $(prefix + distSuffix);
    if (el && el.data) Plotly.Plots.resize(el);
  });

  if (!S.slide) return;
  await loadCurrentView();

  if (v === 'umap' && S.umap) {
    Plotly.relayout('umap-scatter', {autosize: true});
    if (S.selectedUmapIdx !== null) {
      Plotly.restyle('umap-scatter', {
        'marker.color':   [getUmapColors()],
        'marker.opacity': [getUmapOpacities()],
      });
    }
  }
}

async function loadCurrentView() {
  if      (S.view === 'umap')  { if (!S.umap) await loadUmapData(); else renderUmap(); }
  else if (S.view === 'grid')  await loadTiles();
  else if (S.view === 'slide') await loadSlideView();
}

// ── UMAP view ──────────────────────────────────────────────────────────────────
async function loadUmapStatic() {
  const r = await api(`/api/umap_static?slide=${encodeURIComponent(S.slide)}`);
  S.umapStatic = await r.json();
  S.tileNumToIdx = {};
  S.umapStatic.tile_nums.forEach((tn, i) => { S.tileNumToIdx[tn] = i; });
  S.manualTileNums = new Set(S.umapStatic.manual_tile_nums || []);
}

async function loadUmapData() {
  if (!S.slide) return;
  loading(true);
  try {
    if (!S.umapStatic) await loadUmapStatic();
    const r = await api(`/api/umap?slide=${encodeURIComponent(S.slide)}`);
    const dynamic = await r.json();
    S.umap = { ...S.umapStatic, ...dynamic };
    renderUmap();
    updateLabeledCount();
  } finally {
    loading(false);
  }
}

function getUmapColors() {
  if (!S.umap) return [];
  if (S.colorFeature) {
    const palette = ['#4169e1', '#22c55e', '#f87171', '#c084fc', '#f59e0b'];
    return S.umap.associated_data[S.colorFeature].map(v => palette[v] || '#88aaff');
  }
  return S.umap.colors;
}

function getUmapOpacities() {
  if (!S.umap) return [];
  const n = S.umap.x.length;
  if (S.selectedUmapIdx != null) {
    return Array(n).fill(0).map((_, i) => i === S.selectedUmapIdx ? 1.0 : 0.2);
  }
  if (S.colorFeature) {
    return S.umap.associated_data[S.colorFeature].map(v => v > 0 ? 1.0 : 0.2);
  }
  return Array(n).fill(1.0);
}

function renderUmap() {
  if (!S.umap) return;
  const trace = {
    x: S.umap.x, y: S.umap.y,
    mode: 'markers', type: 'scattergl',
    marker: {color: getUmapColors(), size: S.pointSize, opacity: getUmapOpacities()},
    text: S.umap.desc,
    customdata: S.umap.tile_nums,
    hoverinfo: 'text',
  };
  const layout = {
    paper_bgcolor: plotlyPaperBg(), plot_bgcolor: plotlyPaperBg(),
    font: plotlyFont(),
    margin: {l: 10, r: 10, t: 10, b: 10},
    xaxis: {visible: false, showgrid: false},
    yaxis: {visible: false, showgrid: false},
    hovermode: 'closest', dragmode: 'pan',
    uirevision: 'umap',
  };
  Plotly.newPlot('umap-scatter', [trace], layout, {
    responsive: true, displaylogo: false, scrollZoom: true,
    modeBarButtons: [['pan2d', 'autoScale2d']],
  });
  const el = $('umap-scatter');
  el.on('plotly_hover',   e => onUmapHover(e.points[0].pointIndex));
  el.on('plotly_click',   e => onUmapClick(e.points[0].pointIndex));
  el.on('plotly_unhover', () => {
    $('umap-hover').innerHTML = '<div class="hover-hint">Hover a point to preview</div>';
  });
}

function onUmapHover(idx) {
  const imgPath = S.umap.imgs[idx];
  const marked = S.umap.marked[idx];
  let tagsHtml = '';
  if (marked) {
    tagsHtml = LABEL_COLS.map(col => {
      const v = LABEL_STR[col][S.umap.associated_data[col][idx]];
      return v !== 'None' ? `<span class="tag v ${col}" style="margin:1px">${v}</span>` : '';
    }).filter(Boolean).join('');
  }
  $('umap-hover').innerHTML = `
    <div style="font-size:14px;color:var(--txt);margin-bottom:5px">${S.umap.desc[idx]}</div>
    ${imgPath ? `<img src="/img?path=${encodeURIComponent(imgPath)}"
      style="width:100%;border-radius:4px;border:1px solid var(--brd)">` : ''}
    <div style="display:flex;flex-wrap:wrap;gap:2px;margin-top:4px">${tagsHtml}</div>`;
  updateDistSpans(idx);
}

function onUmapClick(idx) {
  const tn = S.umap.tile_nums[idx];
  const paneRaw = S.umap.imgs_pane ? S.umap.imgs_pane[idx] : S.umap.imgs[idx];
  const labels = {};
  LABEL_COLS.forEach(col => {
    labels[col] = LABEL_STR[col][S.umap.associated_data[col][idx]] || 'None';
  });
  openEditPanel({
    tile_num: tn, desc: S.umap.desc[idx], source: 'umap',
    labels, probs: LABEL_COLS.reduce((o, c) => ({...o, [c]: null}), {}),
    raw_path: paneRaw,
  });
}

function updatePointSize(val) {
  S.pointSize = parseInt(val);
  $('pt-size-val').textContent = S.pointSize + 'px';
  if (S.umap && $('umap-scatter').data) {
    Plotly.restyle('umap-scatter', {'marker.size': S.pointSize});
  }
}

function colorUmap(feature, btn) {
  S.colorFeature = feature;
  document.querySelectorAll('#color-btns .f-btn').forEach(b => b.classList.remove('on'));
  if (btn) btn.classList.add('on');
  if (S.umap) {
    Plotly.restyle('umap-scatter', {
      'marker.color': [getUmapColors()],
      'marker.opacity': [getUmapOpacities()],
    });
  }
}

function buildColorButtons() {
  const c = $('color-btns');
  LABEL_COLS.forEach(col => {
    const b = document.createElement('button');
    b.className = 'f-btn';
    b.textContent = col.toUpperCase();
    b.onclick = () => colorUmap(col, b);
    c.appendChild(b);
  });
}

function buildSlideColorButtons() {
  const c = $('slide-color-btns');
  LABEL_COLS.forEach(col => {
    const b = document.createElement('button');
    b.className = 'f-btn';
    b.textContent = col.toUpperCase();
    b.onclick = () => colorSlide(col, b);
    c.appendChild(b);
  });
}

function colorSlide(feature, btn) {
  S.slideColorFeature = feature;
  document.querySelectorAll('#slide-color-btns .f-btn').forEach(b => b.classList.remove('on'));
  if (btn) btn.classList.add('on');
  if (S.umap) drawSlideCanvas();
}

function updateSlideLineThickness(val) {
  S.slideLineThickness = parseInt(val);
  $('slide-thickness-val').textContent = S.slideLineThickness + 'px';
  if (S.umap) drawSlideCanvas();
}

function updateSlideDimAlpha(val) {
  S.slideDimAlpha = parseFloat(val);
  $('slide-dim-val').textContent = Math.round(S.slideDimAlpha * 100) + '%';
  if (S.umap) drawSlideCanvas();
}

// ── Grid view ──────────────────────────────────────────────────────────────────
async function loadTiles() {
  if (!S.slide) return;
  loading(true);
  try {
    const fp = {};
    Object.entries(S.activeFilters).forEach(([h, s]) => { fp[h] = [...s]; });
    const p = new URLSearchParams({slide: S.slide, view: S.gridMode, filters: JSON.stringify(fp)});
    const r = await api('/api/tiles?' + p);
    S.tiles = await r.json();
    $('cntLbl').innerHTML = `<b>${S.tiles.length}</b> tiles`;
    renderGrid();
  } finally {
    loading(false);
  }
}

function renderGrid() {
  const g = $('grid');
  if (!S.tiles.length) {
    g.innerHTML = '<div class="empty"><div class="sym">◫</div>No tiles match filters</div>';
    return;
  }
  g.innerHTML = '';
  const ahs = new Set(Object.keys(S.activeFilters));
  S.tiles.forEach((t, i) => {
    const c = document.createElement('div');
    c.className = 'card'; c.dataset.i = i;
    let maxProb = null;
    if (t.source === 'pred') {
      const ps = Object.entries(t.probs)
        .filter(([k, v]) => v !== null && (ahs.size === 0 || ahs.has(k)));
      if (ps.length) maxProb = Math.max(...ps.map(([, v]) => v));
    }
    const bar = maxProb !== null
      ? `<div class="prob-bar"><div class="prob-fill" style="width:${(maxProb * 100).toFixed(0)}%"></div></div>` : '';
    const tags = LABEL_COLS.map(col => {
      const v = t.labels[col]; const ok = v && v !== 'None';
      const dim = ahs.size > 0 && !ahs.has(col) ? 'style="opacity:.3"' : '';
      return `<span class="tag ${ok ? 'v' : ''} ${col}" ${dim}>${ok ? v : col}</span>`;
    }).join('');
    const sk = t.source === 'pred' ? S.tileStatus.get(t.tile_num) : null;
    const sh = sk ? `<div class="card-status ${sk}">${sk.toUpperCase()}</div>` : '';
    c.innerHTML = `
      <div class="card-hd"><span class="card-name">${t.desc}</span><span class="src ${t.source}">${t.source.toUpperCase()}</span></div>
      <div class="card-img-wrap"><img class="card-img" src="/img?path=${encodeURIComponent(t.grid_path || t.raw_path)}" alt="${t.desc}" loading="lazy">${sh}</div>
      ${bar}<div class="card-tags">${tags}</div>`;
    c.onclick = () => pickGrid(i);
    c.addEventListener('mouseenter', () => {
      const idx = S.tileNumToIdx[t.tile_num];
      if (idx !== undefined) updateDistSpans(idx);
    });
    g.appendChild(c);
  });
}

function pickGrid(i) {
  S.selGridIdx = i;
  const t = S.tiles[i];
  document.querySelectorAll('.card').forEach(c => c.classList.remove('sel'));
  document.querySelector(`.card[data-i="${i}"]`)?.classList.add('sel');
  if (t.source === 'pred' && !S.tileStatus.has(t.tile_num)) {
    S.tileStatus.set(t.tile_num, 'seen');
    setCardStatus(i, 'seen');
  }
  openEditPanel(t);
}

function setCardStatus(i, status) {
  const t = S.tiles[i];
  if (!t || t.source !== 'pred') return;
  S.tileStatus.set(t.tile_num, status);
  const wrap = document.querySelector(`.card[data-i="${i}"] .card-img-wrap`);
  if (!wrap) return;
  let dot = wrap.querySelector('.card-status');
  if (!dot) { dot = document.createElement('div'); dot.className = 'card-status'; wrap.appendChild(dot); }
  dot.className = `card-status ${status}`;
  dot.textContent = status.toUpperCase();
}

function updateGridCard(i, labels) {
  const card = document.querySelector(`.card[data-i="${i}"]`);
  if (!card) return;
  const ahs = new Set(Object.keys(S.activeFilters));
  card.querySelector('.card-tags').innerHTML = LABEL_COLS.map(col => {
    const v = labels[col]; const ok = v && v !== 'None';
    const dim = ahs.size > 0 && !ahs.has(col) ? 'style="opacity:.3"' : '';
    return `<span class="tag ${ok ? 'v' : ''} ${col}" ${dim}>${ok ? v : col}</span>`;
  }).join('');
}

function setGridMode(mode, btn) {
  S.gridMode = mode;
  document.querySelectorAll('#grid-side .f-btn').forEach(b => b.classList.remove('on'));
  if (btn) btn.classList.add('on');
  if (S.view === 'grid') loadTiles();
}

function buildFilterBoxes() {
  const root = $('filter-boxes-root');
  LABEL_COLS.forEach(head => {
    const hd = document.createElement('div');
    hd.className = 'filter-head'; hd.textContent = head.toUpperCase();
    root.appendChild(hd);
    const row = document.createElement('div');
    row.className = 'filter-boxes';
    (LABEL_OPTIONS[head] || []).forEach(cls => {
      const b = document.createElement('button');
      b.className = `fbox ${head}`; b.textContent = cls;
      b.onclick = () => toggleFilter(head, cls, b);
      row.appendChild(b);
    });
    root.appendChild(row);
  });
}

function toggleFilter(head, cls, btn) {
  if (!S.activeFilters[head]) S.activeFilters[head] = new Set();
  if (S.activeFilters[head].has(cls)) {
    S.activeFilters[head].delete(cls); btn.classList.remove('on');
    if (S.activeFilters[head].size === 0) delete S.activeFilters[head];
  } else {
    S.activeFilters[head].add(cls); btn.classList.add('on');
  }
  if (S.view === 'grid') loadTiles();
}

function clearFilters() {
  S.activeFilters = {};
  document.querySelectorAll('#filter-boxes-root .fbox.on').forEach(b => b.classList.remove('on'));
  if (S.view === 'grid') loadTiles();
}

// ── Slide view ─────────────────────────────────────────────────────────────────
let redBoxEnabled = false;

function toggleRedBox() {
  redBoxEnabled = !redBoxEnabled;
  const btn = $('red-box-btn');
  if (redBoxEnabled) {
    btn.textContent = 'BOX OFF';
    btn.style.background = 'var(--red)'; btn.style.color = '#fff'; btn.style.borderColor = 'var(--red)';
    $('rb-raw').classList.add('on');
  } else {
    btn.textContent = 'BOX ON';
    btn.style.background = 'var(--surf2)'; btn.style.color = 'var(--mut)'; btn.style.borderColor = 'var(--brd)';
    $('rb-raw').classList.remove('on');
  }
}

async function loadSlideView() {
  if (!S.slide) return;
  loading(true);
  try {
    await renderSlideView();
  } finally {
    loading(false);
  }
}

async function renderSlideView() {
  if (!S.umapStatic) await loadUmapStatic();
  if (!S.umap) {
    const r = await api(`/api/umap?slide=${encodeURIComponent(S.slide)}`);
    const dynamic = await r.json();
    S.umap = { ...S.umapStatic, ...dynamic };
    updateLabeledCount();
  }
  const r = await api(`/api/slide_image?slide=${encodeURIComponent(S.slide)}`);
  const info = await r.json();
  const toggleEl = $('slide-img-toggle');
  toggleEl.innerHTML = '';
  if (info.available) {
    S.slideImages = info.images;
    S.slideImageIdx = 0;
    if (info.images.length > 1) {
      toggleEl.style.display = 'flex';
      info.images.forEach((img, i) => {
        const b = document.createElement('button');
        b.className = 'f-btn' + (i === 0 ? ' on' : '');
        b.textContent = img.label;
        b.onclick = () => switchSlideImage(i, b);
        toggleEl.appendChild(b);
      });
    } else {
      toggleEl.style.display = 'none';
    }
    renderSlidePlotly(`/img?path=${encodeURIComponent(info.images[0].path)}`);
  } else {
    S.slideImages = [];
    toggleEl.style.display = 'none';
    $('slide-bg-img').style.display = 'none';
    $('slide-plotly').innerHTML = '<div style="padding:20px;color:var(--mut);text-align:center">No slide image available for this slide</div>';
  }
}

function switchSlideImage(idx, btn) {
  S.slideImageIdx = idx;
  document.querySelectorAll('#slide-img-toggle .f-btn').forEach(b => b.classList.remove('on'));
  if (btn) btn.classList.add('on');
  $('slide-bg-img').src = `/img?path=${encodeURIComponent(S.slideImages[idx].path)}`;
}

function renderSlidePlotly(imgUrl) {
  if (!S.umap) return;
  const bgImg = $('slide-bg-img');
  bgImg.src = imgUrl;
  bgImg.style.display = 'block';

  const xData = [], yData = [], hoverText = [], customData = [];
  S.umap.tile_nums.forEach((tn, idx) => {
    const c = S.centroids[tn];
    if (!c) return;
    xData.push(c.x);
    yData.push(S.origH - c.y);
    hoverText.push(S.umap.desc[idx]);
    customData.push([tn, idx]);
  });

  const trace = {
    x: xData, y: yData,
    text: hoverText, customdata: customData,
    mode: 'markers', type: 'scattergl',
    marker: {size: 20, color: 'rgba(0,0,0,0)'},
    hoverinfo: 'text',
  };

  const layout = {
    paper_bgcolor: plotlyPaperBg(), plot_bgcolor: plotlyPaperBg(), font: plotlyFont(),
    margin: {l: 0, r: 0, t: 0, b: 0},
    xaxis: {range: [0, S.origW], visible: false},
    yaxis: {range: [0, S.origH], visible: false, scaleanchor: 'x'},
    hovermode: 'closest', dragmode: 'pan',
    uirevision: 'slide',
  };

  Plotly.newPlot('slide-plotly', [trace], layout, {
    responsive: true, scrollZoom: true, displaylogo: false,
    modeBarButtons: [[
      'pan2d',
      {
        name: 'Reset view',
        icon: Plotly.Icons.autoscale,
        click: () => {
          Plotly.relayout('slide-plotly', {
            'xaxis.range': [0, S.origW],
            'yaxis.range': [0, S.origH],
            'xaxis.autorange': false,
            'yaxis.autorange': false,
          });
        },
      },
    ]],
  }).then(() => {
    requestAnimationFrame(() => { syncBgImage(); drawSlideCanvas(); });
    startSlideBgLoop();
  });

  const el = $('slide-plotly');
  el.removeAllListeners('plotly_relayout');
  el.removeAllListeners('plotly_click');
  el.removeAllListeners('plotly_hover');
  el.removeAllListeners('plotly_unhover');

  el.on('plotly_relayout', () => {
    drawSlideCanvas();
    syncScatterTrace();
  });
  el.on('plotly_click', e => {
    if (e.points.length > 0) pickSlideBox(e.points[0].customdata[0], e.points[0].customdata[1]);
  });
  el.on('plotly_hover', e => {
    if (e.points.length > 0) onSlideHover(e.points[0].customdata[0], e.points[0].customdata[1]);
  });
  el.on('plotly_unhover', () => {
    $('slide-info').innerHTML = '<span style="color:var(--mut);font-size:13px">Click or hover a cell box for details</span>';
  });
}

function syncScatterTrace() {
  const plotEl = $('slide-plotly');
  if (!plotEl || !plotEl.data || !S.umap || !S.centroids) return;
  const xData = [], yData = [], hoverText = [], customData = [];
  S.umap.tile_nums.forEach((tn, idx) => {
    const c = S.centroids[tn];
    if (!c) return;
    xData.push(c.x);
    yData.push(S.origH - c.y);
    hoverText.push(S.umap.desc[idx]);
    customData.push([tn, idx]);
  });
  Plotly.restyle('slide-plotly', {
    x: [xData], y: [yData], text: [hoverText], customdata: [customData],
  }, [0]);
}

function syncBgImage() {
  const plotEl = $('slide-plotly');
  const img = $('slide-bg-img');
  if (!plotEl._fullLayout || !img.src) return;
  const fl = plotEl._fullLayout;
  const xRange = fl.xaxis.range;
  const yRange = fl.yaxis.range;
  const xs = fl._size.w / (xRange[1] - xRange[0]);
  const ys = fl._size.h / (yRange[1] - yRange[0]);
  img.style.left   = (fl._size.l + (0         - xRange[0]) * xs) + 'px';
  img.style.top    = (fl._size.t + (yRange[1] - S.origH)   * ys) + 'px';
  img.style.width  = (S.origW * xs) + 'px';
  img.style.height = (S.origH * ys) + 'px';
}

function startSlideBgLoop() {
  stopSlideBgLoop();
  function loop() {
    const plotEl = $('slide-plotly');
    if (!plotEl || !plotEl._fullLayout) { _slideRafId = requestAnimationFrame(loop); return; }
    const xr = plotEl._fullLayout.xaxis.range;
    const yr = plotEl._fullLayout.yaxis.range;
    if (!_lastXRange ||
        xr[0] !== _lastXRange[0] || xr[1] !== _lastXRange[1] ||
        yr[0] !== _lastYRange[0] || yr[1] !== _lastYRange[1]) {
      _lastXRange = xr.slice();
      _lastYRange = yr.slice();
      syncBgImage();
      drawSlideCanvas();
    }
    _slideRafId = requestAnimationFrame(loop);
  }
  _slideRafId = requestAnimationFrame(loop);
}

function stopSlideBgLoop() {
  if (_slideRafId !== null) { cancelAnimationFrame(_slideRafId); _slideRafId = null; }
  _lastXRange = null;
  _lastYRange = null;
}

function onSlideHover(tileNum, umapIdx) {
  const imgPath = S.umap.imgs[umapIdx];
  const marked = S.umap.marked[umapIdx];
  let tagsHtml = '';
  if (marked) {
    tagsHtml = LABEL_COLS.map(col => {
      const v = LABEL_STR[col][S.umap.associated_data[col][umapIdx]];
      return v !== 'None' ? `<span class="tag v ${col}" style="margin:1px">${v}</span>` : '';
    }).filter(Boolean).join('');
  }
  $('slide-info').innerHTML = `
    <div style="font-size:14px;color:var(--txt);margin-bottom:5px">${S.umap.desc[umapIdx]}</div>
    ${imgPath ? `<img src="/img?path=${encodeURIComponent(imgPath)}"
      style="width:100%;border-radius:4px;border:1px solid var(--brd)">` : ''}
    <div style="display:flex;flex-wrap:wrap;gap:2px;margin-top:4px">${tagsHtml}</div>`;
  updateDistSpans(umapIdx);
}

function pickSlideBox(tileNum, umapIdx) {
  const desc = S.umap.desc[umapIdx];
  const paneRaw = S.umap.imgs_pane ? S.umap.imgs_pane[umapIdx] : S.umap.imgs[umapIdx];
  const labels = {};
  LABEL_COLS.forEach(col => {
    labels[col] = LABEL_STR[col][S.umap.associated_data[col][umapIdx]] || 'None';
  });
  openEditPanel({
    tile_num: tileNum, desc, source: 'slide',
    labels, probs: LABEL_COLS.reduce((o, c) => ({...o, [c]: null}), {}),
    raw_path: paneRaw,
  });
}

function drawSlideCanvas() {
  const canvas = $('slide-canvas');
  const plotEl = $('slide-plotly');
  if (!canvas || !plotEl || !plotEl._fullLayout || !S.umap || !S.centroids) return;

  const dpr = window.devicePixelRatio || 1;
  const w = plotEl.offsetWidth;
  const h = plotEl.offsetHeight;
  canvas.width  = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const fl = plotEl._fullLayout;
  const pl = fl._size.l, pt = fl._size.t;
  const pw = fl._size.w, ph = fl._size.h;
  const [x0, x1] = fl.xaxis.range;
  const [y0, y1] = fl.yaxis.range;
  const xs = pw / (x1 - x0);
  const ys = ph / (y1 - y0);

  function toScreen(dx, dy) {
    return [pl + (dx - x0) * xs, pt + (y1 - dy) * ys];
  }

  function drawBox(cx, cy, color, lineWidth) {
    const [sx0, sy0] = toScreen(cx - 42.5, cy + 42.5);
    const [sx1, sy1] = toScreen(cx + 42.5, cy - 42.5);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
  }

  if (S.slideDimAlpha > 0) {
    const holeNums = new Set();
    S.umap.tile_nums.forEach((tn, idx) => {
      if (!S.umap.marked[idx]) return;
      if (S.slideColorFeature && S.umap.associated_data[S.slideColorFeature][idx] === 0) return;
      holeNums.add(tn);
    });
    if (S.selectedSlideTileNum !== null) holeNums.add(S.selectedSlideTileNum);
    ctx.save();
    ctx.fillStyle = `rgba(0,0,0,${S.slideDimAlpha})`;
    ctx.fillRect(0, 0, w, h);
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = 'rgba(0,0,0,1)';
    holeNums.forEach(tn => {
      const c = S.centroids[tn];
      if (!c) return;
      const [sx0, sy0] = toScreen(c.x - 42.5, S.origH - c.y + 42.5);
      const [sx1, sy1] = toScreen(c.x + 42.5, S.origH - c.y - 42.5);
      ctx.fillRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
    });
    ctx.restore();
  }

  S.umap.tile_nums.forEach((tn, idx) => {
    if (!S.umap.marked[idx]) return;
    if (S.slideColorFeature && S.umap.associated_data[S.slideColorFeature][idx] === 0) return;
    if (tn === S.selectedSlideTileNum) return;
    const c = S.centroids[tn];
    if (!c) return;
    const color = S.slideColorFeature ? (SLIDE_CLASS_COLORS[S.slideColorFeature] || '#00d4ff') : '#00d4ff';
    drawBox(c.x, S.origH - c.y, color, S.slideLineThickness);
  });

  if (S.selectedSlideTileNum !== null) {
    const c = S.centroids[S.selectedSlideTileNum];
    if (c) {
      const selIdx = S.tileNumToIdx[S.selectedSlideTileNum];
      const isAnnotated = selIdx !== undefined && S.umap.marked[selIdx] === 1;
      drawBox(c.x, S.origH - c.y, isAnnotated ? '#22c55e' : '#ef4444', Math.max(S.slideLineThickness, 2));
    }
  }

  if (S.drawingMode && S.polygonPoints.length > 0) {
    ctx.save();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
    ctx.beginPath();
    const [px0, py0] = toScreen(S.polygonPoints[0][0], S.origH - S.polygonPoints[0][1]);
    ctx.moveTo(px0, py0);
    for (let i = 1; i < S.polygonPoints.length; i++) {
      const [px, py] = toScreen(S.polygonPoints[i][0], S.origH - S.polygonPoints[i][1]);
      ctx.lineTo(px, py);
    }
    if (S.polygonMousePos) {
      const [mx, my] = toScreen(S.polygonMousePos[0], S.origH - S.polygonMousePos[1]);
      ctx.lineTo(mx, my);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

// ── Distribution plots ─────────────────────────────────────────────────────────
function renderDistPlots() {
  if (!S.dist || !S.dist.available) return;
  [
    ['dist-cell-umap',  S.dist.cell],
    ['dist-nc-umap',    S.dist.nc_ratio],
    ['dist-cell-slide', S.dist.cell],
    ['dist-nc-slide',   S.dist.nc_ratio],
    ['dist-cell-grid',  S.dist.cell],
    ['dist-nc-grid',    S.dist.nc_ratio],
  ].forEach(([id, data]) => renderOneDist(id, data));
}

function renderOneDist(id, data) {
  const el = $(id);
  if (!el || !data || !data.edges || data.edges.length < 2) return;
  const edges = data.edges;
  const xs = edges.slice(0, -1).map((v, i) => (v + edges[i + 1]) / 2);
  const trace = {
    x: xs, y: data.hist, type: 'bar',
    marker: {color: '#4d7cfe', opacity: 0.55},
    width: edges.slice(0, -1).map((v, i) => edges[i + 1] - v),
    hoverinfo: 'none',
  };
  Plotly.newPlot(id, [trace], {
    paper_bgcolor: plotlyPaperBg(), plot_bgcolor: plotlyPaperBg(),
    font: plotlyFont(),
    margin: {l: 5, r: 5, t: 5, b: 5}, bargap: 0, showlegend: false,
    xaxis: {visible: false, showgrid: false},
    yaxis: {visible: false, showgrid: false},
    shapes: [],
  }, {responsive: true, displayModeBar: false, staticPlot: true});
}

function updateDistSpans(umapIdx) {
  if (!S.dist || !S.dist.available || !S.umap) return;
  const cellSz = S.umap.cell_sizes[umapIdx];
  const ncR    = S.umap.nc_ratios[umapIdx];
  const mk = (val) => [{type: 'line', xref: 'x', yref: 'paper', x0: val, x1: val, y0: 0, y1: 1,
                        line: {color: '#f87171', width: 2}}];
  if (cellSz !== undefined) {
    ['dist-cell-umap', 'dist-cell-slide', 'dist-cell-grid'].forEach(id => {
      if ($(id)?.data) Plotly.relayout(id, {shapes: mk(cellSz)});
    });
  }
  if (ncR !== undefined) {
    ['dist-nc-umap', 'dist-nc-slide', 'dist-nc-grid'].forEach(id => {
      if ($(id)?.data) Plotly.relayout(id, {shapes: mk(ncR)});
    });
  }
}

// ── Edit panel ─────────────────────────────────────────────────────────────────
function checkLabelConsistency() {
  const labels = getLabels();
  const warnings = [];
  if (labels.mitotic !== 'None') {
    const others = LABEL_COLS.filter(c => c !== 'mitotic' && labels[c] !== 'None');
    if (others.length) {
      warnings.push('Mitotic/Bizarre figures usually have all other attributes set to None.');
    }
  }
  if (labels.prominent !== 'None' && labels.nucleoli === 'None') {
    warnings.push('Prominent Nucleoli usually requires Single or Multiple Nucleoli.');
  }
  const el = $('combo-warn');
  if (warnings.length) {
    el.textContent = 'Note: ' + warnings.join(' ');
    el.style.display = 'block';
  } else {
    el.style.display = 'none';
  }
}

function openEditPanel(tile) {
  currentTile = tile;
  S.selectedUmapIdx = S.tileNumToIdx[tile.tile_num] !== undefined ? S.tileNumToIdx[tile.tile_num] : null;
  if (S.umap && $('umap-scatter').data) {
    Plotly.restyle('umap-scatter', {'marker.opacity': [getUmapOpacities()]});
  }
  if (S.view === 'slide') {
    S.selectedSlideTileNum = tile.tile_num;
    if (S.umap) drawSlideCanvas();
  }
  $('p-name').textContent = tile.desc.toUpperCase();
  const idx = S.tileNumToIdx[tile.tile_num];
  const _ann = tile.source === 'pred'
    ? (idx !== undefined && S.umap && S.umap.marked[idx] === 1)
    : LABEL_COLS.some(col => tile.labels[col] && tile.labels[col] !== 'None');
  $('p-ann-status').textContent = _ann ? 'ANNOTATED' : 'NOT ANNOTATED';
  $('p-ann-status').style.color = _ann ? 'var(--grn)' : 'var(--red)';

  // Both preprocess.py and update_celladder.py pad edge crops centered on the cell -> always 50%, 50%
  const rb = $('rb-raw');
  rb.style.borderColor = _ann ? '#22c55e' : '#ef4444';
  rb.style.left = '50%';
  rb.style.top  = '50%';

  $('st').textContent = '';
  $('p-raw').src = tile.raw_path ? `/img?path=${encodeURIComponent(tile.raw_path)}` : '';
  const flds = $('ann-fields');
  flds.innerHTML = '';
  LABEL_COLS.forEach(col => {
    const prob = tile.probs[col];
    const pTxt = (prob !== null && prob !== undefined)
      ? `<span class="prob-txt">${(prob * 100).toFixed(1)}%</span>` : '';
    const div = document.createElement('div');
    div.className = 'ann-row';
    const boxes = (LABEL_OPTIONS[col] || []).map(opt => {
      const on = opt === tile.labels[col] ? 'on' : '';
      return `<button class="abox ${col} ${on}" data-head="${col}" data-cls="${opt}"
              onclick="pickAnn('${col}','${opt}',this)">${opt}</button>`;
    }).join('');
    div.innerHTML = `<div class="ann-lbl"><span>${col.toUpperCase()}</span>${pTxt}</div>
                     <div class="ann-boxes" id="aboxes-${col}">${boxes}</div>`;
    flds.appendChild(div);
  });
  checkLabelConsistency();
  $('btn-delete').style.display = S.manualTileNums.has(tile.tile_num) ? '' : 'none';
  $('panel').classList.add('open');
}

function pickAnn(head, cls, btn) {
  document.querySelectorAll(`#aboxes-${head} .abox`).forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  checkLabelConsistency();
}

function clearEditLabels() {
  LABEL_COLS.forEach(col => {
    document.querySelectorAll(`#aboxes-${col} .abox`).forEach(b => {
      b.classList.toggle('on', b.dataset.cls === 'None');
    });
  });
  checkLabelConsistency();
}

function closePanel() {
  $('panel').classList.remove('open');
  document.querySelectorAll('.card').forEach(c => c.classList.remove('sel'));
  S.selGridIdx = null;
  currentTile = null;
  S.selectedUmapIdx = null;
  if (S.umap && $('umap-scatter').data) {
    Plotly.restyle('umap-scatter', {'marker.opacity': [getUmapOpacities()]});
  }
  if (S.view === 'slide' && S.selectedSlideTileNum !== null) {
    S.selectedSlideTileNum = null;
    if (S.umap) drawSlideCanvas();
  }
}

function getLabels() {
  const out = {};
  LABEL_COLS.forEach(col => {
    const a = document.querySelector(`#aboxes-${col} .abox.on`);
    out[col] = a ? a.dataset.cls : 'None';
  });
  return out;
}

async function doSave() {
  if (!currentTile || !S.slide) return;
  const labels = getLabels();
  const r = await api('/api/save', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({slide: S.slide, tile_num: currentTile.tile_num, labels}),
  });
  const j = await r.json();
  const st = $('st');
  if (r.ok) {
    Object.assign(currentTile.labels, labels);
    const _savedAnn = LABEL_COLS.some(col => labels[col] && labels[col] !== 'None');
    $('p-ann-status').textContent = _savedAnn ? 'ANNOTATED' : 'NOT ANNOTATED';
    $('p-ann-status').style.color = _savedAnn ? 'var(--grn)' : 'var(--red)';
    $('rb-raw').style.borderColor = _savedAnn ? '#22c55e' : '#ef4444';
    const idx = S.tileNumToIdx[currentTile.tile_num];
    if (idx !== undefined && S.umap) {
      S.umap.marked[idx] = _savedAnn ? 1 : 0;
      LABEL_COLS.forEach(col => {
        S.umap.associated_data[col][idx] = LABEL_INT[col][labels[col]] || 0;
      });
      S.umap.colors[idx] = _savedAnn ? 'orange' : 'royalblue';
      if (!S.colorFeature && $('umap-scatter').data) {
        Plotly.restyle('umap-scatter', {'marker.color': [getUmapColors()]});
      }
    }
    if (S.selGridIdx !== null) {
      updateGridCard(S.selGridIdx, labels);
      setCardStatus(S.selGridIdx, 'saved');
    }
    if (S.view === 'slide' && S.umap) drawSlideCanvas();
    updateLabeledCount();
    await refreshSlideBadges();
    st.style.color = 'var(--grn)'; st.textContent = '✓ SAVED';
    setTimeout(() => { st.textContent = ''; }, 700);
  } else {
    st.style.color = 'var(--red)'; st.textContent = '✗ ' + (j.error || 'FAILED');
  }
}

async function refreshSlideBadges() {
  const resp = await api('/api/slides');
  const slides = await resp.json();
  slides.forEach(s => {
    const el = document.querySelector(`.slide-item[data-name="${s.name}"]`);
    if (!el) return;
    const badge = el.querySelector('.s-badge');
    if (badge) badge.textContent = s.labeled;
  });
}

function getSlideNavigationList() {
  if (!S.umap || !S.centroids) return [];
  return S.umap.tile_nums.filter(tn => {
    const idx = S.tileNumToIdx[tn];
    return S.umap.marked[idx] && S.centroids[tn];
  });
}

function doNext() {
  if (!currentTile) return;
  if (S.view === 'grid' && S.selGridIdx !== null) {
    const next = S.selGridIdx + 1;
    if (next < S.tiles.length) {
      pickGrid(next);
      document.querySelector(`.card[data-i="${next}"]`)?.scrollIntoView({behavior: 'smooth', block: 'nearest'});
    } else closePanel();
  } else if (S.view === 'umap' && S.selectedUmapIdx !== null && S.umap) {
    const next = S.selectedUmapIdx + 1;
    if (next < S.umap.tile_nums.length) onUmapClick(next);
    else closePanel();
  } else if (S.view === 'slide' && S.selectedSlideTileNum !== null && S.umap) {
    const list = getSlideNavigationList();
    const next = list.indexOf(S.selectedSlideTileNum) + 1;
    if (next < list.length) pickSlideBox(list[next], S.tileNumToIdx[list[next]]);
    else closePanel();
  } else closePanel();
}

function doPrev() {
  if (!currentTile) return;
  if (S.view === 'grid' && S.selGridIdx !== null) {
    const prev = S.selGridIdx - 1;
    if (prev >= 0) {
      pickGrid(prev);
      document.querySelector(`.card[data-i="${prev}"]`)?.scrollIntoView({behavior: 'smooth', block: 'nearest'});
    }
  } else if (S.view === 'umap' && S.selectedUmapIdx !== null && S.umap) {
    const prev = S.selectedUmapIdx - 1;
    if (prev >= 0) onUmapClick(prev);
  } else if (S.view === 'slide' && S.selectedSlideTileNum !== null && S.umap) {
    const list = getSlideNavigationList();
    const prev = list.indexOf(S.selectedSlideTileNum) - 1;
    if (prev >= 0) pickSlideBox(list[prev], S.tileNumToIdx[list[prev]]);
  }
}

// ── Version snapshots & CSV export ─────────────────────────────────────────────
async function doUpload() {
  if (!S.user || !S.slide || !S.umap) return;
  const state = {
    tile_nums: S.umap.tile_nums,
    embeddingXY: {x: S.umap.x, y: S.umap.y, marked: S.umap.marked, colors: S.umap.colors},
    associated_data: S.umap.associated_data,
    actions_log: [],
  };
  const r = await api('/api/upload', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({slide: S.slide, state}),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    setUserStatus(j.error || 'snapshot save failed', 'var(--red)');
    return;
  }
  S.umap.marked.forEach((m, i) => {
    if (m) S.umap.colors[i] = 'lime';
  });
  if (!S.colorFeature && $('umap-scatter').data) {
    Plotly.restyle('umap-scatter', {'marker.color': [getUmapColors()]});
  }
  await loadVersions();
  setUserStatus(`snapshot ${j.timestamp}`, 'var(--grn)');
}

function downloadAnnotationCSV() {
  if (!S.umap || !S.centroids) return;
  const header = ['x', 'y', 'desc', ...LABEL_COLS];
  const rows = [header.join(',')];
  S.umap.tile_nums.forEach((tn, idx) => {
    if (!S.umap.marked[idx]) return;
    const c = S.centroids[tn];
    if (!c) return;
    const labels = LABEL_COLS.map(col => LABEL_STR[col][S.umap.associated_data[col][idx]] || 'None');
    rows.push([c.x, c.y, S.umap.desc[idx], ...labels].join(','));
  });
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const filename = `annotation_data_${S.slide || 'slide'}_${S.user || 'user'}_${ts}.csv`;
  const blob = new Blob([rows.join('\n')], {type: 'text/csv;charset=utf-8;'});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function loadVersions() {
  if (!S.user || !S.slide) return;
  const r = await api(`/api/versions?slide=${encodeURIComponent(S.slide)}&user=${encodeURIComponent(S.user)}`);
  S.versions = await r.json();
  const sel = $('version-select');
  sel.innerHTML = '<option value="">-- select version --</option>';
  S.versions.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.id;
    opt.textContent = `${v.timestamp} (${v.marked_count} labeled)`;
    sel.appendChild(opt);
  });
}

async function doLoadVersion() {
  const vid = $('version-select').value;
  if (!vid || !S.user || !S.slide) return;
  const r = await api('/api/load_version', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({version_id: vid, user: S.user, slide: S.slide}),
  });
  const j = await r.json();
  if (r.ok && S.umap) {
    S.umap.marked = j.marked;
    S.umap.associated_data = j.associated_data;
    S.umap.colors = j.colors;
    if ($('umap-scatter').data) Plotly.restyle('umap-scatter', {'marker.color': [getUmapColors()]});
    if (S.view === 'slide') drawSlideCanvas();
    updateLabeledCount();
  }
}

// ── Recluster ──────────────────────────────────────────────────────────────────
async function doRecluster() {
  if (!S.slide || S.reclustering) return;
  S.reclustering = true;
  $('recluster-btn').disabled = true;
  $('recluster-prog').style.display = 'block';
  $('prog-fill').style.width = '0%';
  $('prog-lbl').textContent = '0%';

  const state = S.umap
    ? {tile_nums: S.umap.tile_nums, marked: S.umap.marked, associated_data: S.umap.associated_data}
    : {};
  const r = await api('/api/recluster', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({slide: S.slide, user: S.user || '', state}),
  });
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    $('prog-lbl').textContent = 'Error: ' + (j.error || r.statusText);
    $('prog-fill').style.background = 'var(--red)';
    S.reclustering = false;
    $('recluster-btn').disabled = false;
    return;
  }

  const src = new EventSource(
    `/api/recluster_stream?slide=${encodeURIComponent(S.slide)}&user=${encodeURIComponent(S.user || '')}`);
  src.onmessage = async e => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'progress') {
      $('prog-fill').style.width = msg.progress + '%';
      $('prog-fill').style.background = 'var(--grn)';
      $('prog-lbl').textContent = msg.progress.toFixed(1) + '%' + (msg.message ? ' - ' + msg.message : '');
    } else if (msg.type === 'done') {
      src.close();
      S.reclustering = false;
      $('recluster-btn').disabled = false;
      if (msg.success) {
        $('prog-fill').style.width = '100%';
        $('prog-lbl').textContent = 'Done!';
        setTimeout(() => { $('recluster-prog').style.display = 'none'; }, 2500);
        S.umapStatic = null;
        await loadUmapData();
        if (S.view === 'grid')  await loadTiles();
        if (S.view === 'slide') drawSlideCanvas();
      } else {
        $('prog-fill').style.width = '100%';
        $('prog-fill').style.background = 'var(--red)';
        $('prog-lbl').textContent = 'Failed: ' + (msg.error || 'check server logs');
      }
    }
  };
  src.onerror = () => {
    src.close();
    S.reclustering = false;
    $('recluster-btn').disabled = false;
    $('prog-fill').style.background = 'var(--red)';
    $('prog-lbl').textContent = 'Stream error';
  };
}

window.addEventListener('resize', () => {
  if (S.view === 'umap' && S.umap) Plotly.relayout('umap-scatter', {autosize: true});
});

// ── Image Jump Logic (padded 512x512 crop is always centered at 0.5, 0.5) ─────
function onRawImageClick(e) {
  if (!currentTile || !S.slide || !S.centroids || !S.umap) return;
  const currentTn = currentTile.tile_num;
  const center = S.centroids[currentTn];
  if (!center) return;

  const rect = e.target.getBoundingClientRect();
  const pctX = (e.clientX - rect.left) / rect.width;
  const pctY = (e.clientY - rect.top) / rect.height;

  const globalX = center.x + (pctX - 0.5) * (TILE2_CROP_HALF * 2);
  const globalY = center.y + (pctY - 0.5) * (TILE2_CROP_HALF * 2);

  let closestTn = null;
  let minDist = Infinity;
  S.umap.tile_nums.forEach(tn => {
    const c = S.centroids[tn];
    if (!c) return;
    const dx = c.x - globalX;
    const dy = c.y - globalY;
    const distSq = dx * dx + dy * dy;
    if (distSq < minDist) { minDist = distSq; closestTn = tn; }
  });

  if (closestTn !== null && closestTn !== currentTn) {
    const idx = S.tileNumToIdx[closestTn];
    if (idx === undefined) return;
    const paneRaw = S.umap.imgs_pane ? S.umap.imgs_pane[idx] : S.umap.imgs[idx];
    const labels = {};
    LABEL_COLS.forEach(col => {
      labels[col] = LABEL_STR[col][S.umap.associated_data[col][idx]] || 'None';
    });
    openEditPanel({
      tile_num: closestTn, desc: S.umap.desc[idx], source: currentTile.source,
      labels, probs: LABEL_COLS.reduce((o, c) => ({...o, [c]: null}), {}),
      raw_path: paneRaw,
    });
  }
}

// ── Manual cell addition ───────────────────────────────────────────────────────
let _lassoActive = false;

function screenToImageCoords(sx, sy) {
  const plotEl = $('slide-plotly');
  if (!plotEl._fullLayout) return null;
  const fl = plotEl._fullLayout;
  const [x0, x1] = fl.xaxis.range;
  const [y0, y1] = fl.yaxis.range;
  const xs = fl._size.w / (x1 - x0);
  const ys = fl._size.h / (y1 - y0);
  const imgX     = x0 + (sx - fl._size.l) / xs;
  const flippedY = y1 - (sy - fl._size.t) / ys;
  return [imgX, S.origH - flippedY];
}

function _onLassoStart(e) {
  if (e.button !== 0) return;
  _lassoActive = true;
  S.polygonPoints = [];
  const coords = screenToImageCoords(e.offsetX, e.offsetY);
  if (coords) S.polygonPoints.push(coords);
}

function _onLassoMove(e) {
  if (!_lassoActive) return;
  const coords = screenToImageCoords(e.offsetX, e.offsetY);
  if (!coords) return;
  if (S.polygonPoints.length > 0) {
    const fl = $('slide-plotly')._fullLayout;
    const xs = fl._size.w / (fl.xaxis.range[1] - fl.xaxis.range[0]);
    const ys = fl._size.h / (fl.yaxis.range[1] - fl.yaxis.range[0]);
    const last = S.polygonPoints[S.polygonPoints.length - 1];
    const dsx = (coords[0] - last[0]) * xs;
    const dsy = (coords[1] - last[1]) * ys;
    if (dsx * dsx + dsy * dsy < 9) return;
  }
  S.polygonPoints.push(coords);
  drawSlideCanvas();
}

function _onLassoEnd() {
  if (!_lassoActive) return;
  _lassoActive = false;
  if (S.polygonPoints.length < 3) {
    S.polygonPoints = [];
    drawSlideCanvas();
    return;
  }
  submitManualCell();
}

function toggleAddCellMode() {
  const canvas = $('slide-canvas');
  const btn    = $('add-cell-btn');
  const hint   = $('add-cell-hint');

  S.drawingMode     = !S.drawingMode;
  S.polygonPoints   = [];
  S.polygonMousePos = null;
  _lassoActive      = false;

  if (S.drawingMode) {
    canvas.style.pointerEvents = 'all';
    canvas.style.cursor        = 'crosshair';
    btn.style.background       = 'var(--amb)';
    btn.style.color            = '#000';
    btn.style.borderColor      = 'var(--amb)';
    btn.textContent            = 'CANCEL ADD';
    hint.style.display         = '';
    Plotly.relayout('slide-plotly', {dragmode: false});
    canvas.addEventListener('mousedown', _onLassoStart);
    canvas.addEventListener('mousemove', _onLassoMove);
    canvas.addEventListener('mouseup',   _onLassoEnd);
  } else {
    canvas.style.pointerEvents = 'none';
    canvas.style.cursor        = '';
    btn.style.background  = '';
    btn.style.color       = '';
    btn.style.borderColor = '';
    btn.textContent = 'ADD CELL';
    hint.style.display         = 'none';
    Plotly.relayout('slide-plotly', {dragmode: 'pan'});
    canvas.removeEventListener('mousedown', _onLassoStart);
    canvas.removeEventListener('mousemove', _onLassoMove);
    canvas.removeEventListener('mouseup',   _onLassoEnd);
    drawSlideCanvas();
  }
}

async function submitManualCell() {
  const polygon = S.polygonPoints.slice();
  _lassoActive = false;
  toggleAddCellMode();
  loading(true);
  try {
    const r = await api('/api/add_cell', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({slide: S.slide, polygon}),
    });
    let j = {};
    try { j = await r.json(); } catch (_) {}
    loading(false);
    if (!r.ok) { alert('Add cell failed: ' + (j.error || `HTTP ${r.status}`)); return; }
    insertManualCell(j);
  } catch (err) {
    loading(false);
    alert('Add cell failed: ' + err.message);
  }
}

async function doDeleteCell() {
  if (!currentTile || !S.slide) return;
  const tn = currentTile.tile_num;
  if (!S.manualTileNums.has(tn)) return;
  if (!confirm(`Delete manually added Tile ${tn}? This cannot be undone.`)) return;

  loading(true);
  try {
    const r = await api('/api/delete_cell', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({slide: S.slide, tile_num: tn}),
    });
    let j = {};
    try { j = await r.json(); } catch (_) {}
    loading(false);
    if (!r.ok) { alert('Delete failed: ' + (j.error || `HTTP ${r.status}`)); return; }
  } catch (err) {
    loading(false);
    alert('Delete failed: ' + err.message);
    return;
  }

  const idx = S.tileNumToIdx[tn];
  if (idx !== undefined && S.umap) {
    S.umap.x[idx]      = NaN;
    S.umap.y[idx]      = NaN;
    S.umap.marked[idx] = 0;
  }
  delete S.centroids[tn];
  S.manualTileNums.delete(tn);
  S.umapStatic = null;

  closePanel();
  if ($('umap-scatter').data) renderUmap();
  drawSlideCanvas();
  syncScatterTrace();
  updateLabeledCount();
  await refreshSlideBadges();
  if (S.view === 'grid') loadTiles();
}

function insertManualCell(result) {
  const { tile_num, centroid, umap_xy, img_path, img_pane_path, cell_area, nc_ratio } = result;
  S.centroids[tile_num] = centroid;

  const idx = S.umap.tile_nums.length;
  S.umap.tile_nums.push(tile_num);
  S.umap.x.push(umap_xy[0]);
  S.umap.y.push(umap_xy[1]);
  S.umap.marked.push(0);
  S.umap.colors.push('royalblue');
  S.umap.desc.push(`Tile ${tile_num}`);
  S.umap.imgs.push(img_path);
  if (S.umap.imgs_pane)     S.umap.imgs_pane.push(img_pane_path);
  if (S.umap.cell_sizes)    S.umap.cell_sizes.push(cell_area);
  if (S.umap.nc_ratios)     S.umap.nc_ratios.push(nc_ratio);
  if (S.umap.nuclear_sizes) S.umap.nuclear_sizes.push(0);
  LABEL_COLS.forEach(col => { S.umap.associated_data[col].push(0); });

  S.tileNumToIdx[tile_num] = idx;
  S.manualTileNums.add(tile_num);

  if ($('umap-scatter').data) renderUmap();
  drawSlideCanvas();
  if (S.view === 'slide') syncScatterTrace();
  updateLabeledCount();

  openEditPanel({
    tile_num,
    desc:   `Tile ${tile_num}`,
    source: 'manual',
    labels: LABEL_COLS.reduce((o, c) => ({...o, [c]: 'None'}), {}),
    probs:  LABEL_COLS.reduce((o, c) => ({...o, [c]: null}), {}),
    raw_path: img_pane_path,
  });
}

// ── Keyboard ───────────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if ($('upload-modal').classList.contains('open')) {
    if (e.key === 'Escape') closeUploadModal();
    return;
  }
  if (e.key === 'Escape') {
    if (S.drawingMode) { toggleAddCellMode(); return; }
    closePanel();
  }
  if (S.drawingMode) return;
  if (document.activeElement && ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if (e.key === 'ArrowRight' && currentTile) doNext();
  if (e.key === 'ArrowLeft'  && currentTile) doPrev();
  if (e.key === 'Enter'      && currentTile) doSave();
});

init();
