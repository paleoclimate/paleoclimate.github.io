"""
Spatial similarity comparison between Floegel reference paleoclimate maps (JPG,
no georeference) and the project-generated KNN+IDW GeoTIFFs.

Pipeline (see COMPARISON plan):
  1. render-reference : rasterize the GeoTIFF into a class-colored PNG on the
     raster's native grid, draw the coastline on top, and emit a self-contained
     GCP picker HTML.
  2. (manual)         : open COMPARISON/gcp_picker.html, mark matching control
     points on the Floegel image and the reference render, export gcp_<age>.json.
  3. compare          : warp the Floegel image onto the GeoTIFF grid (thin-plate
     spline, homography fallback), reclassify both sides into
     Dry / Semi-arid / Humid, build a common land mask and compute per-class IoU,
     overall accuracy and Cohen's kappa. Emits CSV + HTML/PNG report.

Color de-para (Floegel legend left panel -> 3 classes):
  dark green + light green      -> Semi-arid (2)
  orange/tan (evaporites) + yellow (arid) -> Dry (1)
  pink/magenta                  -> Humid (3)
  light blue (ocean), gray/black (symbols, grid, labels), white -> ignored
"""

import argparse
import base64
import csv
import json
import os

import cv2
import numpy as np
import rasterio
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist

# Reuse helpers from the main renderer so behaviour stays consistent.
from render_map import (
    apply_paleo_reference_frame_correction,
    load_geojson,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMP_DIR = os.path.join(BASE_DIR, 'COMPARISON')
INPUT_DIR = os.path.join(COMP_DIR, 'inputs')
GEOTIFF_DIR = os.path.join(BASE_DIR, 'GENERATED_GEOTIFFS')
GEOJSON_DIR = os.path.join(BASE_DIR, 'GEOJSON')

GEOTIFF_SUFFIX = 'knn_idw_power4.0_gradient_sharp18.0'

# Class codes
DRY, SEMI, HUMID = 1, 2, 3
CLASS_NAMES = {DRY: 'Seco', SEMI: 'Semi-arido', HUMID: 'Umido'}
CLASS_ORDER = [DRY, SEMI, HUMID]

# Class render colors (RGB), matching render_map.py's climate palette.
CLASS_RGB = {
    DRY: (234, 245, 29),    # yellow
    SEMI: (10, 122, 24),    # dark green
    HUMID: (7, 152, 219),   # blue
}

AGES = ('115', '105')


def geotiff_path(age):
    return os.path.join(GEOTIFF_DIR, f'{age}_ma_{GEOTIFF_SUFFIX}.tif')


def floegel_path(age):
    return os.path.join(INPUT_DIR, f'{age}_floegel.png')


def costa_path(age):
    return os.path.join(GEOJSON_DIR, f'{age}_ma_costa.geojson')


def reference_path(age):
    return os.path.join(COMP_DIR, f'reference_render_{age}.png')


def gcp_path(age):
    return os.path.join(COMP_DIR, f'gcp_{age}.json')


def warped_path(age):
    return os.path.join(COMP_DIR, f'warped_floegel_{age}.png')


def classes_path(age):
    return os.path.join(COMP_DIR, f'classes_{age}.png')


def metrics_path(age):
    return os.path.join(COMP_DIR, f'metrics_{age}.csv')


def report_path(age):
    return os.path.join(COMP_DIR, f'comparison_report_{age}.html')


# ---------------------------------------------------------------------------
# Legend prototypes / Floegel color classification
# ---------------------------------------------------------------------------

# Extra prototypes for pixels that must be ignored (not a climate zone).
# (rgb, label) where label is a class code or None (ignore).
IGNORE_PROTOTYPES = [
    ((187, 219, 244), None),  # ocean light blue
    ((168, 216, 240), None),  # ocean light blue (variant)
    ((150, 200, 235), None),  # ocean / river blue
    ((120, 160, 210), None),  # darker coastline blue
    ((255, 255, 255), None),  # white background
    ((245, 245, 245), None),  # near-white
    ((48, 48, 48), None),     # dark symbols / labels / coast line
    ((95, 95, 93), None),     # gray graticule / arrows
    ((150, 150, 150), None),  # mid gray
]


def extract_legend_prototypes(legend_img_path):
    """Auto-detect the 5 swatch colors on the legend's left panel.

    Scans a vertical strip on the left, groups contiguous saturated rows into
    swatch bands and takes the median color of each band. Returns a list of
    (rgb_tuple, class_code) following the de-para. Falls back to hardcoded
    values (sampled from the provided legend) if detection is off.
    """
    fallback = [
        ((49, 147, 74), SEMI),    # dark green  -> Semi-arid
        ((133, 184, 65), SEMI),   # light green -> Semi-arid
        ((228, 188, 119), DRY),   # orange/tan  -> Dry
        ((241, 235, 89), DRY),    # yellow      -> Dry
        ((207, 128, 176), HUMID), # pink        -> Humid
    ]

    img = cv2.imread(legend_img_path)
    if img is None:
        return fallback
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    xcol = int(w * 0.05)

    bands = []
    current = []
    for y in range(h):
        r, g, b = (int(v) for v in img[y, xcol])
        is_white = r > 230 and g > 230 and b > 230
        is_black = r < 40 and g < 40 and b < 40
        if not is_white and not is_black:
            current.append((r, g, b))
        else:
            if len(current) >= 6:
                bands.append(current)
            current = []
    if len(current) >= 6:
        bands.append(current)

    # Expect exactly the 5 swatches, top-to-bottom order matches the de-para.
    order = [SEMI, SEMI, DRY, DRY, HUMID]
    if len(bands) != 5:
        return fallback

    prototypes = []
    for band, cls in zip(bands, order):
        arr = np.array(band)
        med = tuple(int(v) for v in np.median(arr, axis=0))
        prototypes.append((med, cls))
    return prototypes


def _rgb_to_lab(rgb_uint8):
    """rgb_uint8: (..., 3) uint8 -> (..., 3) float Lab."""
    arr = rgb_uint8.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab.reshape(-1, 3)


def classify_floegel(rgb_image, prototypes, valid_mask=None):
    """Classify an RGB image into class codes using nearest color prototype (Lab).

    rgb_image : (H, W, 3) uint8
    prototypes: list of (rgb, class_code) for climate zones
    valid_mask: optional (H, W) bool; False pixels become 0 (no-data)
    Returns   : (H, W) int array with values in {0,1,2,3} (0 = ignored/no-data).
    """
    all_proto = list(prototypes) + IGNORE_PROTOTYPES
    proto_rgb = np.array([p[0] for p in all_proto], dtype=np.uint8)
    proto_label = [p[1] for p in all_proto]  # class code or None
    proto_lab = _rgb_to_lab(proto_rgb)

    h, w, _ = rgb_image.shape
    pix_lab = _rgb_to_lab(rgb_image)
    dists = cdist(pix_lab, proto_lab)
    nearest = np.argmin(dists, axis=1)

    label_lut = np.array([lbl if lbl is not None else 0 for lbl in proto_label])
    out = label_lut[nearest].reshape(h, w).astype(np.int32)

    if valid_mask is not None:
        out[~valid_mask] = 0
    return out


# ---------------------------------------------------------------------------
# Reference render (GeoTIFF class map + coastline) on native grid
# ---------------------------------------------------------------------------

def geotiff_classes(age):
    """Read the GeoTIFF and threshold continuous values into class codes.

    Returns (classes (H,W) int, src_meta dict with transform + shape).
    Bins: <1.5 Dry, [1.5,2.5) Semi-arid, >=2.5 Humid.
    """
    with rasterio.open(geotiff_path(age)) as src:
        data = src.read(1).astype(np.float64)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        height, width = data.shape

    classes = np.full(data.shape, SEMI, dtype=np.int32)
    classes[data < 1.5] = DRY
    classes[(data >= 1.5) & (data < 2.5)] = SEMI
    classes[data >= 2.5] = HUMID

    meta = {
        'transform': transform,
        'crs': crs,
        'bounds': bounds,
        'height': height,
        'width': width,
    }
    return classes, meta


def classes_to_rgb(classes):
    """Map a class-code array (0..3) to an RGB image. 0 -> white."""
    h, w = classes.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    for code, color in CLASS_RGB.items():
        rgb[classes == code] = color
    return rgb


def _lonlat_to_colrow(transform, lons, lats):
    """Vectorized inverse affine: (lon,lat) -> fractional (col,row)."""
    inv = ~transform
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    cols = inv.a * lons + inv.b * lats + inv.c
    rows = inv.d * lons + inv.e * lats + inv.f
    return cols, rows


def _iter_linestrings(geom):
    gtype = geom.get('type')
    coords = geom.get('coordinates')
    if not coords:
        return
    if gtype == 'LineString':
        yield coords
    elif gtype == 'MultiLineString':
        for line in coords:
            yield line


def draw_coastline(rgb_image, meta, age):
    """Draw the (Euler-corrected) coastline for `age` onto the reference image."""
    coast = load_geojson(costa_path(age))
    coast = apply_paleo_reference_frame_correction(coast, f'{age}_ma')
    transform = meta['transform']

    for feature in coast.get('features', []):
        geom = feature.get('geometry', {})
        for line in _iter_linestrings(geom):
            lons = [c[0] for c in line]
            lats = [c[1] for c in line]
            cols, rows = _lonlat_to_colrow(transform, lons, lats)
            pts = np.column_stack([cols, rows]).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(rgb_image, [pts], isClosed=False,
                              color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return rgb_image


def render_reference(age):
    """Build reference_render_<age>.png = class colors + coastline (native grid)."""
    classes, meta = geotiff_classes(age)
    rgb = classes_to_rgb(classes)
    rgb = draw_coastline(rgb, meta, age)
    # cv2 writes BGR
    cv2.imwrite(reference_path(age), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"[render-reference] {age} Ma -> {reference_path(age)} "
          f"({meta['width']}x{meta['height']})")
    return meta


# ---------------------------------------------------------------------------
# GCP picker HTML
# ---------------------------------------------------------------------------

def _img_to_data_uri(path):
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'data:image/png;base64,{b64}'


def generate_gcp_picker():
    """Emit a self-contained COMPARISON/gcp_picker.html with images embedded."""
    payload = {}
    for age in AGES:
        fp = floegel_path(age)
        rp = reference_path(age)
        if os.path.exists(fp) and os.path.exists(rp):
            payload[age] = {
                'floegel': _img_to_data_uri(fp),
                'reference': _img_to_data_uri(rp),
            }
    if not payload:
        print("[pick-gcps] No reference renders found; run render-reference first.")
        return

    data_json = json.dumps(payload)
    html = _GCP_PICKER_TEMPLATE.replace('__DATA__', data_json)
    out = os.path.join(COMP_DIR, 'gcp_picker.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[pick-gcps] GCP picker -> {out}")
    print("            Open it, mark >= 6 matching points per age (8-12 recommended),")
    print("            then download gcp_<age>.json into the COMPARISON/ folder.")


_GCP_PICKER_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="UTF-8">
<title>GCP Picker - Floegel vs GeoTIFF</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: Arial, sans-serif; margin: 0; background: #ecf0f1; }
  .toolbar { background: #2c3e50; color: #fff; padding: 8px 14px; display: flex;
             gap: 14px; align-items: center; flex-wrap: wrap; }
  .toolbar b { font-size: 15px; }
  .toolbar select, .toolbar button { padding: 4px 8px; font-size: 13px; border-radius: 4px;
             border: 1px solid #7f8c8d; }
  .toolbar button { background: #3498db; color: #fff; cursor: pointer; font-weight: bold; }
  .toolbar button.danger { background: #e74c3c; }
  .toolbar .hint { font-size: 12px; color: #bdc3c7; }
  .panes { display: flex; gap: 10px; padding: 10px; align-items: flex-start; }
  .pane { flex: 1; background: #fff; border-radius: 6px; padding: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.2); }
  .pane h3 { margin: 4px 6px; font-size: 14px; color: #2c3e50; }
  .canvas-wrap { position: relative; width: 100%; overflow: auto; border: 1px solid #ddd; }
  canvas { display: block; width: 100%; height: auto; cursor: crosshair; }
  .side { width: 240px; }
  .side ol { padding-left: 20px; font-size: 12px; max-height: 60vh; overflow: auto; }
  .status { font-size: 12px; margin: 6px; padding: 6px; background: #f8f9fa; border-radius: 4px; }
  .next { font-weight: bold; color: #e67e22; }
</style>
</head>
<body>
<div class="toolbar">
  <b>GCP Picker</b>
  <label>Idade:
    <select id="ageSel"></select>
  </label>
  <button id="undoBtn">Desfazer ultimo</button>
  <button id="clearBtn" class="danger">Limpar</button>
  <button id="exportBtn">Exportar gcp_&lt;idade&gt;.json</button>
  <span class="hint">Clique 1x no mapa Floegel (esquerda), depois no ponto equivalente no render (direita). Repita 6-12x.</span>
</div>
<div class="panes">
  <div class="pane">
    <h3>Floegel (referencia publicada)</h3>
    <div class="canvas-wrap"><canvas id="cvFloegel"></canvas></div>
  </div>
  <div class="pane">
    <h3>Render do GeoTIFF (meu mapa)</h3>
    <div class="canvas-wrap"><canvas id="cvRef"></canvas></div>
  </div>
  <div class="pane side">
    <h3>Pontos de controle</h3>
    <div class="status" id="status"></div>
    <ol id="list"></ol>
  </div>
</div>
<script>
var DATA = __DATA__;
var state = {}; // age -> {pairs:[{floegel:[x,y], ref:[x,y]}], pending:null}
var imgs = {};  // age -> {floegel:Image, reference:Image}
var curAge = null;

function ages() { return Object.keys(DATA); }

function initAge(age) {
  if (!state[age]) state[age] = { pairs: [], pending: null };
  if (!imgs[age]) {
    imgs[age] = {};
    ['floegel', 'reference'].forEach(function(k) {
      var im = new Image();
      im.onload = function() { if (age === curAge) redraw(); };
      im.src = DATA[age][k];
      imgs[age][k] = im;
    });
  }
}

function setupCanvas(canvas, img) {
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
}

function toNatural(canvas, evt) {
  var rect = canvas.getBoundingClientRect();
  var x = (evt.clientX - rect.left) / rect.width * canvas.width;
  var y = (evt.clientY - rect.top) / rect.height * canvas.height;
  return [Math.round(x), Math.round(y)];
}

function drawMarkers(ctx, pts, color) {
  pts.forEach(function(p, i) {
    ctx.beginPath();
    ctx.arc(p[0], p[1], 6, 0, 2 * Math.PI);
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = color; ctx.font = 'bold 16px Arial';
    ctx.fillText(String(i + 1), p[0] + 8, p[1] - 8);
  });
}

function redraw() {
  var cvF = document.getElementById('cvFloegel');
  var cvR = document.getElementById('cvRef');
  var imF = imgs[curAge].floegel, imR = imgs[curAge].reference;
  if (!imF.complete || !imR.complete) return;
  setupCanvas(cvF, imF); setupCanvas(cvR, imR);
  var cF = cvF.getContext('2d'), cR = cvR.getContext('2d');
  cF.drawImage(imF, 0, 0); cR.drawImage(imR, 0, 0);
  var st = state[curAge];
  drawMarkers(cF, st.pairs.map(function(p){return p.floegel;}), '#e74c3c');
  drawMarkers(cR, st.pairs.map(function(p){return p.ref;}), '#e74c3c');
  if (st.pending) {
    cF.beginPath(); cF.arc(st.pending[0], st.pending[1], 6, 0, 2*Math.PI);
    cF.strokeStyle = '#27ae60'; cF.lineWidth = 3; cF.stroke();
  }
  updateStatus();
}

function updateStatus() {
  var st = state[curAge];
  var s = 'Idade: ' + curAge + ' Ma<br>Pares: ' + st.pairs.length + '<br>';
  s += st.pending ? '<span class="next">Agora clique no RENDER (direita)</span>'
                  : '<span class="next">Clique no FLOEGEL (esquerda)</span>';
  document.getElementById('status').innerHTML = s;
  var ol = document.getElementById('list');
  ol.innerHTML = '';
  st.pairs.forEach(function(p) {
    var li = document.createElement('li');
    li.textContent = 'F(' + p.floegel + ')  R(' + p.ref + ')';
    ol.appendChild(li);
  });
}

document.getElementById('cvFloegel').addEventListener('click', function(e) {
  var st = state[curAge];
  st.pending = toNatural(this, e);
  redraw();
});
document.getElementById('cvRef').addEventListener('click', function(e) {
  var st = state[curAge];
  if (!st.pending) { alert('Clique primeiro no mapa Floegel (esquerda).'); return; }
  var ref = toNatural(this, e);
  st.pairs.push({ floegel: st.pending, ref: ref });
  st.pending = null;
  redraw();
});
document.getElementById('undoBtn').addEventListener('click', function() {
  var st = state[curAge];
  if (st.pending) { st.pending = null; }
  else { st.pairs.pop(); }
  redraw();
});
document.getElementById('clearBtn').addEventListener('click', function() {
  if (!confirm('Limpar todos os pontos desta idade?')) return;
  state[curAge] = { pairs: [], pending: null };
  redraw();
});
document.getElementById('exportBtn').addEventListener('click', function() {
  var st = state[curAge];
  if (st.pairs.length < 4) { alert('Marque pelo menos 4 pares (6-12 recomendado).'); return; }
  var out = { age: curAge, points: st.pairs };
  var blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'gcp_' + curAge + '.json';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
});

(function() {
  var sel = document.getElementById('ageSel');
  ages().forEach(function(a) {
    initAge(a);
    var o = document.createElement('option'); o.value = a; o.textContent = a + ' Ma';
    sel.appendChild(o);
  });
  sel.addEventListener('change', function() { curAge = this.value; redraw(); });
  curAge = ages()[0];
  redraw();
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Warp
# ---------------------------------------------------------------------------

def load_gcps(age):
    path = gcp_path(age)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pairs = data.get('points', [])
    ref = np.array([p['ref'] for p in pairs], dtype=np.float64)       # (n,2) [x=col,y=row]
    flo = np.array([p['floegel'] for p in pairs], dtype=np.float64)   # (n,2) [x,y]
    return ref, flo


def build_warp_maps(ref_pts, flo_pts, width, height):
    """Return (map_x, map_y) float32 arrays (H,W) sampling Floegel for each ref cell.

    ref_pts, flo_pts: (n,2) arrays of [x=col, y=row]. Uses thin-plate-spline RBF
    (target->source). Falls back to homography when fewer than 6 points.
    """
    xs = np.arange(width)
    ys = np.arange(height)
    gx, gy = np.meshgrid(xs, ys)  # (H,W)
    grid = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)

    n = len(ref_pts)
    if n >= 6:
        rbf = RBFInterpolator(ref_pts, flo_pts, kernel='thin_plate_spline',
                              smoothing=0.0)
        mapped = rbf(grid)  # (H*W, 2)
        method = 'thin_plate_spline'
    else:
        H, _ = cv2.findHomography(ref_pts, flo_pts, method=0)
        ones = np.ones((grid.shape[0], 1))
        homog = np.hstack([grid, ones])
        proj = (H @ homog.T).T
        mapped = proj[:, :2] / proj[:, 2:3]
        method = 'homography'

    map_x = mapped[:, 0].reshape(height, width).astype(np.float32)
    map_y = mapped[:, 1].reshape(height, width).astype(np.float32)
    return map_x, map_y, method


def warp_floegel(age, meta):
    """Warp the Floegel image onto the GeoTIFF grid. Returns (warped_rgb, valid_mask)."""
    gcps = load_gcps(age)
    if gcps is None:
        raise FileNotFoundError(
            f"GCP file not found: {gcp_path(age)}. Run the picker "
            f"(COMPARISON/gcp_picker.html) and export the JSON first."
        )
    ref_pts, flo_pts = gcps
    width, height = meta['width'], meta['height']

    flo_bgr = cv2.imread(floegel_path(age))
    flo_rgb = cv2.cvtColor(flo_bgr, cv2.COLOR_BGR2RGB)
    fh, fw, _ = flo_rgb.shape

    map_x, map_y, method = build_warp_maps(ref_pts, flo_pts, width, height)

    warped = cv2.remap(flo_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    # Valid where the sampled source coordinate lies inside the Floegel image.
    valid = (map_x >= 0) & (map_x < fw) & (map_y >= 0) & (map_y < fh)

    cv2.imwrite(warped_path(age), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    print(f"[compare] {age} Ma warp={method} ({len(ref_pts)} GCPs) -> {warped_path(age)}")
    return warped, valid


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_matrix(ref_classes, flo_classes, mask):
    """3x3 confusion matrix over `mask`. Rows = GeoTIFF (ref), cols = Floegel."""
    cm = np.zeros((3, 3), dtype=np.int64)
    r = ref_classes[mask]
    f = flo_classes[mask]
    for i, rc in enumerate(CLASS_ORDER):
        for j, fc in enumerate(CLASS_ORDER):
            cm[i, j] = np.sum((r == rc) & (f == fc))
    return cm


def compute_metrics(cm):
    total = cm.sum()
    metrics = {'total_pixels': int(total)}
    if total == 0:
        return metrics

    diag = np.trace(cm)
    metrics['overall_accuracy'] = diag / total

    # Cohen's kappa
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = np.sum(row * col) / (total * total)
    po = diag / total
    metrics['kappa'] = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    per_class = {}
    for i, code in enumerate(CLASS_ORDER):
        tp = cm[i, i]
        fp = col[i] - tp
        fn = row[i] - tp
        union = tp + fp + fn
        iou = tp / union if union else 0.0
        recall = tp / row[i] if row[i] else 0.0        # ref coverage matched
        precision = tp / col[i] if col[i] else 0.0
        per_class[code] = {
            'iou': iou, 'recall': recall, 'precision': precision,
            'ref_pixels': int(row[i]), 'floegel_pixels': int(col[i]),
        }
    metrics['per_class'] = per_class
    return metrics


def class_area_pct(classes, mask):
    """Percentage of each class within the mask (areal composition)."""
    sel = classes[mask]
    total = sel.size
    out = {}
    for code in CLASS_ORDER:
        out[code] = (np.sum(sel == code) / total * 100.0) if total else 0.0
    return out


def save_metrics_csv(age, cm, metrics, ref_pct, flo_pct):
    with open(metrics_path(age), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([f'Comparacao Floegel x GeoTIFF - {age} Ma'])
        w.writerow([])
        w.writerow(['Matriz de confusao (linhas=GeoTIFF, colunas=Floegel)'])
        w.writerow([''] + [CLASS_NAMES[c] for c in CLASS_ORDER])
        for i, rc in enumerate(CLASS_ORDER):
            w.writerow([CLASS_NAMES[rc]] + [int(cm[i, j]) for j in range(3)])
        w.writerow([])
        w.writerow(['Similaridade por classe'])
        w.writerow(['Classe', 'IoU (%)', 'Recall (%)', 'Precisao (%)',
                    'Area GeoTIFF (%)', 'Area Floegel (%)'])
        pc = metrics.get('per_class', {})
        for c in CLASS_ORDER:
            d = pc.get(c, {})
            w.writerow([
                CLASS_NAMES[c],
                f"{d.get('iou', 0) * 100:.2f}",
                f"{d.get('recall', 0) * 100:.2f}",
                f"{d.get('precision', 0) * 100:.2f}",
                f"{ref_pct.get(c, 0):.2f}",
                f"{flo_pct.get(c, 0):.2f}",
            ])
        w.writerow([])
        w.writerow(['Similaridade geral'])
        w.writerow(['Acuracia global (%)', f"{metrics.get('overall_accuracy', 0) * 100:.2f}"])
        w.writerow(['Kappa de Cohen', f"{metrics.get('kappa', 0):.4f}"])
        w.writerow(['Pixels comparados', metrics.get('total_pixels', 0)])
    print(f"[compare] metrics -> {metrics_path(age)}")


# ---------------------------------------------------------------------------
# Report (PNG composite + HTML)
# ---------------------------------------------------------------------------

def save_class_composite(age, ref_classes, flo_classes, mask, cm):
    """Save a PNG figure: reference classes | floegel classes | confusion matrix."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # 0 no-data -> white, 1 dry -> yellow, 2 semi -> green, 3 humid -> blue
    cmap = ListedColormap([
        (1, 1, 1),
        tuple(c / 255 for c in CLASS_RGB[DRY]),
        tuple(c / 255 for c in CLASS_RGB[SEMI]),
        tuple(c / 255 for c in CLASS_RGB[HUMID]),
    ])

    flo_masked = flo_classes.copy()
    flo_masked[~mask] = 0
    ref_masked = ref_classes.copy()
    ref_masked[~mask] = 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(ref_masked, cmap=cmap, vmin=0, vmax=3)
    axes[0].set_title(f'GeoTIFF (meu) - {age} Ma')
    axes[1].imshow(flo_masked, cmap=cmap, vmin=0, vmax=3)
    axes[1].set_title('Floegel deformado (warp)')
    for ax in axes[:2]:
        ax.axis('off')

    ax = axes[2]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([CLASS_NAMES[c] for c in CLASS_ORDER], rotation=30, ha='right')
    ax.set_yticklabels([CLASS_NAMES[c] for c in CLASS_ORDER])
    ax.set_xlabel('Floegel'); ax.set_ylabel('GeoTIFF')
    ax.set_title('Matriz de confusao')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(int(cm[i, j])), ha='center', va='center',
                    color='black', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(classes_path(age), dpi=110)
    plt.close(fig)
    print(f"[compare] composite -> {classes_path(age)}")


def _pct(x):
    return f"{x:.2f}%"


def save_report_html(age, cm, metrics, ref_pct, flo_pct, warp_method, n_gcps):
    pc = metrics.get('per_class', {})

    def data_uri(path):
        if os.path.exists(path):
            return _img_to_data_uri(path)
        return ''

    rows_class = ''
    for c in CLASS_ORDER:
        d = pc.get(c, {})
        rows_class += (
            f"<tr><td>{CLASS_NAMES[c]}</td>"
            f"<td>{_pct(d.get('iou', 0) * 100)}</td>"
            f"<td>{_pct(d.get('recall', 0) * 100)}</td>"
            f"<td>{_pct(d.get('precision', 0) * 100)}</td>"
            f"<td>{_pct(ref_pct.get(c, 0))}</td>"
            f"<td>{_pct(flo_pct.get(c, 0))}</td></tr>"
        )

    cm_rows = ''
    for i, rc in enumerate(CLASS_ORDER):
        cells = ''.join(f"<td>{int(cm[i, j])}</td>" for j in range(3))
        cm_rows += f"<tr><th>{CLASS_NAMES[rc]}</th>{cells}</tr>"
    cm_header = ''.join(f"<th>{CLASS_NAMES[c]}</th>" for c in CLASS_ORDER)

    html = f"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="UTF-8">
<title>Comparacao Floegel x GeoTIFF - {age} Ma</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 0; background: #ecf0f1; color: #2c3e50; }}
  header {{ background: #2c3e50; color: #fff; padding: 14px 20px; }}
  header h1 {{ margin: 0; font-size: 20px; }}
  .content {{ padding: 20px; max-width: 1100px; margin: 0 auto; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
  .card {{ background: #fff; border-radius: 8px; padding: 14px 18px; box-shadow: 0 1px 4px rgba(0,0,0,.15);
           flex: 1; min-width: 180px; text-align: center; }}
  .card .big {{ font-size: 30px; font-weight: bold; color: #2980b9; }}
  .card .lbl {{ font-size: 13px; color: #7f8c8d; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; margin-bottom: 20px;
           box-shadow: 0 1px 4px rgba(0,0,0,.15); }}
  th, td {{ border: 1px solid #dfe4ea; padding: 8px 10px; text-align: center; font-size: 14px; }}
  th {{ background: #34495e; color: #fff; }}
  img {{ max-width: 100%; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.15); background:#fff; }}
  .note {{ background: #fef9e7; border-left: 4px solid #f1c40f; padding: 10px 14px; font-size: 13px;
           border-radius: 4px; margin-bottom: 20px; }}
  h2 {{ font-size: 17px; border-bottom: 2px solid #bdc3c7; padding-bottom: 6px; }}
</style>
</head>
<body>
<header><h1>Comparacao de similaridade - Floegel x GeoTIFF ({age} Ma)</h1></header>
<div class="content">
  <div class="cards">
    <div class="card"><div class="big">{_pct(metrics.get('overall_accuracy', 0) * 100)}</div>
      <div class="lbl">Similaridade geral (acuracia global)</div></div>
    <div class="card"><div class="big">{metrics.get('kappa', 0):.3f}</div>
      <div class="lbl">Kappa de Cohen</div></div>
    <div class="card"><div class="big">{metrics.get('total_pixels', 0):,}</div>
      <div class="lbl">Pixels comparados</div></div>
  </div>

  <div class="note">
    <b>Metodo:</b> warp {warp_method} com {n_gcps} pontos de controle. O Floegel foi
    deformado para a grade do GeoTIFF; oceano/simbolos/rotulos foram ignorados. O teto
    de concordancia e limitado pelo descasamento real de linha de costa e projecao entre
    os dois mapas, alem da qualidade/numero dos pontos de controle.
  </div>

  <h2>Similaridade por classe</h2>
  <table>
    <tr><th>Classe</th><th>IoU</th><th>Recall</th><th>Precisao</th>
        <th>Area GeoTIFF</th><th>Area Floegel</th></tr>
    {rows_class}
  </table>

  <h2>Matriz de confusao (linhas = GeoTIFF, colunas = Floegel)</h2>
  <table>
    <tr><th></th>{cm_header}</tr>
    {cm_rows}
  </table>

  <h2>Mapas de classe e matriz</h2>
  <img src="{data_uri(classes_path(age))}" alt="composite">

  <h2>Floegel deformado (validacao visual do alinhamento)</h2>
  <img src="{data_uri(warped_path(age))}" alt="warped floegel">

  <h2>Render de referencia usado</h2>
  <img src="{data_uri(reference_path(age))}" alt="reference render">
</div>
</body>
</html>"""

    with open(report_path(age), 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[compare] report -> {report_path(age)}")


def generate_comparison_index():
    items = []
    for age in AGES:
        if os.path.exists(report_path(age)):
            items.append((age, os.path.basename(report_path(age))))
    if not items:
        return
    links = '\n'.join(
        f'<li><a href="{fn}">Comparacao {age} Ma</a> '
        f'(<a href="{os.path.basename(metrics_path(age))}">CSV</a>)</li>'
        for age, fn in items
    )
    html = f"""<!DOCTYPE html>
<html lang="pt-br"><head><meta charset="UTF-8">
<title>Comparacoes Floegel x GeoTIFF</title>
<style>body{{font-family:Arial;margin:40px;color:#2c3e50}}
h1{{font-size:22px}} li{{margin:8px 0;font-size:15px}}</style></head>
<body><h1>Comparacoes de similaridade - Floegel x GeoTIFF</h1>
<ul>{links}</ul></body></html>"""
    out = os.path.join(COMP_DIR, 'index_comparison.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[compare] index -> {out}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def cmd_render_reference(args):
    os.makedirs(COMP_DIR, exist_ok=True)
    ages = [args.age] if args.age else list(AGES)
    for age in ages:
        if not os.path.exists(geotiff_path(age)):
            print(f"[render-reference] skip {age}: missing {geotiff_path(age)}")
            continue
        render_reference(age)
    generate_gcp_picker()


def cmd_compare(args):
    ages = [args.age] if args.age else list(AGES)
    prototypes = extract_legend_prototypes(os.path.join(INPUT_DIR, 'legenda.png'))
    print(f"[compare] legend prototypes: "
          f"{[(rgb, CLASS_NAMES[c]) for rgb, c in prototypes]}")

    for age in ages:
        if not os.path.exists(reference_path(age)):
            print(f"[compare] {age}: reference render missing, generating it...")
            render_reference(age)

        ref_classes, meta = geotiff_classes(age)
        warped_rgb, valid = warp_floegel(age, meta)

        flo_classes = classify_floegel(warped_rgb, prototypes, valid_mask=valid)

        # Common mask: Floegel pixels that were classified as a climate zone.
        mask = flo_classes > 0

        cm = confusion_matrix(ref_classes, flo_classes, mask)
        metrics = compute_metrics(cm)
        ref_pct = class_area_pct(ref_classes, mask)
        flo_pct = class_area_pct(flo_classes, mask)

        gcps = load_gcps(age)
        n_gcps = len(gcps[0]) if gcps else 0
        warp_method = 'thin_plate_spline' if n_gcps >= 6 else 'homography'

        save_metrics_csv(age, cm, metrics, ref_pct, flo_pct)
        save_class_composite(age, ref_classes, flo_classes, mask, cm)
        save_report_html(age, cm, metrics, ref_pct, flo_pct, warp_method, n_gcps)

        acc = metrics.get('overall_accuracy', 0) * 100
        print(f"[compare] {age} Ma  ->  similaridade geral {acc:.2f}%  "
              f"kappa {metrics.get('kappa', 0):.3f}")

    generate_comparison_index()


def main():
    parser = argparse.ArgumentParser(
        description='Compara mapas Floegel (JPG) com GeoTIFFs gerados (similaridade espacial).')
    sub = parser.add_subparsers(dest='command', required=True)

    p_ref = sub.add_parser('render-reference',
                           help='Gera imagens-alvo (classe+costa) e o gcp_picker.html')
    p_ref.add_argument('--age', choices=AGES, help='Processar apenas esta idade')
    p_ref.set_defaults(func=cmd_render_reference)

    p_pick = sub.add_parser('pick-gcps', help='(Re)gera apenas o gcp_picker.html')
    p_pick.set_defaults(func=lambda a: generate_gcp_picker())

    p_cmp = sub.add_parser('compare', help='Warp + metricas + relatorio')
    p_cmp.add_argument('--age', choices=AGES, help='Processar apenas esta idade')
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
