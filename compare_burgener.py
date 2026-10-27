"""
Spatial similarity between the Burgener et al. (2023) palaeo-Koppen maps and the
project class rasters.

Source: Scotese, Verard, Burgener, Elling & Kocsis (2024), The Cretaceous world,
Geological Society, London, Special Publications 544, doi:10.1144/SP544-2024-28,
Figs. 89-97 (CC BY 4.0). Each figure is Burgener et al. (2023) for one age:

  Fig. 89 145 Ma   Fig. 90 130 Ma   Fig. 91 120 Ma   Fig. 92 105 Ma   Fig. 93 95 Ma
  Fig. 94  90 Ma   Fig. 95  85 Ma   Fig. 96  80 Ma   Fig. 97  70 Ma

``COMPARISON/inputs/{age}_burgener.jpg`` is the embedded image of that figure,
byte for byte as it is stored in the PDF.

Unlike Floegel, these maps need no hand-picked control points. They are global
equirectangular plates (180 W to 180 E, 90 S to 90 N) inside a black frame, so
the frame alone georeferences them. Their palaeogeography is PALEOMAP, ours is
the GPlates model plus the per-age frame correction, so the continents do not sit
in the same place. A rotation plus a shift in degrees, fitted by matching the
Burgener shoreline to our coastline (truncated chamfer distance), takes up that
offset, the way the Floegel control points do. Both scores are written: frame
only, and frame plus shoreline fit (the headline number).

Koppen-Trewartha class -> the project's three classes (standard Koppen B group):
  BWh, BWk                                  -> Dry (1)
  BSh, BSk                                  -> Semi-arid (2)
  Af/Am, As/Aw, Ca, Cb, Cc, Da, Db, Dc/Do   -> Humid (3)
  Marine, frame, labels, graticule          -> ignored
A sensitivity score also counts As/Aw (seasonally dry tropics) as Semi-arid.

Usage:
  python compare_burgener.py compare            # every age with an image
  python compare_burgener.py compare --age 105
"""

import argparse
import csv
import os

import cv2
import numpy as np
from rasterio.transform import from_origin
from scipy.optimize import minimize

import compare_floegel as cf
from compare_floegel import DRY, SEMI, HUMID, CLASS_NAMES, CLASS_ORDER

ATLAS = 'burgener'
AGES = cf.BURGENER_INDEX_AGES
FIGURES = {'145': 89, '130': 90, '120': 91, '105': 92, '95': 93,
           '90': 94, '85': 95, '80': 96, '70': 97}

# Legend swatches of Fig. 2c, sampled from the figure (median RGB of each band).
KOPPEN_LEGEND = (
    ('Dc/Do', (147, 91, 198), HUMID),
    ('Db', (166, 131, 199), HUMID),
    ('Da', (214, 199, 254), HUMID),
    ('Cc', (226, 255, 127), HUMID),
    ('Cb', (203, 255, 128), HUMID),
    ('Ca', (126, 255, 128), HUMID),
    ('BWk', (255, 255, 139), DRY),
    ('BWh', (254, 217, 64), DRY),
    ('BSk', (216, 191, 127), SEMI),
    ('BSh', (217, 170, 80), SEMI),
    ('As/Aw', (64, 154, 64), HUMID),
    ('Af/Am', (65, 101, 65), HUMID),
)
# Colors that are not land: sea, frame/labels/shore, paper.
IGNORED_COLORS = ((230, 230, 230), (0, 0, 0), (255, 255, 255))
SENSITIVITY_OVERRIDES = {'As/Aw': SEMI}
MAX_LAB_DISTANCE = 30.0
MODE_WINDOW = 5             # px; one Burgener cell is about 4 px
MODE_FILL_SHARE = 0.6       # land share that fills an unclassified pixel

FRAME_DARK_SUM = 200        # RGB sum below which a pixel is frame ink
FRAME_ROW_SHARE = 0.6       # a frame row/column is mostly ink
FIT_RES = 0.1               # degrees per cell of the coastline distance grid
FIT_MARGIN = 20.0           # degrees of coastline kept around the raster window
FIT_TRUNCATE = 3.0          # degrees; farther shore pixels count as this far
FIT_WINDOW_PAD = 5.0        # Burgener shore kept this far outside the raster window
FIT_STARTS = [(t, x, y) for t in (-10.0, 0.0, 10.0)
              for x in (-8.0, 0.0, 8.0) for y in (-8.0, 0.0, 8.0)]
FIT_MAX_ROTATION = 20.0
FIT_MAX_SHIFT = 15.0
COARSE_CANDIDATES = 3       # separated minima of a coarse scan refined by Powell


def image_path(age):
    return os.path.join(cf.INPUT_DIR, f'{age}_burgener.jpg')


def images_present(ages=AGES):
    return [age for age in ages if os.path.isfile(image_path(age))]


def summary_path():
    return os.path.join(cf.COMP_DIR, 'burgener_summary.csv')


def load_image(age):
    bgr = cv2.imread(image_path(age))
    if bgr is None:
        raise FileNotFoundError(image_path(age))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def find_frame(rgb):
    """Pixel columns/rows of the black plot frame: (x0, x1, y0, y1)."""
    ink = rgb.astype(np.int32).sum(axis=2) < FRAME_DARK_SUM
    rows = np.where(ink.mean(axis=1) > FRAME_ROW_SHARE)[0]
    cols = np.where(ink.mean(axis=0) > FRAME_ROW_SHARE)[0]
    if rows.size < 2 or cols.size < 2:
        raise RuntimeError('Burgener frame not found')
    return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())


class PlateCarree:
    """Linear pixel <-> lon/lat mapping of an equirectangular figure.

    Pixel (x0, y0) is (west, north); ``dx``/``dy`` are degrees per pixel. The
    corner is provisional when the figure has no labels: the shoreline fit then
    finds the offset.
    """

    def __init__(self, x0, y0, dx, dy, west=-180.0, north=90.0):
        self.x0, self.y0, self.dx, self.dy = float(x0), float(y0), float(dx), float(dy)
        self.west, self.north = float(west), float(north)

    def to_lonlat(self, xs, ys):
        return (self.west + (np.asarray(xs) - self.x0) * self.dx,
                self.north - (np.asarray(ys) - self.y0) * self.dy)

    def to_pixel(self, lons, lats):
        return (self.x0 + (np.asarray(lons) - self.west) / self.dx,
                self.y0 + (self.north - np.asarray(lats)) / self.dy)

    def as_dict(self):
        return dict(x0=self.x0, y0=self.y0, dx=self.dx, dy=self.dy,
                    west=self.west, north=self.north)


def frame_plate(frame):
    """Global plate whose black frame spans 180 W-180 E and 90 N-90 S (Burgener)."""
    x0, x1, y0, y1 = frame
    return PlateCarree(x0, y0, 360.0 / (x1 - x0), 180.0 / (y1 - y0))


def pixel_to_lonlat(frame, xs, ys):
    return frame_plate(frame).to_lonlat(xs, ys)


def lonlat_to_pixel(frame, lons, lats):
    return frame_plate(frame).to_pixel(lons, lats)


def _lab(colors):
    arr = np.asarray(colors, dtype=np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def classify(rgb, overrides=None):
    """Class code per pixel by nearest legend color in Lab (0 = not land/unknown)."""
    overrides = overrides or {}
    codes = [overrides.get(name, code) for name, _rgb, code in KOPPEN_LEGEND]
    protos = _lab([c for _n, c, _k in KOPPEN_LEGEND] + list(IGNORED_COLORS))
    lookup = np.array(codes + [0] * len(IGNORED_COLORS), dtype=np.int32)
    pixels = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    best = np.zeros(pixels.shape[0], dtype=np.int64)
    best_d = np.full(pixels.shape[0], np.inf, dtype=np.float32)
    for index, proto in enumerate(protos):
        d = np.linalg.norm(pixels - proto, axis=1)
        closer = d < best_d
        best[closer] = index
        best_d[closer] = d[closer]
    out = lookup[best]
    out[best_d > MAX_LAB_DISTANCE] = 0
    return out.reshape(rgb.shape[:2])


def classify_cells(rgb, overrides=None):
    """Classify the figure, then take the 5x5 majority class on land.

    Burgener cells are about 4 px wide, so the vote keeps them and removes the
    graticule and anti-aliased edges that cross them. A pixel the legend misses
    (graticule on land) takes the vote when most of its window is land.
    """
    classes = classify(rgb, overrides)
    votes = np.stack([
        cv2.boxFilter((classes == code).astype(np.float32), -1, (MODE_WINDOW, MODE_WINDOW),
                      normalize=True)
        for code in CLASS_ORDER
    ])
    winner = np.array(CLASS_ORDER, dtype=np.int32)[np.argmax(votes, axis=0)]
    land_share = votes.sum(axis=0)
    out = np.where(classes > 0, winner, 0)
    out[(classes == 0) & (land_share >= MODE_FILL_SHARE)] = winner[
        (classes == 0) & (land_share >= MODE_FILL_SHARE)]
    return out.astype(np.int32)


def land_mask(rgb, frame):
    """Burgener land: any legend class inside the frame."""
    x0, x1, y0, y1 = frame
    land = classify(rgb) > 0
    land[:, :x0 + 3] = False
    land[:, x1 - 2:] = False
    land[:y0 + 3, :] = False
    land[y1 - 2:, :] = False
    return cv2.morphologyEx(land.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)) > 0


def _rotate_shift(params, center, lons, lats):
    theta, dlon, dlat = np.radians(params[0]), params[1], params[2]
    u, v = lons - center[0], lats - center[1]
    return (center[0] + np.cos(theta) * u - np.sin(theta) * v + dlon,
            center[1] + np.sin(theta) * u + np.cos(theta) * v + dlat)


def _unrotate_shift(params, center, lons, lats):
    """Inverse of _rotate_shift: our lon/lat -> Burgener lon/lat."""
    theta, dlon, dlat = np.radians(params[0]), params[1], params[2]
    u, v = lons - center[0] - dlon, lats - center[1] - dlat
    return (center[0] + np.cos(theta) * u + np.sin(theta) * v,
            center[1] - np.sin(theta) * u + np.cos(theta) * v)


def window_center(meta):
    b = meta['bounds']
    return ((b.left + b.right) / 2.0, (b.top + b.bottom) / 2.0)


class _ShoreGrid:
    """Truncated distance (degrees) to a shoreline drawn on a lon/lat grid."""

    def __init__(self, shore, west, north):
        self.west, self.north = west, north
        self.height, self.width = shore.shape
        distance = cv2.distanceTransform((~shore).astype(np.uint8), cv2.DIST_L2, 5) * FIT_RES
        self.distance = np.minimum(distance, FIT_TRUNCATE)

    def lookup(self, lons, lats):
        cols = np.floor((lons - self.west) / FIT_RES).astype(np.int64)
        rows = np.floor((self.north - lats) / FIT_RES).astype(np.int64)
        inside = (cols >= 0) & (cols < self.width) & (rows >= 0) & (rows < self.height)
        d = np.full(np.shape(lons), FIT_TRUNCATE)
        d[inside] = self.distance[rows[inside], cols[inside]]
        return d


def shore_edge(land):
    """Pixels on the boundary of a land mask."""
    return cv2.morphologyEx(land.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)) > 0


def fit_shoreline(age, land, plate, meta, around=(0.0, 0.0, 0.0), coarse=False):
    """Rotation (deg) and shift (deg) that lay the atlas shore on our coastline.

    ``land`` is the atlas land mask in figure pixels and ``plate`` its
    PlateCarree. The cost is the symmetric truncated chamfer distance: atlas
    shore that lands in the raster window to our coastline, plus our coastline
    in the window to the atlas shore. One direction alone lets the map slide
    onto any dense stretch of shore. Powell searches +-FIT_MAX_ROTATION and
    +-FIT_MAX_SHIFT around ``around``. With ``coarse`` (lon range, lat range)
    the shift is first scanned in 2 degree steps without rotation, for a figure
    whose corner is only approximately known; the best COARSE_CANDIDATES
    separated minima are each refined and the lowest cost wins.
    Returns (params, distance_before, distance_after), distances in degrees.
    """
    b = meta['bounds']
    west, north = b.left - FIT_MARGIN, b.top + FIT_MARGIN
    width = int(round((b.right - b.left + 2 * FIT_MARGIN) / FIT_RES))
    height = int(round((b.top - b.bottom + 2 * FIT_MARGIN) / FIT_RES))
    lon_axis = west + (np.arange(width) + 0.5) * FIT_RES
    lat_axis = north - (np.arange(height) + 0.5) * FIT_RES

    # Our coastline on the grid, and its cells inside the raster window.
    canvas = np.full((height, width, 3), 255, np.uint8)
    cf.draw_coastline(canvas, {'transform': from_origin(west, north, FIT_RES, FIT_RES)}, age)
    ours = canvas[..., 0] < 128
    ours_grid = _ShoreGrid(ours, west, north)
    rows, cols = np.nonzero(ours)
    our_lons, our_lats = lon_axis[cols], lat_axis[rows]
    inside = ((our_lons >= b.left) & (our_lons <= b.right)
              & (our_lats >= b.bottom) & (our_lats <= b.top))
    our_lons, our_lats = our_lons[inside], our_lats[inside]

    # Atlas shoreline on a lon/lat grid: the raster window plus its margin for a
    # georeferenced figure, the whole figure when the offset is scanned.
    edge = shore_edge(land)
    if coarse:
        fh, fw = land.shape
        f_west, f_north = plate.to_lonlat(0, 0)
        f_east, f_south = plate.to_lonlat(fw, fh)
        g_width = int(np.ceil((f_east - f_west) / FIT_RES))
        g_height = int(np.ceil((f_north - f_south) / FIT_RES))
    else:
        f_west, f_north, g_width, g_height = west, north, width, height
    g_lons, g_lats = np.meshgrid(f_west + (np.arange(g_width) + 0.5) * FIT_RES,
                                 f_north - (np.arange(g_height) + 0.5) * FIT_RES)
    px, py = plate.to_pixel(g_lons, g_lats)
    theirs = cv2.remap(edge.astype(np.uint8), px.astype(np.float32), py.astype(np.float32),
                       interpolation=cv2.INTER_NEAREST, borderValue=0) > 0
    theirs_grid = _ShoreGrid(theirs, f_west, f_north)
    rows, cols = np.nonzero(theirs)
    lons, lats = g_lons[rows, cols], g_lats[rows, cols]
    if our_lons.size < 100:
        raise RuntimeError(f'{age} Ma: too little of our coastline in the raster window')
    center = window_center(meta)

    def near_window(x, y):
        return ((x > b.left - FIT_WINDOW_PAD) & (x < b.right + FIT_WINDOW_PAD)
                & (y > b.bottom - FIT_WINDOW_PAD) & (y < b.top + FIT_WINDOW_PAD))

    if not coarse:
        # A georeferenced figure: the atlas shore in the window is fixed before the fit.
        fixed = near_window(lons, lats)
        lons, lats = lons[fixed], lats[fixed]
        if lons.size < 100:
            raise RuntimeError(f'{age} Ma: too little atlas shoreline in the raster window')

    def cost(params):
        x, y = _rotate_shift(params, center, lons, lats)
        near = near_window(x, y) if coarse else np.ones(x.shape, dtype=bool)
        forward = ours_grid.lookup(x[near], y[near]).mean() if near.sum() >= 100 else FIT_TRUNCATE
        x, y = _unrotate_shift(params, center, our_lons, our_lats)
        backward = theirs_grid.lookup(x, y).mean()
        return float((forward + backward) / 2.0)

    centers = [around]
    starts = FIT_STARTS
    if coarse:
        (lon_lo, lon_hi), (lat_lo, lat_hi) = coarse
        scan = sorted((cost((0.0, dx, dy)), float(dx), float(dy))
                      for dx in np.arange(lon_lo, lon_hi + 0.1, 2.0)
                      for dy in np.arange(lat_lo, lat_hi + 0.1, 2.0))
        centers = []
        for _c, dx, dy in scan:
            if all(abs(dx - c[1]) > FIT_MAX_SHIFT or abs(dy - c[2]) > FIT_MAX_SHIFT for c in centers):
                centers.append((0.0, dx, dy))
            if len(centers) == COARSE_CANDIDATES:
                break
        starts = [(t, 0.0, 0.0) for t in (-10.0, 0.0, 10.0)]
    best, best_limits = None, None
    for center_params in centers:
        limits = [(center_params[0] - FIT_MAX_ROTATION, center_params[0] + FIT_MAX_ROTATION),
                  (center_params[1] - FIT_MAX_SHIFT, center_params[1] + FIT_MAX_SHIFT),
                  (center_params[2] - FIT_MAX_SHIFT, center_params[2] + FIT_MAX_SHIFT)]
        for start in starts:
            start = tuple(a + s for a, s in zip(center_params, start))
            result = minimize(cost, start, method='Powell', bounds=limits,
                              options={'xtol': 0.01, 'ftol': 1e-4})
            if best is None or result.fun < best.fun:
                best, best_limits = result, limits
    params = tuple(float(p) for p in best.x)
    if any(min(value - low, high - value) < 0.05 for value, (low, high) in zip(params, best_limits)):
        raise RuntimeError(f'{age} Ma: shoreline fit stopped at the search limit {params}')
    return params, cost((0.0, 0.0, 0.0)), float(best.fun)


def warp_to_raster(rgb, plate, meta, params=None):
    """Sample a figure on the class raster's grid (nearest neighbour).

    ``plate`` is a PlateCarree, or a Burgener frame tuple.
    """
    if not isinstance(plate, PlateCarree):
        plate = frame_plate(plate)
    transform = meta['transform']
    cols, rows = np.meshgrid(np.arange(meta['width']) + 0.5, np.arange(meta['height']) + 0.5)
    lons = transform.c + cols * transform.a
    lats = transform.f + rows * transform.e
    if params is not None:
        lons, lats = _unrotate_shift(params, window_center(meta), lons, lats)
    px, py = plate.to_pixel(lons, lats)
    border = (255, 255, 255) if rgb.ndim == 3 else 0
    warped = cv2.remap(rgb, px.astype(np.float32), py.astype(np.float32),
                       interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return warped


def score(ref_classes, rgb, frame, meta, params=None, overrides=None):
    """Classify the figure, sample it on the raster grid and score it."""
    source = classify_cells(rgb, overrides).astype(np.uint8)
    atlas_classes = warp_to_raster(source, frame, meta, params).astype(np.int32)
    mask = atlas_classes > 0
    cm = cf.confusion_matrix(ref_classes, atlas_classes, mask)
    return atlas_classes, mask, cm, cf.compute_metrics(cm)


def _method_note(age, params, before, after):
    return (
        f"<b>Fonte:</b> Burgener et al. (2023), em Scotese et al. (2024), Fig. {FIGURES[age]} "
        "(CC BY 4.0). <b>Metodo:</b> o mapa e equiretangular global; a moldura georreferencia a "
        "imagem sem pontos de controle. A paleogeografia do Burgener (PALEOMAP) nao coincide com a "
        "nossa (GPlates com correcao por idade), entao uma rotacao e um deslocamento, ajustados "
        "pela distancia entre a costa do Burgener e a nossa linha de costa, alinham os continentes: "
        f"rotacao {params[0]:+.2f} graus, deslocamento {params[1]:+.2f} graus em longitude e "
        f"{params[2]:+.2f} graus em latitude; distancia media de costa {before:.2f} -> {after:.2f} "
        "graus. Uma unica transformacao vale para a janela inteira; a posicao relativa entre "
        "America do Sul e Africa nao e corrigida. Compara-se so onde o Burgener tem terra. "
        "Classes: BW = Seco, BS = Semi-arido, demais (A, C, D) = Umido."
    )


def _extra_rows(frame_metrics, sensitivity_metrics):
    def pct(metrics):
        return f"{metrics.get('overall_accuracy', 0) * 100:.2f}%"
    return (
        '<h2>Variantes</h2>\n  <table>\n'
        '    <tr><th>Variante</th><th>Acuracia</th><th>Kappa</th></tr>\n'
        f"    <tr><td>So moldura (sem ajuste de costa)</td><td>{pct(frame_metrics)}</td>"
        f"<td>{frame_metrics.get('kappa', 0):.3f}</td></tr>\n"
        f"    <tr><td>As/Aw como Semi-arido (sensibilidade do de-para)</td>"
        f"<td>{pct(sensitivity_metrics)}</td>"
        f"<td>{sensitivity_metrics.get('kappa', 0):.3f}</td></tr>\n"
        '  </table>'
    )


def score_age(age):
    """Score one age; returns the summary row, or None when an input is missing."""
    if not os.path.isfile(image_path(age)):
        print(f'[burgener] {age} Ma: missing {cf._rel(image_path(age))}')
        return None
    if not os.path.isfile(cf.geotiff_path(age)):
        print(f'[burgener] {age} Ma: missing {cf._rel(cf.geotiff_path(age))}')
        return None
    rgb = load_image(age)
    frame = find_frame(rgb)
    ref_classes, meta = cf.geotiff_classes(age)
    params, before, after = fit_shoreline(age, land_mask(rgb, frame), frame_plate(frame), meta)

    atlas_classes, mask, cm, metrics = score(ref_classes, rgb, frame, meta, params)
    _c, _m, _cm, frame_metrics = score(ref_classes, rgb, frame, meta)
    _c, _m, _cm, sensitivity_metrics = score(ref_classes, rgb, frame, meta, params,
                                             SENSITIVITY_OVERRIDES)

    ref_pct = cf.class_area_pct(ref_classes, mask)
    atlas_pct = cf.class_area_pct(atlas_classes, mask)
    shown = warp_to_raster(rgb, frame, meta, params)
    cf.draw_coastline(shown, meta, age)
    cv2.imwrite(cf.warped_path(age, ATLAS), cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
    cf.save_metrics_csv(age, cm, metrics, ref_pct, atlas_pct, ATLAS)
    cf.save_class_composite(age, ref_classes, atlas_classes, mask, cm, ATLAS)
    cf.save_report_html(age, cm, metrics, ref_pct, atlas_pct, 'frame', 0, ATLAS,
                        method_note=_method_note(age, params, before, after),
                        extra_rows=_extra_rows(frame_metrics, sensitivity_metrics))
    print(f"[burgener] {age} Ma -> acuracia {metrics.get('overall_accuracy', 0) * 100:.2f}% "
          f"kappa {metrics.get('kappa', 0):.3f} (so moldura {frame_metrics.get('kappa', 0):.3f}; "
          f"As/Aw semi {sensitivity_metrics.get('kappa', 0):.3f}) "
          f"ajuste rot {params[0]:+.2f} dlon {params[1]:+.2f} dlat {params[2]:+.2f}")
    return {
        'age_ma': age,
        'figure': FIGURES[age],
        'accuracy_pct': round(metrics.get('overall_accuracy', 0) * 100, 2),
        'kappa': round(metrics.get('kappa', 0), 3),
        'accuracy_frame_only_pct': round(frame_metrics.get('overall_accuracy', 0) * 100, 2),
        'kappa_frame_only': round(frame_metrics.get('kappa', 0), 3),
        'kappa_as_aw_semiarid': round(sensitivity_metrics.get('kappa', 0), 3),
        'rotation_deg': round(params[0], 2),
        'shift_lon_deg': round(params[1], 2),
        'shift_lat_deg': round(params[2], 2),
        'shore_distance_before_deg': round(before, 3),
        'shore_distance_after_deg': round(after, 3),
        'pixels': metrics.get('total_pixels', 0),
        **{f'area_pcvs_{CLASS_NAMES[c]}_pct': round(ref_pct[c], 2) for c in CLASS_ORDER},
        **{f'area_burgener_{CLASS_NAMES[c]}_pct': round(atlas_pct[c], 2) for c in CLASS_ORDER},
    }


def write_summary(rows):
    with open(summary_path(), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'[burgener] summary -> {summary_path()}')


def cmd_compare(args):
    ages = [args.age] if args.age else images_present()
    if not ages:
        print('[burgener] No Burgener images in COMPARISON/inputs ({age}_burgener.jpg).')
        return
    rows, failures = [], 0
    for age in ages:
        try:
            row = score_age(age)
        except Exception as exc:
            failures += 1
            print(f'[burgener] {age} Ma failed: {exc}')
            continue
        if row:
            rows.append(row)
    if rows and not args.age:
        write_summary(rows)
    if rows:
        cf.generate_comparison_index()
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Compara os mapas paleo-Koppen de Burgener et al. (2023) com os GeoTIFFs.')
    parser.add_argument('--suffix', default=cf.GEOTIFF_SUFFIX,
                        help='Sufixo dos GeoTIFFs a comparar (default: %(default)s)')
    sub = parser.add_subparsers(dest='command', required=True)
    p_cmp = sub.add_parser('compare', help='Georreferencia, ajuste de costa, metricas e relatorio')
    p_cmp.add_argument('--age', choices=AGES, help='Processar apenas esta idade')
    p_cmp.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    cf.GEOTIFF_SUFFIX = args.suffix
    args.func(args)


if __name__ == '__main__':
    main()
