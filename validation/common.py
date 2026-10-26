"""Shared pieces of the validation scripts.

The scripts read the private point and conceptual files from local folders,
rebuild the interpolator input with the functions of ``render_map.py``
(condensation, anchor weights, rotation, class IDW) and score held-out
citations. Nothing here writes data; the scripts write only aggregate metrics,
and only to a folder the git repository ignores.

Unit of evaluation: the condensed group of citations (0.5 degree radius) with
a single winning class. Tied groups stay in training as anchors but are not
targets. The prediction at a held-out group is the winning class of the class
IDW (power 4, every remaining anchor), without the edge smoothing, which acts
on the grid.
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import render_map as rm  # noqa: E402

DEFAULT_GEOJSON_DIR = REPO / 'GEOJSON'
DEFAULT_CONCEPTUAL_DIR = Path(rm.CONCEPTUAL_DIR)
DEFAULT_OUT_DIR = REPO / 'VALIDATION_OUT'

POWER = 4.0
K = rm.KNN_NEIGHBORS
EPS = rm.CONDENSATION_EPS_DEGREES
CODES = np.array([1.0, 2.0, 3.0])  # D, S, H
NAMES = {1.0: 'D', 2.0: 'S', 3.0: 'H'}
PRIORITY = [rm.CLIMATE_CODE[c] for c in rm.CLASS_TIE_PRIORITY]
AGES = [65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
BLOCK_DEG = 10.0
SCHEMES = ('loo', 'bacia', 'bloco10')


@dataclass
class Age:
    base: str
    data_xy: np.ndarray            # condensed anchors, rotated
    data_val: np.ndarray           # class codes 1-3
    data_w: np.ndarray             # mean Weight of the winning citations
    data_group: np.ndarray         # condensed-group id of each anchor
    targets: list = field(default_factory=list)  # [group, class, basin, block]
    conc_xy: np.ndarray = None     # conceptual anchors, rotated
    conc_val: np.ndarray = None


def _norm_basin(value) -> str:
    text = str(value or '').strip().casefold()
    return '' if text in {'', 'n/a', 'na', 'none'} else text


def rotate_xy(xy: np.ndarray, base: str) -> np.ndarray:
    """The pipeline's reference-frame rotation for ``base`` applied to (lon, lat) rows."""
    corrections = rm.PALEO_REFERENCE_FRAME_CORRECTIONS
    pole_lat, pole_lon, angle = corrections.get(base, corrections['default'])
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) == 0 or angle == 0:
        return xy
    return np.array([rm.apply_euler_rotation(x, y, pole_lat, pole_lon, angle) for x, y in xy])


def build_age(base: str, points: dict, conceptual: list) -> Age:
    """Interpolator input of one age from a point collection and conceptual anchors.

    Citations are grouped with ``rm._cluster_index_groups`` and each group is
    condensed with ``rm.condense_proxy_points`` (winning class, mean Weight),
    so the anchors are those the maps use. The rotation comes after the
    condensation, as in the pipeline.
    """
    feats = []
    for f in (points or {}).get('features', []):
        g = f.get('geometry') or {}
        p = f.get('properties') or {}
        if g.get('type') != 'Point' or rm.is_conceptual_point(p):
            continue
        if rm.get_climate_class(p, None) in rm.CLIMATE_CODE:
            feats.append(f)
    coords = [(float(f['geometry']['coordinates'][0]), float(f['geometry']['coordinates'][1]))
              for f in feats]
    xy, val, wt, grp, targets = [], [], [], [], []
    for gi, group in enumerate(rm._cluster_index_groups(coords, EPS)):
        members = [feats[i] for i in group]
        _, condensed, _ = rm.condense_proxy_points(
            {'type': 'FeatureCollection', 'features': members}, eps_degrees=EPS)
        for a in condensed['features']:
            xy.append(a['geometry']['coordinates'][:2])
            val.append(rm.CLIMATE_CODE[a['properties']['Climate_Cl']])
            wt.append(a['properties'][rm.ANCHOR_WEIGHT_KEY])
            grp.append(gi)
        if len(condensed['features']) == 1:
            basins = Counter(_norm_basin(m['properties'].get('Basin_Sub_')) for m in members)
            basins.pop('', None)
            basin = basins.most_common(1)[0][0] if basins else f'sem-bacia-{gi}'
            targets.append([gi, val[-1], basin, None])
    a = Age(base, rotate_xy(np.array(xy), base), np.array(val), np.array(wt), np.array(grp))
    for t in targets:
        i = np.flatnonzero(a.data_group == t[0])[0]
        x, y = a.data_xy[i]
        t[3] = f'{math.floor(x / BLOCK_DEG)}:{math.floor(y / BLOCK_DEG)}'
    a.targets = targets
    anchors = [f for f in conceptual
               if rm.conceptual_climate_class(f['properties']) in rm.CLIMATE_CODE]
    cxy = np.array([f['geometry']['coordinates'][:2] for f in anchors], dtype=float)
    a.conc_xy = rotate_xy(cxy, base) if len(cxy) else np.zeros((0, 2))
    a.conc_val = np.array([rm.CLIMATE_CODE[rm.conceptual_climate_class(f['properties'])]
                           for f in anchors]) if anchors else np.zeros(0)
    return a


def load_age(age: int, geojson_dir=DEFAULT_GEOJSON_DIR, conceptual_dir=DEFAULT_CONCEPTUAL_DIR) -> Age:
    base = f'{age}_ma'
    pts = rm.load_geojson(str(Path(geojson_dir) / f'{base}.geojson'))
    zones_path = Path(geojson_dir) / f'{base}{rm.GEOJSON_PALEOZONES_SUFFIX}'
    zones = rm.load_geojson(str(zones_path)) if zones_path.exists() else None
    with contextlib.redirect_stdout(io.StringIO()):
        anchors, _ = rm.conceptual_anchors_for_dataset(base, zones, directory=str(conceptual_dir))
    return build_age(base, pts, anchors)


def load_ages(geojson_dir=DEFAULT_GEOJSON_DIR, conceptual_dir=DEFAULT_CONCEPTUAL_DIR, ages=AGES):
    return {g: load_age(g, geojson_dir, conceptual_dir) for g in ages}


# ---------------------------------------------------------------- KNN variants
def knn_ordinal(xy: np.ndarray, val: np.ndarray) -> np.ndarray:
    """KNN reclassification (the discarded ``--knn`` option): weighted mean of the codes, cut in thirds."""
    return rm.classify_climate_values(rm.knn_smooth_values(xy, val, k=K, power=POWER, exclude_self=True))


def knn_vote(xy: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Categorical alternative: class-wise weighted vote among the k neighbours."""
    if len(xy) <= 1:
        return val.copy()
    k = min(K, len(xy) - 1)
    d = cdist(xy, xy)
    np.fill_diagonal(d, np.inf)
    out = np.empty_like(val)
    for i in range(len(xy)):
        idx = np.argsort(d[i], kind='stable')[:k]
        w = 1.0 / np.maximum(d[i][idx], 1e-10) ** POWER
        score = {c: w[val[idx] == c].sum() for c in CODES}
        best = max(score.values())
        tied = [c for c in CODES if score[c] == best]
        out[i] = tied[0] if len(tied) == 1 else next(c for c in PRIORITY if c in tied)
    return out


# ---------------------------------------------------------------- prediction
def anchors_for(a: Age, keep: np.ndarray, mesh_weight, citation_weights: bool = True,
                only_mesh: bool = False):
    """(xy, values, weights) of the training anchors.

    ``keep`` masks the data anchors that stay. ``mesh_weight`` is the weight of
    every conceptual anchor, or None for no mesh. Without ``citation_weights``
    every data anchor counts 1.
    """
    if only_mesh:
        return a.conc_xy, a.conc_val, np.full(len(a.conc_val), float(mesh_weight))
    xy, val = a.data_xy[keep], a.data_val[keep]
    w = a.data_w[keep] if citation_weights else np.ones(len(val))
    if mesh_weight is not None and len(a.conc_xy):
        xy = np.vstack([xy, a.conc_xy])
        val = np.concatenate([val, a.conc_val])
        w = np.concatenate([w, np.full(len(a.conc_val), float(mesh_weight))])
    return xy, val, w


def idw_class(xy, val, w, q):
    """Winning class and its weight share at the query points ``q`` (``rm`` class IDW)."""
    d = cdist(np.atleast_2d(q), xy)
    win = rm._indicator_winners(d, val, POWER, w)
    shares = rm._indicator_shares(d, val, POWER, w)
    return win, shares.max(axis=1)


# name: mesh weight (None = no mesh), citation Weight on, KNN reclassification, only the mesh
CONFIGS = {
    # Method B: Weight on the citations, mesh at CONCEPTUAL_ANCHOR_WEIGHT, no KNN.
    'M_metodo_B': dict(mesh=rm.CONCEPTUAL_ANCHOR_WEIGHT, cit_w=True, knn=None),
    'N_sem_malha': dict(mesh=None, cit_w=True, knn=None),
    # Earlier variants: no Weight, mesh at 1.
    'A_knn_ordinal': dict(mesh=1.0, cit_w=False, knn='ordinal'),
    'B_sem_malha': dict(mesh=None, cit_w=False, knn='ordinal'),
    'C_sem_knn': dict(mesh=1.0, cit_w=False, knn=None),
    'D_knn_voto': dict(mesh=1.0, cit_w=False, knn='voto'),
    'E_sem_malha_sem_knn': dict(mesh=None, cit_w=False, knn=None),
    'F_so_malha': dict(mesh=1.0, cit_w=False, knn=None, only_mesh=True),
}
BASELINES = ('G_maioria', 'H_vizinho_1')


def predict(a: Age, keep: np.ndarray, q: np.ndarray, cfg: dict):
    xy, val, w = anchors_for(a, keep, cfg['mesh'], cfg['cit_w'], cfg.get('only_mesh', False))
    if cfg['knn'] == 'ordinal':
        val = knn_ordinal(xy, val)
    elif cfg['knn'] == 'voto':
        val = knn_vote(xy, val)
    return idw_class(xy, val, w, q)


def baselines(a: Age, keep: np.ndarray, q: np.ndarray):
    xy, val = a.data_xy[keep], a.data_val[keep]
    maj = Counter(val.tolist()).most_common(1)[0][0]
    nn = val[np.argmin(cdist(np.atleast_2d(q), xy), axis=1)]
    return np.full(len(q), maj), nn


def folds(a: Age, scheme: str):
    """Targets withdrawn together: one citation group, one basin, or one 10 degree block."""
    if scheme == 'loo':
        return [[t] for t in a.targets]
    col = {'bacia': 2, 'bloco10': 3}[scheme]
    by = {}
    for t in a.targets:
        by.setdefault(t[col], []).append(t)
    return list(by.values())


def fold_arrays(a: Age, fold):
    """(keep mask, query xy, true classes) of a fold: the withdrawn groups leave every anchor."""
    drop = np.isin(a.data_group, [t[0] for t in fold])
    q = np.array([a.data_xy[np.flatnonzero(a.data_group == t[0])[0]] for t in fold])
    return ~drop, q, np.array([t[1] for t in fold])


# ---------------------------------------------------------------- metrics
def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    if n == 0:
        return dict(n=0)
    acc = float(np.mean(y == p))
    pe = sum(np.mean(y == c) * np.mean(p == c) for c in CODES)
    kappa = float((acc - pe) / (1 - pe)) if pe < 1 else float('nan')
    f1s, rec = [], {}
    for c in CODES:
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else float('nan')
        rec[NAMES[c]] = r
        if tp + fn:
            f1s.append(2 * prec * r / (prec + r) if prec + r else 0.0)
    return dict(n=n, acc=acc, kappa=kappa, f1=float(np.mean(f1s)),
                rec_D=rec['D'], rec_S=rec['S'], rec_H=rec['H'])


def boot_ci(y, p, stat, reps=1000, seed=7):
    """95% percentile bootstrap interval of a metric of ``metrics``."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    vals = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        vals.append(metrics(y[i], p[i])[stat])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def add_data_args(parser):
    parser.add_argument('--geojson-dir', default=str(DEFAULT_GEOJSON_DIR),
                        help='folder with {age}_ma.geojson (default: GEOJSON/ of the repository)')
    parser.add_argument('--conceptual-dir', default=str(DEFAULT_CONCEPTUAL_DIR),
                        help='folder with {age}_ma_conceptual.geojson (default: conceptual/)')
    parser.add_argument('--out', default=str(DEFAULT_OUT_DIR),
                        help='results folder (default: VALIDATION_OUT/, ignored by git)')
    parser.add_argument('--ages', type=int, nargs='+', default=AGES,
                        help='ages in Ma (default: 65 to 145)')
    return parser
