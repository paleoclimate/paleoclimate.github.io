"""Sensitivity of the held-out kappa to the weight of the conceptual mesh.

One factor at a time: the mesh weight takes 0, 0.1, 0.25, 0.5 and 0.75 while
everything else is method B (citations with their Weight, mesh of the same
age, no KNN, power 4). The weight lives in [0, 1): a conceptual point is an
interpretation of already interpreted data and cannot count as much as the
weakest citation (minimum Weight 1). Schemes: leave-one-citation-out and
leave-one-basin-out. Only aggregate metrics are written.

    python -m validation.mesh_weight_sensitivity [--geojson-dir D] [--conceptual-dir D] [--out D]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation import common as c  # noqa: E402

MESH_WEIGHTS = [0.0, 0.1, 0.25, 0.5, 0.75]
SCHEMES = ('loo', 'bacia')


def collect(ages, mesh_weights=MESH_WEIGHTS, schemes=SCHEMES):
    """(true, predicted) lists per scheme and mesh weight, over every age."""
    recs = {s: {w: ([], []) for w in mesh_weights} for s in schemes}
    for a in ages.values():
        for scheme in schemes:
            for fold in c.folds(a, scheme):
                keep, q, truth = c.fold_arrays(a, fold)
                for w in mesh_weights:
                    cfg = dict(mesh=None if w == 0 else w, cit_w=True, knn=None)
                    pred, _ = c.predict(a, keep, q, cfg)
                    recs[scheme][w][0].extend(truth)
                    recs[scheme][w][1].extend(pred)
    return recs


def recall_ci(y, p, cls, reps=1000, seed=7):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    vals = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        m = y[i] == cls
        vals.append(np.mean(p[i][m] == cls) if m.any() else np.nan)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def summarize(recs, mesh_weights=MESH_WEIGHTS, schemes=SCHEMES):
    rows = []
    for w in mesh_weights:
        row = dict(peso_malha=w)
        for scheme in schemes:
            y, p = recs[scheme][w]
            m = c.metrics(y, p)
            row.update({f'kappa_{scheme}': m['kappa'], f'kappa_{scheme}_ic95': c.boot_ci(y, p, 'kappa'),
                        f'acc_{scheme}': m['acc'], f'n_{scheme}': m['n'],
                        f'acerto_seco_{scheme}': m['rec_D'], f'acerto_semiarido_{scheme}': m['rec_S'],
                        f'acerto_umido_{scheme}': m['rec_H'],
                        f'acerto_seco_{scheme}_ic95': recall_ci(y, p, 1.0)})
        rows.append(row)
    return rows


def main(argv=None):
    parser = c.add_data_args(argparse.ArgumentParser(description=__doc__.split('\n')[0]))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ages = c.load_ages(args.geojson_dir, args.conceptual_dir, args.ages)
    rows = summarize(collect(ages))
    for r in rows:
        print(f"w={r['peso_malha']:<5} kLOO={r['kappa_loo']:.3f} "
              f"{tuple(round(x, 3) for x in r['kappa_loo_ic95'])} "
              f"kBac={r['kappa_bacia']:.3f} {tuple(round(x, 3) for x in r['kappa_bacia_ic95'])} "
              f"Seco={r['acerto_seco_loo']:.3f}/{r['acerto_seco_bacia']:.3f}", flush=True)
    with open(out / 'sensibilidade_peso_malha.json', 'w', encoding='utf-8') as fh:
        json.dump(dict(dominio='[0,1)', referencia=0.5, linhas=rows), fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
