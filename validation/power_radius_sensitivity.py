"""Sensitivity of the held-out kappa to the IDW power and the condensation radius.

Power 4 and the 0.5 degree radius were picked by macro IoU against the
paleozones on 100, 120 and 130 Ma, before method B (ADR 0010); that score is
circular. This script checks them with method B and the held-out citation, one
factor at a time:

- power 1, 2, 3, 4, 5, 6 and 8 at the 0.5 degree radius;
- radius 0.25, 0.5 and 1.0 degree at power 4. The radius changes the
  condensed groups, so the number of targets changes with it.

Schemes: leave-one-citation-out and leave-one-basin-out. Only aggregate
metrics are written.

    python -m validation.power_radius_sensitivity [--geojson-dir D] [--conceptual-dir D] [--out D]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation import common as c  # noqa: E402
from validation.mesh_weight_sensitivity import recall_ci  # noqa: E402

POWERS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
RADII = [0.25, 0.5, 1.0]
SCHEMES = ('loo', 'bacia')


def method_b(power: float = c.POWER) -> dict:
    return dict(mesh=c.rm.CONCEPTUAL_ANCHOR_WEIGHT, cit_w=True, knn=None, power=power)


def collect(ages, configs, schemes=SCHEMES):
    """(true, predicted) lists per scheme and configuration name, over every age."""
    recs = {s: {name: ([], []) for name in configs} for s in schemes}
    for a in ages.values():
        for scheme in schemes:
            for fold in c.folds(a, scheme):
                keep, q, truth = c.fold_arrays(a, fold)
                for name, cfg in configs.items():
                    pred, _ = c.predict(a, keep, q, cfg)
                    recs[scheme][name][0].extend(truth)
                    recs[scheme][name][1].extend(pred)
    return recs


def summarize(recs, label: dict, schemes=SCHEMES):
    rows = []
    for name, fields in label.items():
        row = dict(fields)
        for scheme in schemes:
            y, p = recs[scheme][name]
            m = c.metrics(y, p)
            row.update({f'kappa_{scheme}': m['kappa'], f'kappa_{scheme}_ic95': c.boot_ci(y, p, 'kappa'),
                        f'acc_{scheme}': m['acc'], f'n_{scheme}': m['n'],
                        f'acerto_seco_{scheme}': m['rec_D'],
                        f'acerto_seco_{scheme}_ic95': recall_ci(y, p, 1.0)})
        rows.append(row)
    return rows


def _print(rows, key):
    for r in rows:
        print(f"{key}={r[key]:<5} n={r['n_loo']:<5} kLOO={r['kappa_loo']:.3f} "
              f"{tuple(round(x, 3) for x in r['kappa_loo_ic95'])} "
              f"kBac={r['kappa_bacia']:.3f} {tuple(round(x, 3) for x in r['kappa_bacia_ic95'])} "
              f"Seco={r['acerto_seco_loo']:.3f}/{r['acerto_seco_bacia']:.3f}", flush=True)


def main(argv=None):
    parser = c.add_data_args(argparse.ArgumentParser(description=__doc__.split('\n')[0]))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ages = c.load_ages(args.geojson_dir, args.conceptual_dir, args.ages)
    configs = {f'p{p:g}': method_b(p) for p in POWERS}
    power_rows = summarize(collect(ages, configs),
                           {f'p{p:g}': dict(potencia=p, raio=c.EPS) for p in POWERS})
    _print(power_rows, 'potencia')

    radius_rows = []
    for radius in RADII:
        by_radius = c.load_ages(args.geojson_dir, args.conceptual_dir, args.ages, eps=radius)
        recs = collect(by_radius, {'r': method_b()})
        radius_rows += summarize(recs, {'r': dict(potencia=c.POWER, raio=radius)})
    _print(radius_rows, 'raio')

    with open(out / 'sensibilidade_potencia_raio.json', 'w', encoding='utf-8') as fh:
        json.dump(dict(referencia=dict(potencia=c.POWER, raio=c.EPS),
                       potencia=power_rows, raio=radius_rows), fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
