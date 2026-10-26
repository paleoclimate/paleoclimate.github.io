"""Leave-out validation of the class IDW against the withdrawn citations.

Schemes: ``loo`` (one condensed citation group at a time), ``bacia`` (a whole
basin at a time) and ``bloco10`` (a 10 degree block at a time). Each target is
predicted from the remaining anchors, with and without the conceptual mesh and
with and without the KNN reclassification (see ``common.CONFIGS``). Cohen's
kappa carries a 95% bootstrap interval. Only aggregate metrics are written.

    python -m validation.leave_out [--geojson-dir D] [--conceptual-dir D] [--out D]
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


def run_scheme(a: c.Age, scheme: str, configs=None):
    """Records (age, scheme, config, true class, predicted class, confidence) for one age."""
    configs = c.CONFIGS if configs is None else configs
    recs = []
    for fold in c.folds(a, scheme):
        keep, q, truth = c.fold_arrays(a, fold)
        if keep.sum() < 2:
            continue
        for name, cfg in configs.items():
            if cfg.get('only_mesh') and len(a.conc_xy) == 0:
                continue
            win, conf = c.predict(a, keep, q, cfg)
            recs.extend((a.base, scheme, name, tr, pr, float(s)) for tr, pr, s in zip(truth, win, conf))
        maj, nn = c.baselines(a, keep, q)
        for tr, pm, pn in zip(truth, maj, nn):
            recs.append((a.base, scheme, 'G_maioria', tr, pm, float('nan')))
            recs.append((a.base, scheme, 'H_vizinho_1', tr, pn, float('nan')))
    return recs


def summarize(recs, schemes=c.SCHEMES, configs=None):
    configs = list(c.CONFIGS if configs is None else configs) + list(c.BASELINES)
    table = []
    for scheme in schemes:
        for name in configs:
            sel = [r for r in recs if r[1] == scheme and r[2] == name]
            if not sel:
                continue
            y = [r[3] for r in sel]
            p = [r[4] for r in sel]
            m = c.metrics(y, p)
            m.update(esquema=scheme, config=name,
                     acc_ic=c.boot_ci(y, p, 'acc'), kappa_ic=c.boot_ci(y, p, 'kappa'))
            table.append(m)
    return table


def calibration(recs, config='M_metodo_B'):
    """Hit rate by confidence band (largest class share), leave-one-out."""
    sel = [r for r in recs if r[1] == 'loo' and r[2] == config]
    out = []
    for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        s = [r for r in sel if lo <= r[5] < hi]
        out.append(dict(faixa=f'{lo:.1f}-{min(hi, 1):.1f}', n=len(s),
                        acc=float(np.mean([r[3] == r[4] for r in s])) if s else float('nan')))
    return out


def main(argv=None):
    parser = c.add_data_args(argparse.ArgumentParser(description=__doc__.split('\n')[0]))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ages = c.load_ages(args.geojson_dir, args.conceptual_dir, args.ages)
    summary_ages = []
    for g, a in ages.items():
        dist = {c.NAMES[k]: int(sum(t[1] == k for t in a.targets)) for k in c.CODES}
        summary_ages.append(dict(idade=g, ancoras_dado=len(a.data_val), alvos=len(a.targets),
                                 ancoras_conceituais=len(a.conc_val),
                                 bacias=len({t[2] for t in a.targets}), **dist))
        print(summary_ages[-1], flush=True)
    recs = []
    for g, a in ages.items():
        for scheme in c.SCHEMES:
            recs.extend(run_scheme(a, scheme))
        print('ok', a.base, len(recs), flush=True)
    table = summarize(recs)
    with open(out / 'metricas.json', 'w', encoding='utf-8') as fh:
        json.dump(dict(idades=summary_ages, geral=table, calibracao=calibration(recs)),
                  fh, ensure_ascii=False, indent=1)
    for m in table:
        print(f"{m['esquema']:8s} {m['config']:22s} n={m['n']:4d} acc={m['acc']:.3f} "
              f"[{m['acc_ic'][0]:.3f},{m['acc_ic'][1]:.3f}] kappa={m['kappa']:.3f} "
              f"[{m['kappa_ic'][0]:.3f},{m['kappa_ic'][1]:.3f}] f1={m['f1']:.3f} "
              f"recD={m['rec_D']:.2f} recS={m['rec_S']:.2f} recH={m['rec_H']:.2f}")
    print(calibration(recs))


if __name__ == '__main__':
    main()
