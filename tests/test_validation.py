"""Validation scripts on synthetic data (a few dozen fictitious points in three classes)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import render_map as rm
from validation import common as c
from validation import leave_out, mesh_weight_sensitivity as sens


def _point(lon, lat, climate, weight=3, basin='Bacia X', ident=None):
    ident = ident or f'P{lon:g}_{lat:g}'
    return {'type': 'Feature',
            'properties': {'ID': ident, 'Climate_Cl': climate, 'Weight': weight, 'Basin_Sub_': basin},
            'geometry': {'type': 'Point', 'coordinates': [float(lon), float(lat)]}}


def _mesh_point(lon, lat, climate):
    return {'type': 'Feature', 'properties': {'ID': 'N/A', 'Climate_Cl': climate},
            'geometry': {'type': 'Point', 'coordinates': [float(lon), float(lat)]}}


def _collection(features):
    return {'type': 'FeatureCollection', 'features': features}


def _band(lon):
    return 'D' if lon < 12 else 'S' if lon < 24 else 'H'


def synthetic_points():
    """36 citations on a 3 degree grid: dry, semi-arid and humid bands, six basins."""
    feats = []
    for col in range(12):
        for lat in (-6, 0, 6):
            lon = col * 3
            feats.append(_point(lon, lat, _band(lon), weight=1 + (col + lat) % 5,
                                basin=f'Bacia {col // 2}'))
    return _collection(feats)


def synthetic_mesh():
    return [_mesh_point(lon, lat, _band(lon)) for lon in range(0, 36, 6) for lat in (-20, -10, 0, 10, 20)]


@pytest.fixture(scope='module')
def age():
    return c.build_age('synthetic', synthetic_points(), synthetic_mesh())


def test_build_age_condenses_and_counts(age):
    assert len(age.targets) == 36
    assert len(age.data_val) == 36
    assert set(age.data_val) == {1.0, 2.0, 3.0}
    assert len(age.conc_val) == 30
    assert age.data_w.min() >= 1 and age.data_w.max() <= 5


def test_leave_one_out_removes_the_target_and_does_not_leak_its_class():
    # One dry citation among semi-arid and humid ones: withdrawn, no dry anchor remains.
    feats = [_point(0, 0, 'D', ident='alvo', basin='A')]
    feats += [_point(10 + 3 * i, 0, 'S', basin='B') for i in range(4)]
    feats += [_point(10 + 3 * i, 6, 'H', basin='C') for i in range(4)]
    a = c.build_age('synthetic', _collection(feats), [])
    target = next(t for t in a.targets if t[1] == 1.0)
    keep, q, truth = c.fold_arrays(a, [target])
    assert truth.tolist() == [1.0]
    assert keep.sum() == len(a.data_val) - 1
    assert 1.0 not in a.data_val[keep]
    assert not np.any(np.all(np.isclose(a.data_xy[keep], q[0]), axis=1))
    for cfg in (c.CONFIGS['N_sem_malha'], c.CONFIGS['E_sem_malha_sem_knn'], c.CONFIGS['A_knn_ordinal']):
        pred, _ = c.predict(a, keep, q, cfg)
        assert pred[0] != 1.0
    # Control: with every anchor kept the dry class sits on the query and wins.
    pred, _ = c.predict(a, np.ones(len(a.data_val), dtype=bool), q, c.CONFIGS['N_sem_malha'])
    assert pred[0] == 1.0


def test_leave_one_out_has_one_fold_per_target(age):
    folds = c.folds(age, 'loo')
    assert len(folds) == len(age.targets)
    assert all(len(f) == 1 for f in folds)


def test_basin_scheme_removes_the_whole_basin(age):
    folds = c.folds(age, 'bacia')
    assert len(folds) == 6
    for fold in folds:
        keep, q, truth = c.fold_arrays(age, fold)
        assert len(fold) == 6
        assert keep.sum() == len(age.data_val) - 6
        assert len(q) == len(truth) == 6
        assert {t[2] for t in fold} == {fold[0][2]}
        dropped = {int(g) for g in age.data_group[~keep]}
        assert dropped == {t[0] for t in fold}


def test_kappa_of_a_trivial_case():
    y = [1.0, 1.0, 2.0, 2.0]
    assert c.metrics(y, y)['kappa'] == pytest.approx(1.0)
    assert c.metrics(y, [2.0] * 4)['kappa'] == pytest.approx(0.0)
    # accuracy 0.75, chance agreement 0.5 -> kappa 0.5
    assert c.metrics(y, [1.0, 2.0, 2.0, 2.0])['kappa'] == pytest.approx(0.5)


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.choice([1.0, 2.0, 3.0], 200)
    p = np.where(rng.random(200) < 0.7, y, rng.choice([1.0, 2.0, 3.0], 200))
    lo, hi = c.boot_ci(y, p, 'kappa', reps=200)
    assert lo < c.metrics(y, p)['kappa'] < hi


def test_leave_out_runs_every_config_and_scheme(age):
    recs = []
    for scheme in c.SCHEMES:
        recs.extend(leave_out.run_scheme(age, scheme))
    table = leave_out.summarize(recs)
    assert {m['config'] for m in table} == set(c.CONFIGS) | set(c.BASELINES)
    assert all(m['n'] == 36 for m in table)
    by = {(m['esquema'], m['config']): m for m in table}
    # The bands are clean, so the method B kappa beats the majority-class baseline.
    assert by[('loo', 'M_metodo_B')]['kappa'] > by[('loo', 'G_maioria')]['kappa']
    assert by[('loo', 'M_metodo_B')]['kappa_ic'][0] <= by[('loo', 'M_metodo_B')]['kappa'] <= \
        by[('loo', 'M_metodo_B')]['kappa_ic'][1]


def test_mesh_weight_sensitivity_runs(age):
    recs = sens.collect({'synthetic': age})
    rows = sens.summarize(recs)
    assert [r['peso_malha'] for r in rows] == sens.MESH_WEIGHTS
    for r in rows:
        assert r['n_loo'] == r['n_bacia'] == 36
        assert -1.0 <= r['kappa_loo'] <= 1.0
        lo, hi = r['kappa_bacia_ic95']
        assert lo <= r['kappa_bacia'] <= hi
    # Leaving a whole basin out on clean bands, the mesh helps: weight 0.5 is not worse than none.
    assert rows[3]['kappa_bacia'] >= rows[0]['kappa_bacia']


def test_scripts_read_local_folders_and_write_only_aggregates(tmp_path):
    data = tmp_path / 'dados'
    conc = tmp_path / 'malha'
    out = tmp_path / 'saida'
    data.mkdir()
    conc.mkdir()
    (data / '100_ma.geojson').write_text(json.dumps(synthetic_points()), encoding='utf-8')
    (conc / '100_ma_conceptual.geojson').write_text(
        json.dumps(_collection(synthetic_mesh())), encoding='utf-8')
    args = ['--geojson-dir', str(data), '--conceptual-dir', str(conc), '--out', str(out), '--ages', '100']
    leave_out.main(args)
    sens.main(args)
    metrics = json.loads((out / 'metricas.json').read_text(encoding='utf-8'))
    assert metrics['idades'][0]['alvos'] == 36
    assert metrics['idades'][0]['ancoras_conceituais'] == 30
    sensitivity = json.loads((out / 'sensibilidade_peso_malha.json').read_text(encoding='utf-8'))
    assert len(sensitivity['linhas']) == 5
    text = (out / 'metricas.json').read_text(encoding='utf-8') + \
        (out / 'sensibilidade_peso_malha.json').read_text(encoding='utf-8')
    assert 'coordinates' not in text and 'Bacia' not in text


def test_validation_output_folder_is_ignored_by_git():
    ignore = (c.REPO / '.gitignore').read_text(encoding='utf-8').splitlines()
    assert 'VALIDATION_OUT/' in ignore
    assert c.DEFAULT_OUT_DIR.name == 'VALIDATION_OUT'


def test_condensation_matches_the_map_pipeline(age):
    """Anchors of build_age are those of condense_proxy_points over the whole collection."""
    _, condensed, _ = rm.condense_proxy_points(synthetic_points())
    assert len(condensed['features']) == len(age.data_val)
