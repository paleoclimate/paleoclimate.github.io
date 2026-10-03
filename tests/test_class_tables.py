"""The published class tables match the sources they were counted from."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

import render_map as renderer

REPO = Path(__file__).resolve().parents[1]
IDW_CSV = REPO / 'idw_classes_por_idade_raio_aglutinacao_0.5_power_4.0.csv'
POINTS_CSV = REPO / 'pontos_dado_por_idade_raio_aglutinacao_0.5_power_4.0.csv'


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _data_point_counts(age: int) -> Counter:
    raw = json.loads((REPO / 'GEOJSON' / f'{age}_ma.geojson').read_text(encoding='utf-8'))
    counts: Counter = Counter()
    for feature in raw.get('features', []):
        if (feature.get('geometry') or {}).get('type') != 'Point':
            continue
        props = feature.get('properties') or {}
        if renderer.is_conceptual_point(props):
            continue
        climate = renderer.get_climate_class(props)
        if climate in ('D', 'S', 'H'):
            counts[climate] += 1
    return counts


def _audit_counts(age: int) -> tuple[Counter, Counter]:
    condensed = json.loads(
        (REPO / 'CONDENSED' / f'{age}_ma_condensed.geojson').read_text(encoding='utf-8')
    )
    overruled = json.loads(
        (REPO / 'CONDENSED' / f'{age}_ma_overruled.geojson').read_text(encoding='utf-8')
    )
    anchors = Counter(
        feature['properties']['Climate_Cl'] for feature in condensed['features']
    )
    lost = Counter(
        (feature.get('properties') or {}).get('Climate_Cl')
        for feature in overruled['features']
    )
    return anchors, lost


def test_point_census_excludes_conceptual_points():
    if not (REPO / 'GEOJSON' / '65_ma.geojson').is_file():
        pytest.skip('GEOJSON/ is local and gitignored')
    rows = _rows(POINTS_CSV)
    assert [row['idade_ma'] for row in rows] == [str(age) for age in range(65, 150, 5)]
    for row in rows:
        counts = _data_point_counts(int(row['idade_ma']))
        assert int(row['pontos_seco']) == counts['D']
        assert int(row['pontos_semiarido']) == counts['S']
        assert int(row['pontos_umido']) == counts['H']


def test_idw_class_table_matches_condensation_audit():
    rows = _rows(IDW_CSV)
    assert [row['idade_ma'] for row in rows] == [str(age) for age in range(65, 150, 5)]
    for row in rows:
        anchors, lost = _audit_counts(int(row['idade_ma']))
        assert int(row['idw_ancora_seco']) == anchors['D']
        assert int(row['idw_ancora_semiarido']) == anchors['S']
        assert int(row['idw_ancora_umido']) == anchors['H']
        assert int(row['citacoes_seco_que_perderam']) == lost['D']
        assert int(row['citacoes_semiarido_que_perderam']) == lost['S']
        assert int(row['citacoes_umido_que_perderam']) == lost['H']
