"""Discovery of point + coastline pairs and optional Paleozones companions."""

from __future__ import annotations

from pathlib import Path

import json

import render_map as renderer


def _write_geojson(path, features=None):
    path.write_text(
        json.dumps({'type': 'FeatureCollection', 'features': features or []}),
        encoding='utf-8',
    )


def test_discover_requires_coastline_and_skips_companions(tmp_path):
    _write_geojson(tmp_path / '110_ma.geojson', [{'type': 'Feature'}])
    _write_geojson(tmp_path / '110_ma_coastline.geojson')
    _write_geojson(tmp_path / '110_ma_paleozones.geojson', [{'type': 'Feature'}])
    _write_geojson(tmp_path / '115_ma.geojson', [{'type': 'Feature'}])
    _write_geojson(tmp_path / '115_ma_coastline.geojson')
    _write_geojson(tmp_path / 'orphan_ma.geojson')  # no coastline

    datasets = renderer.discover_geojson_datasets(str(tmp_path))
    by_base = {row[0]: row for row in datasets}

    assert set(by_base) == {'110_ma', '115_ma'}
    assert Path(by_base['110_ma'][3]) == tmp_path / '110_ma_paleozones.geojson'
    assert by_base['115_ma'][3] is None


def test_conceptual_points_drop_from_drawing_and_stay_in_interpolation():
    points = {
        'type': 'FeatureCollection',
        'features': [
            {'type': 'Feature', 'properties': {'ID': '3', 'Climate_Cl': 'H'},
             'geometry': {'type': 'Point', 'coordinates': [1.0, 2.0]}},
            {'type': 'Feature', 'properties': {'ID': None, 'Climate_Cl': 'S'},
             'geometry': {'type': 'Point', 'coordinates': [3.0, 4.0]}},
            {'type': 'Feature', 'properties': {'ID': 'N/A', 'Climate_Cl': 'D'},
             'geometry': {'type': 'Point', 'coordinates': [5.0, 6.0]}},
            {'type': 'Feature', 'properties': {'ID': ' n/a ', 'Climate_Cl': 'H'},
             'geometry': {'type': 'Point', 'coordinates': [7.0, 8.0]}},
        ],
    }
    drawn = renderer.drawable_point_features(points)
    assert [feature['properties']['ID'] for feature in drawn] == ['3']
    _, values = renderer.extract_points_and_values(points)
    assert list(values) == [3.0, 2.0, 1.0, 3.0]


def test_paleozone_labels_normalize_humid_typo():
    assert renderer.paleozone_climate_class({'Paleozona': 'humid'}) == 'H'
    assert renderer.paleozone_display_label({'Paleozona': 'humid'}) == 'Humid'
    assert renderer.paleozone_climate_class({'Paleozona': 'Semi-arid'}) == 'S'
    assert renderer.paleozone_climate_class({'Paleozona': 'Dry'}) == 'D'
    styled = renderer.paleozone_style({'properties': {'Paleozona': 'Humid'}})
    assert styled['fillColor'] == renderer.CLIMATE_COLORS['H']
    assert styled['fillOpacity'] == renderer.PALEOZONE_FILL_OPACITY
    assert styled['opacity'] == renderer.PALEOZONE_STROKE_OPACITY
    labeled = renderer.with_paleozone_tooltip_labels({
        'type': 'FeatureCollection',
        'features': [{'properties': {'Paleozona': 'humid'}, 'geometry': None}],
    })
    assert labeled['features'][0]['properties']['Paleozone'] == 'Humid'
    assert labeled['features'][0]['properties']['Paleozona'] == 'humid'
