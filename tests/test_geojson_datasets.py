"""Discovery of point + coastline pairs and optional Paleozones companions."""

from __future__ import annotations

from pathlib import Path

import json

import pytest

import render_map as renderer

from tests.helpers import write_hover_stack_map


def _write_geojson(path, features=None):
    path.write_text(
        json.dumps({'type': 'FeatureCollection', 'features': features or []}),
        encoding='utf-8',
    )


def test_discover_requires_coastline_and_skips_companions(tmp_path):
    _write_geojson(tmp_path / '110_ma.geojson', [{'type': 'Feature'}])
    _write_geojson(tmp_path / '110_ma_coastline.geojson')
    _write_geojson(tmp_path / '110_ma_paleozones.geojson', [{'type': 'Feature'}])
    _write_geojson(tmp_path / '110_ma_basins.geojson', [{'type': 'Feature'}])
    _write_geojson(tmp_path / '115_ma.geojson', [{'type': 'Feature'}])
    _write_geojson(tmp_path / '115_ma_coastline.geojson')
    _write_geojson(tmp_path / 'orphan_ma.geojson')  # no coastline

    datasets = renderer.discover_geojson_datasets(str(tmp_path))
    by_base = {row[0]: row for row in datasets}

    assert set(by_base) == {'110_ma', '115_ma'}
    assert Path(by_base['110_ma'][3]) == tmp_path / '110_ma_paleozones.geojson'
    assert Path(by_base['110_ma'][4]) == tmp_path / '110_ma_basins.geojson'
    assert by_base['115_ma'][3] is None
    assert by_base['115_ma'][4] is None


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


def test_paleozone_layer_is_not_drawn(tmp_path):
    html = write_hover_stack_map(tmp_path).read_text(encoding='utf-8')
    assert 'Paleozone:' not in html
    assert '"Paleozones"' not in html


def test_real_point_count_skips_conceptual_points():
    points = {
        'type': 'FeatureCollection',
        'features': [
            {'type': 'Feature', 'properties': {'ID': '3', 'Climate_Cl': 'H'},
             'geometry': {'type': 'Point', 'coordinates': [1.0, 2.0]}},
            {'type': 'Feature', 'properties': {'ID': None, 'Climate_Cl': 'S'},
             'geometry': {'type': 'Point', 'coordinates': [3.0, 4.0]}},
            {'type': 'Feature', 'properties': {'ID': 'N/A', 'Climate_Cl': 'D'},
             'geometry': {'type': 'Point', 'coordinates': [5.0, 6.0]}},
        ],
    }
    assert renderer.real_point_count(points) == 1


def test_point_popup_shows_the_reference_and_tucks_the_notes(tmp_path):
    from PIL import Image

    points = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'properties': {
                    'ID': '1',
                    'Formation': 'Alc?tara',
                    'Basin_Sub_': 'S? Lu?',
                    'Country': 'Brazil',
                    'Climate_Cl': 'H',
                    'TIME': 100,
                    'REF(Authors, Year)': 'Gon?lves et al. 2001',
                    'Paleoenvironment': '',
                    'Dating evidence': '109 ?18 Ma',
                    'Lithology, structures, paleowind': 'sandstone',
                },
                'geometry': {'type': 'Point', 'coordinates': [0, 0]},
            },
            {
                'type': 'Feature',
                'properties': {'ID': 'N/A', 'Climate_Cl': 'D', 'Formation': 'Grid'},
                'geometry': {'type': 'Point', 'coordinates': [1, 1]},
            },
        ],
    }
    coast = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'properties': {'NAME': 'Shore', 'TIME': 100},
            'geometry': {'type': 'LineString', 'coordinates': [[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]]},
        }],
    }
    paleozones = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'properties': {'Paleozona': 'Humid'},
            'geometry': {'type': 'Polygon', 'coordinates': [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]]},
        }],
    }
    image = tmp_path / 'overlay.png'
    Image.new('RGB', (8, 8), (40, 80, 140)).save(image)
    output = tmp_path / 'map.html'
    renderer.create_map(
        points,
        coast,
        output_file=str(output),
        color_stats_img_path=str(image),
        paleozones_data=paleozones,
    )
    html_text = output.read_text(encoding='utf-8')
    assert 'Alcântara' in html_text
    assert 'São Luís' in html_text
    assert '<dt>Reference</dt>' in html_text
    assert 'Gonçalves' in html_text
    assert 'class="pcvs-more"' in html_text
    assert 'Paleoenvironment' in html_text
    assert 'class="pcvs-na"' in html_text
    assert 'max-height: 280px' in html_text
    assert 'min-width: 0' in html_text
    assert 'Dating Evidence' in html_text
    assert '>Lithology</dt>' in html_text
    assert '109 ± 18 Ma' in html_text
    assert 'Data points' in html_text
    assert '<strong>1</strong>' in html_text
    assert '"Paleozones"' not in html_text
    assert 'Grid' not in html_text


def test_loaded_point_names_used_on_click_keep_their_accents():
    geojson = Path(__file__).resolve().parents[1] / 'GEOJSON'
    if not geojson.is_dir():
        pytest.skip('GEOJSON/ is local and gitignored')
    forbidden = (
        'Alc?tara',
        'S? Mateus',
        'S? Lu?',
        'Alter do Ch?',
        'Tr? Barras',
        'Algod?s',
        'Gon?lves',
        'C?doba',
        'Embor?S? Jos?N/A',
        'Zim?',
        'Françaois',
        'Pedrãoo',
    )
    broken = []
    for path in sorted(geojson.glob('*.geojson')):
        if any(token in path.name for token in ('_coastline', '_paleozones', '_basins')):
            continue
        data = renderer.load_geojson(path)
        for feature in data['features']:
            props = feature.get('properties') or {}
            for key in (
                'Formation',
                'Basin_Sub_',
                renderer.POINT_REFERENCE_FIELD,
                'Paleoenvironment',
                'Dating evidence',
            ):
                value = props.get(key)
                if not isinstance(value, str):
                    continue
                for fragment in forbidden:
                    if fragment in value:
                        broken.append(f'{path.name} {key}: {value[:120]}')
                        break
    assert not broken, 'Point fields still show broken text:\n' + '\n'.join(broken[:40])


def test_coastline_is_the_dark_shore():
    styled = renderer.coastline_style({})
    assert styled['color'] == renderer.COASTLINE_COLOR == '#334155'
    assert styled['weight'] == renderer.COASTLINE_WEIGHT_PX == 1.05
    assert styled['opacity'] == renderer.COASTLINE_OPACITY == 0.88
    assert styled['lineCap'] == 'round'
    assert styled['lineJoin'] == 'round'
    assert renderer._hex_to_rgb(styled['color']) <= renderer._hex_to_rgb('#334155')


def test_basin_outlines_are_a_stroke_and_keep_only_the_name():
    assert renderer.LAYER_BASINS == 'Basins'
    assert renderer.BASIN_OUTLINE_SHOW is True
    assert 145 in renderer.VIEWER_HIDDEN_AGES
    styled = renderer.basin_outline_style({})
    assert styled['fill'] is True
    assert styled['fillOpacity'] == 0
    assert styled['color'] == renderer.BASIN_OUTLINE_COLOR == '#94a3b8'
    assert styled['opacity'] == renderer.BASIN_OUTLINE_OPACITY == 0.40
    assert styled['weight'] == renderer.BASIN_OUTLINE_WEIGHT_PX == 1.35
    red, green, blue = renderer._hex_to_rgb(styled['color'])
    assert max(red, green, blue) - min(red, green, blue) < 40
    assert styled['opacity'] < renderer.COASTLINE_OPACITY
    assert renderer._hex_to_rgb(styled['color']) > renderer._hex_to_rgb(renderer.COASTLINE_COLOR)
    displayed = renderer.basin_outline_for_display({
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'properties': {'BASIN_NAME': 'Jatoba', 'EXP_STATUS': 'Little Explored'},
            'geometry': {'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        }],
    })
    feature = displayed['features'][0]
    assert feature['properties'] == {'BASIN_NAME': 'Jatoba'}
    assert feature['geometry']['type'] == 'Polygon'
