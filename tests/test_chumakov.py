"""Hay & Floegel (2012) / Chumakov et al. (1995) ages scored by shoreline fit.

Synthetic images, plus checks on the committed crops and georeference table.
"""
import json

import numpy as np
import pytest

import compare_chumakov as chumakov
import compare_floegel as cf
from compare_floegel import DRY, SEMI, HUMID

OCEAN = (187, 219, 244)


def _figure_with_graticule(spacing=120, frame=(10, 1450, 10, 730)):
    x0, x1, y0, y1 = frame
    rgb = np.full((y1 + 12, x1 + 12, 3), 255, np.uint8)
    rgb[y0:y1 + 1, x0:x1 + 1] = OCEAN
    for x in range(x0 + spacing, x1, spacing):
        rgb[y0:y1, x] = (150, 150, 150)
    for y in range(y0 + spacing, y1, spacing):
        rgb[y, x0:x1] = (150, 150, 150)
    rgb[y0:y0 + 2, x0:x1] = 0
    rgb[y1 - 1:y1 + 1, x0:x1] = 0
    rgb[y0:y1, x0:x0 + 2] = 0
    rgb[y0:y1, x1 - 1:x1 + 1] = 0
    return rgb, frame


def test_graticule_spacing_ignores_the_frame():
    rgb, frame = _figure_with_graticule(spacing=119)
    assert chumakov.graticule_spacing(rgb, frame) == pytest.approx(119, abs=0.6)


def test_graticule_spacing_with_missing_lines():
    rgb, frame = _figure_with_graticule(spacing=130)
    rgb[:, 400:700] = OCEAN                  # lines hidden under a filled continent
    assert chumakov.graticule_spacing(rgb, frame) == pytest.approx(130, abs=0.6)


def test_land_mask_closes_symbols_on_land():
    prototypes = cf.extract_legend_prototypes(str(cf.INPUT_DIR) + '/legenda.png')
    rgb = np.full((60, 60, 3), OCEAN, np.uint8)
    yellow = next(rgb_ for rgb_, code in prototypes if code == DRY)
    rgb[10:50, 10:50] = yellow
    rgb[28:32, 20:40] = 0                    # a black symbol on land
    land = chumakov.land_mask(rgb, prototypes)
    assert land[30, 30] and not land[2, 2]


def test_legend_classes_follow_the_floegel_de_para():
    prototypes = cf.extract_legend_prototypes(str(cf.INPUT_DIR) + '/legenda.png')
    assert {code for _rgb, code in prototypes} == {DRY, SEMI, HUMID}


def test_committed_crops_have_a_georeference():
    table = json.loads(open(chumakov.GEOREF_PATH, encoding='utf-8').read())
    for age in chumakov.AGES:
        assert age in table, age
        entry = table[age]
        assert set(entry['plate']) == {'x0', 'y0', 'dx', 'dy', 'west', 'north'}
        assert entry['plate']['dx'] == pytest.approx(30.0 / entry['graticule_px'], rel=1e-3)
        assert len(entry['fit']) == 3
        assert entry['shore_distance_after_deg'] < entry['shore_distance_before_deg']
        assert chumakov.image_path(age).endswith(f'{age}_chumakov.png')
        rgb, plate, _entry = chumakov.load_crop(age)
        assert rgb.shape[0] < 700 and rgb.shape[1] < 700   # a crop, not the whole figure


def test_stage_ages_are_the_nearest_pcvs_age_to_the_stage_middle():
    # Berriasian 145.0-139.8, Cenomanian 100.5-93.9, Santonian 86.3-83.6, Maastrichtian 72.1-66.0
    middles = {'140': 142.4, '95': 97.2, '85': 85.0, '70': 69.1}
    for age, middle in middles.items():
        assert int(age) == 5 * round(middle / 5)
        assert chumakov.FIGURES[age][1] in ('Berriasian', 'Cenomanian', 'Santonian', 'Maastrichtian')
