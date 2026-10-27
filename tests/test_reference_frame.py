"""Reference-frame correction: one angle per age, 'default' as the fallback."""

from __future__ import annotations

import pytest

import render_map as rm

AGES = list(range(65, 150, 5))


def _point(lon, lat):
    return {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {},
         'geometry': {'type': 'Point', 'coordinates': [lon, lat]}},
    ]}


def test_every_age_has_its_own_angle_about_the_same_pole():
    for age in AGES:
        pole_lat, pole_lon, _ = rm.PALEO_REFERENCE_FRAME_CORRECTIONS[f'{age}_ma']
        assert (pole_lat, pole_lon) == rm.PALEO_REFERENCE_FRAME_POLE == (0.0, 60.0)


def test_fitted_angles():
    corrections = rm.PALEO_REFERENCE_FRAME_CORRECTIONS
    assert corrections['100_ma'][2] == pytest.approx(-13.1)
    assert corrections['65_ma'][2] == pytest.approx(1.6)
    assert corrections['140_ma'][2] == corrections['145_ma'][2] == pytest.approx(-21.4)


def test_dataset_outside_the_table_falls_back_to_default():
    data = _point(-40.0, -20.0)
    rotated = rm.apply_paleo_reference_frame_correction(data, 'synthetic')
    expected = rm.apply_euler_rotation(-40.0, -20.0, *rm.PALEO_REFERENCE_FRAME_CORRECTIONS['default'])
    assert rotated['features'][0]['geometry']['coordinates'][:2] == pytest.approx(list(expected))
    assert rm.PALEO_REFERENCE_FRAME_CORRECTIONS['default'] == (0.0, 60.0, -13.0)


def test_each_age_is_rotated_by_its_own_angle():
    data = _point(-40.0, -20.0)
    young = rm.apply_paleo_reference_frame_correction(data, '65_ma')
    old = rm.apply_paleo_reference_frame_correction(data, '140_ma')
    assert young['features'][0]['geometry']['coordinates'][:2] == pytest.approx(
        list(rm.apply_euler_rotation(-40.0, -20.0, 0.0, 60.0, 1.6)))
    assert old['features'][0]['geometry']['coordinates'][:2] == pytest.approx(
        list(rm.apply_euler_rotation(-40.0, -20.0, 0.0, 60.0, -21.4)))
    # a negative angle about (0° N, 60° E) moves South America north
    assert old['features'][0]['geometry']['coordinates'][1] > young['features'][0]['geometry']['coordinates'][1]
    assert data['features'][0]['geometry']['coordinates'] == [-40.0, -20.0]
