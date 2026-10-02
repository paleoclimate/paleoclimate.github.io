"""Condensation of data points and the indicator-IDW class raster."""

from __future__ import annotations

import numpy as np

import render_map as renderer


def _point(lon, lat, climate, ident='1', weight=None):
    props = {'ID': ident, 'Climate_Cl': climate}
    if weight is not None:
        props['Weight'] = weight
    return {
        'type': 'Feature',
        'properties': props,
        'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
    }


def _collection(*features):
    return {'type': 'FeatureCollection', 'features': list(features)}


def test_points_within_half_degree_form_one_cluster_and_just_outside_do_not():
    near = _collection(
        _point(0.0, 0.0, 'D', 'a'),
        _point(0.49, 0.0, 'H', 'b'),
    )
    far = _collection(
        _point(0.0, 0.0, 'D', 'a'),
        _point(0.51, 0.0, 'H', 'b'),
    )
    exactly = _collection(
        _point(0.0, 0.0, 'D', 'a'),
        _point(0.5, 0.0, 'H', 'b'),
    )
    _, near_condensed, _ = renderer.condense_proxy_points(near)
    _, far_condensed, _ = renderer.condense_proxy_points(far)
    _, exact_condensed, _ = renderer.condense_proxy_points(exactly)
    assert {feature['properties']['cluster_size'] for feature in near_condensed['features']} == {2}
    assert {feature['properties']['cluster_size'] for feature in far_condensed['features']} == {1}
    assert {feature['properties']['cluster_size'] for feature in exact_condensed['features']} == {2}
    exact_coords = [feature['geometry']['coordinates'] for feature in exact_condensed['features']]
    assert exact_coords[0] == exact_coords[1]


def test_chain_keeps_linking_while_a_neighbor_is_inside_half_degree():
    points = _collection(
        _point(0.0, 0.0, 'D', 'a'),
        _point(0.4, 0.0, 'D', 'b'),
        _point(0.8, 0.0, 'H', 'c'),
    )
    _, condensed, overruled = renderer.condense_proxy_points(points)
    assert len(condensed['features']) == 1
    assert condensed['features'][0]['properties']['Climate_Cl'] == 'D'
    assert condensed['features'][0]['properties']['votes'] == 2
    assert [feature['properties']['ID'] for feature in overruled['features']] == ['c']


def test_majority_is_one_row_one_vote_and_weight_does_not_count():
    points = _collection(
        _point(0.0, 0.0, 'D', 'dry', weight=10),
        _point(0.5, 0.0, 'H', 'h1', weight=1),
        _point(1.0, 0.0, 'H', 'h2', weight=1),
    )
    _, condensed, overruled = renderer.condense_proxy_points(points)
    assert len(condensed['features']) == 1
    anchor = condensed['features'][0]
    assert anchor['properties']['Climate_Cl'] == 'H'
    assert anchor['properties']['votes'] == 2
    assert anchor['properties']['tie'] is False
    assert anchor['geometry']['coordinates'] == [0.5, 0.0]
    assert [feature['properties']['ID'] for feature in overruled['features']] == ['dry']
    assert overruled['features'][0]['properties']['winner'] == 'H'


def test_exact_tie_emits_one_anchor_per_class_at_the_mean_without_jitter():
    points = _collection(
        _point(0.0, 0.0, 'D', 'd1'),
        _point(0.0, 0.5, 'D', 'd2'),
        _point(0.5, 0.0, 'H', 'h1'),
        _point(0.5, 0.5, 'H', 'h2'),
        _point(0.25, 0.25, 'S', 's1'),
    )
    _, condensed, overruled = renderer.condense_proxy_points(points)
    classes = [feature['properties']['Climate_Cl'] for feature in condensed['features']]
    assert classes == ['H', 'D']
    coords = [feature['geometry']['coordinates'] for feature in condensed['features']]
    assert coords[0] == coords[1]
    assert coords[0] == [0.25, 0.25]
    assert all(feature['properties']['tie'] is True for feature in condensed['features'])
    assert all(feature['properties']['status'] == 'tie' for feature in condensed['features'])
    assert [feature['properties']['ID'] for feature in overruled['features']] == ['s1']


def test_conceptual_points_skip_the_cluster_and_stay_on_the_interpolator():
    points = _collection(
        _point(0.0, 0.0, 'D', 'formation'),
        _point(0.4, 0.0, 'H', None),
        _point(1.6, 0.0, 'S', 'other'),
    )
    interpolator, condensed, overruled = renderer.condense_proxy_points(points)
    assert len(condensed['features']) == 2
    assert overruled['features'] == []
    climates = [
        renderer.get_climate_class(feature['properties'])
        for feature in interpolator['features']
    ]
    assert climates == ['D', 'S', 'H']
    conceptual = interpolator['features'][-1]
    assert conceptual['geometry']['coordinates'] == [0.4, 0.0]


def test_a_conceptual_point_does_not_bridge_two_data_points():
    points = _collection(
        _point(0.0, 0.0, 'D', 'a'),
        _point(0.8, 0.0, 'H', None),
        _point(1.6, 0.0, 'H', 'b'),
    )
    _, condensed, _ = renderer.condense_proxy_points(points)
    assert len(condensed['features']) == 2
    assert {feature['properties']['Climate_Cl'] for feature in condensed['features']} == {'D', 'H'}


def test_indicator_idw_does_not_paint_semi_arid_between_dry_and_humid():
    points = np.array([[0.0, 0.0], [3.0, 0.0]])
    values = np.array([1.0, 3.0])
    grid_lons = np.array([1.5])
    grid_lats = np.array([0.0])
    painted = renderer.indicator_idw(points, values, grid_lons, grid_lats, power=1.0)
    assert painted[0, 0] == renderer.CLIMATE_CODE['D']

    closer_to_humid = renderer.indicator_idw(
        points, values, np.array([2.0]), np.array([0.0]), power=1.0
    )
    assert closer_to_humid[0, 0] == renderer.CLIMATE_CODE['H']


def test_indicator_idw_keeps_semi_arid_on_a_semi_arid_anchor():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 2.0]])
    values = np.array([1.0, 3.0, 2.0])
    painted = renderer.indicator_idw(
        points, values, np.array([1.5]), np.array([2.0]), power=1.0
    )
    assert painted[0, 0] == renderer.CLIMATE_CODE['S']


def test_class_colors_are_solid_thirds_and_ignore_gradient_sharp():
    data = np.array([[1.0, 1.9, 3.0]])
    valid = np.ones_like(data, dtype=bool)
    soft = renderer.climate_values_to_rgb(data, valid, gradient_sharp=1.0)
    sharp = renderer.climate_values_to_rgb(data, valid, gradient_sharp=18.0)
    assert np.array_equal(soft, sharp)
    dry, semi, humid = (
        renderer._hex_to_rgb(renderer.CLIMATE_COLORS[code]) for code in ('D', 'S', 'H')
    )
    assert tuple(soft[0, 0]) == dry
    assert tuple(soft[0, 1]) == semi
    assert tuple(soft[0, 2]) == humid
