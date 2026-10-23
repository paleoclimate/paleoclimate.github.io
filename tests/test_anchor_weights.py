"""Anchor weights: citation reliability and the conceptual-anchor weight."""

from __future__ import annotations

import numpy as np
import pytest

import render_map as rm


def _point(lon, lat, climate, weight=None, ident='F1'):
    props = {'ID': ident, 'Climate_Cl': climate}
    if weight is not None:
        props['Weight'] = weight
    return {'type': 'Feature', 'properties': props,
            'geometry': {'type': 'Point', 'coordinates': [lon, lat]}}


def _conceptual(lon, lat, climate):
    return {'type': 'Feature', 'properties': {'ID': 'N/A', 'Climate_Cl': climate},
            'geometry': {'type': 'Point', 'coordinates': [lon, lat]}}


def _collection(*features):
    return {'type': 'FeatureCollection', 'features': list(features)}


def test_condensed_anchor_carries_the_mean_weight_of_its_winning_votes():
    data = _collection(_point(0, 0, 'D', 4, 'A'), _point(0.2, 0, 'D', 2, 'B'), _point(0.1, 0, 'H', 9, 'C'))
    interpolator, condensed, _ = rm.condense_proxy_points(data)
    (anchor,) = condensed['features']
    assert anchor['properties']['Climate_Cl'] == 'D'
    assert anchor['properties'][rm.ANCHOR_WEIGHT_KEY] == pytest.approx(3.0)


def test_stacked_citations_do_not_add_up_their_weights():
    data = _collection(*[_point(0, 0, 'S', 3, f'P{i}') for i in range(10)])
    _, condensed, _ = rm.condense_proxy_points(data)
    (anchor,) = condensed['features']
    assert anchor['properties'][rm.ANCHOR_WEIGHT_KEY] == pytest.approx(3.0)


def test_missing_or_invalid_weight_falls_back_to_one():
    assert rm.citation_weight({}) == rm.DEFAULT_CITATION_WEIGHT
    assert rm.citation_weight({'Weight': 'N/A'}) == rm.DEFAULT_CITATION_WEIGHT
    assert rm.citation_weight({'Weight': 0}) == rm.DEFAULT_CITATION_WEIGHT
    assert rm.citation_weight({'Weight': '7'}) == 7.0


def test_conceptual_anchors_enter_the_interpolator_with_half_weight():
    data = _collection(_point(0, 0, 'D', 4), _conceptual(10, 0, 'H'))
    interpolator, _, _ = rm.condense_proxy_points(data)
    _, values, weights = rm.extract_points_values_weights(interpolator)
    assert list(values) == [rm.CLIMATE_CODE['D'], rm.CLIMATE_CODE['H']]
    assert list(weights) == [4.0, rm.CONCEPTUAL_ANCHOR_WEIGHT]


def test_without_weights_the_indicator_idw_is_unchanged():
    points = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])
    values = np.array([1.0, 3.0, 2.0])
    lons = np.linspace(-2, 12, 15); lats = np.linspace(-2, 10, 13)
    plain = rm.indicator_idw(points, values, lons, lats, power=4.0)
    ones = rm.indicator_idw(points, values, lons, lats, power=4.0, weights=np.ones(3))
    assert np.array_equal(plain, ones)


def test_the_more_reliable_anchor_wins_the_midpoint():
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    values = np.array([rm.CLIMATE_CODE['D'], rm.CLIMATE_CODE['H']])
    lons = np.array([5.0]); lats = np.array([0.0])
    dry_reliable = rm.indicator_idw(points, values, lons, lats, power=4.0, weights=np.array([3.0, 1.0]))
    humid_reliable = rm.indicator_idw(points, values, lons, lats, power=4.0, weights=np.array([1.0, 3.0]))
    assert dry_reliable[0, 0] == rm.CLIMATE_CODE['D']
    assert humid_reliable[0, 0] == rm.CLIMATE_CODE['H']


def test_no_semi_arid_appears_between_dry_and_humid():
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    values = np.array([rm.CLIMATE_CODE['D'], rm.CLIMATE_CODE['H']])
    lons = np.linspace(0, 10, 101); lats = np.array([0.0])
    surface = rm.indicator_idw(points, values, lons, lats, power=4.0, weights=np.array([2.0, 5.0]))
    assert rm.CLIMATE_CODE['S'] not in set(surface.ravel())


def test_shares_sum_to_one_and_follow_the_weights():
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    values = np.array([rm.CLIMATE_CODE['D'], rm.CLIMATE_CODE['H']])
    _, shares = rm.indicator_idw_with_shares(points, values, np.array([5.0]), np.array([0.0]),
                                             power=4.0, weights=np.array([3.0, 1.0]))
    assert shares[:, 0, 0].sum() == pytest.approx(1.0)
    assert shares[0, 0, 0] == pytest.approx(0.75)


def test_wrong_number_of_weights_is_rejected():
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    with pytest.raises(ValueError):
        rm.indicator_idw(points, np.array([1.0, 3.0]), np.array([5.0]), np.array([0.0]),
                         power=4.0, weights=np.array([1.0]))
