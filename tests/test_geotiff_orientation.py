"""GeoTIFFs are written north-up: row 0 is the north edge, as every GIS expects."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

import render_map as rm

SOUTH = (10.0, -20.0)
NORTH = (10.0, 20.0)


@pytest.fixture()
def rasters(tmp_path):
    points = np.array([SOUTH, NORTH])
    values = np.array([1.0, 3.0])  # dry only in the south, humid only in the north
    classes = tmp_path / 'classes.tif'
    confidence = tmp_path / 'confidence.tif'
    rm.create_indicator_raster(points, values, str(classes), resolution=1.0, power=2.0,
                               confidence_path=str(confidence))
    return classes, confidence


def _sample(path, lon, lat):
    with rasterio.open(path) as src:
        return float(next(src.sample([(lon, lat)]))[0])


def test_transform_is_north_up(rasters):
    with rasterio.open(rasters[0]) as src:
        assert src.transform.e < 0
        assert src.transform.f == pytest.approx(src.bounds.top)


def test_class_raster_has_each_class_on_its_own_side(rasters):
    classes, _ = rasters
    assert _sample(classes, *SOUTH) == 1.0
    assert _sample(classes, *NORTH) == 3.0
    with rasterio.open(classes) as src:
        data = src.read(1)
        row_south, _ = src.index(*SOUTH)
        row_north, _ = src.index(*NORTH)
    assert row_north < row_south
    assert data[: data.shape[0] // 2].mean() > data[data.shape[0] // 2:].mean()


def test_confidence_raster_follows_the_same_orientation(tmp_path):
    # South: a dry and a humid anchor side by side (contested). North: one humid anchor
    # (uncontested). A raster stored upside down would swap the two readings.
    points = np.array([(8.0, -20.0), (12.0, -20.0), (10.0, 20.0)])
    values = np.array([1.0, 3.0, 3.0])
    classes, confidence = tmp_path / 'c.tif', tmp_path / 'conf.tif'
    rm.create_indicator_raster(points, values, str(classes), resolution=1.0, power=2.0,
                               confidence_path=str(confidence))
    with rasterio.open(classes) as a, rasterio.open(confidence) as b:
        assert a.transform == b.transform
        assert a.shape == b.shape
    south_midpoint = _sample(confidence, 10.0, -20.0)
    north_anchor = _sample(confidence, 10.0, 20.0)
    assert north_anchor > 0.9
    assert south_midpoint < 0.7
    assert _sample(classes, 10.0, 20.0) == 3.0
