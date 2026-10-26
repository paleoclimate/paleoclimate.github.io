"""Burgener et al. (2023) comparison: georeference, legend, alignment, viewer slot.

Synthetic images only; no project points or rasters are read.
"""
import cv2
import numpy as np
import pytest
from rasterio.transform import from_origin

import compare_burgener as burgener
import render_map
from compare_floegel import DRY, SEMI, HUMID

MARINE = (230, 230, 230)


def _legend(name):
    return next(rgb for n, rgb, _code in burgener.KOPPEN_LEGEND if n == name)


def _global_figure(width=1460, height=840, pad=20):
    """Equirectangular canvas with a black frame; returns (rgb, frame)."""
    rgb = np.full((height + 2 * pad, width + 2 * pad, 3), 255, np.uint8)
    x0, y0, x1, y1 = pad, pad, pad + width, pad + height
    rgb[y0:y1 + 1, x0:x1 + 1] = MARINE
    cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 0, 0), 2)
    return rgb, (x0, x1, y0, y1)


def _paint(rgb, frame, west, east, south, north, color):
    (px0, px1), (py1, py0) = (burgener.lonlat_to_pixel(frame, [west, east], [south, north]))
    rgb[int(py0):int(py1), int(px0):int(px1)] = color


def _meta(west, north, width, height, res=0.1):
    class Bounds:
        left, top = west, north
        right, bottom = west + width * res, north - height * res
    return {'transform': from_origin(west, north, res, res), 'bounds': Bounds,
            'width': width, 'height': height}


def test_all_nine_figures_are_in_the_inputs():
    assert burgener.images_present() == list(burgener.AGES)
    assert set(burgener.FIGURES) == set(burgener.AGES)


def test_frame_is_found_and_maps_lon_lat_linearly():
    rgb, frame = _global_figure()
    found = burgener.find_frame(rgb)
    assert all(abs(a - b) <= 2 for a, b in zip(found, frame))
    lon, lat = burgener.pixel_to_lonlat(frame, [frame[0], frame[1]], [frame[2], frame[3]])
    assert np.allclose(lon, [-180, 180]) and np.allclose(lat, [90, -90])
    x, y = burgener.lonlat_to_pixel(frame, 0.0, 0.0)
    assert x == pytest.approx((frame[0] + frame[1]) / 2)
    assert y == pytest.approx((frame[2] + frame[3]) / 2)


@pytest.mark.parametrize('name, code', [
    ('BWh', DRY), ('BWk', DRY), ('BSh', SEMI), ('BSk', SEMI),
    ('Af/Am', HUMID), ('As/Aw', HUMID), ('Ca', HUMID), ('Cc', HUMID), ('Dc/Do', HUMID),
])
def test_legend_colors_map_to_the_three_classes(name, code):
    patch = np.full((4, 4, 3), _legend(name), np.uint8)
    assert (burgener.classify(patch) == code).all()


def test_sea_ink_and_paper_are_not_land():
    for color in (MARINE, (0, 0, 0), (255, 255, 255)):
        patch = np.full((4, 4, 3), color, np.uint8)
        assert (burgener.classify(patch) == 0).all()


def test_sensitivity_counts_seasonal_tropics_as_semiarid():
    patch = np.full((4, 4, 3), _legend('As/Aw'), np.uint8)
    assert (burgener.classify(patch, burgener.SENSITIVITY_OVERRIDES) == SEMI).all()


def test_majority_vote_removes_a_graticule_line_across_a_cell_block():
    rgb = np.full((40, 40, 3), _legend('BWh'), np.uint8)
    rgb[20, :] = (200, 200, 200)            # graticule over land
    rgb[:, 10] = _legend('BSh')             # one-pixel anti-aliased seam
    cells = burgener.classify_cells(rgb)
    assert (cells[5:35, 5:35] == DRY).all()


def test_rotation_and_shift_invert_exactly():
    params, center = (7.5, -3.0, 2.0), (-20.0, -10.0)
    lons = np.array([-40.0, -20.0, 5.0]); lats = np.array([-30.0, -10.0, 15.0])
    x, y = burgener._rotate_shift(params, center, lons, lats)
    back = burgener._unrotate_shift(params, center, x, y)
    assert np.allclose(back[0], lons) and np.allclose(back[1], lats)


def test_frame_warp_puts_a_block_on_its_raster_cells():
    rgb, frame = _global_figure()
    _paint(rgb, frame, -40, -20, -20, 0, _legend('BWh'))
    meta = _meta(-60.0, 20.0, 600, 600)       # 60W..0, 40S..20N at 0.1 deg
    cells = burgener.classify_cells(rgb).astype(np.uint8)
    warped = burgener.warp_to_raster(cells, frame, meta)
    # lon -30, lat -10 -> col 300, row 300 (inside); lon -50, lat 10 -> sea
    assert warped[300, 300] == DRY
    assert warped[100, 100] == 0


def test_shoreline_fit_recovers_a_known_offset(monkeypatch):
    rgb, frame = _global_figure()
    # An L-shaped continent, so no rotation maps it onto itself.
    _paint(rgb, frame, -45, -15, -35, -25, _legend('Ca'))
    _paint(rgb, frame, -45, -35, -25, -5, _legend('Ca'))
    shift = (4.0, -3.0)                        # our continent sits 4E, 3S of theirs

    def fake_coastline(image, meta, age):
        inverse = ~meta['transform']
        corners = [(-45, -35), (-15, -35), (-15, -25), (-35, -25), (-35, -5), (-45, -5)]
        pts = []
        for lon, lat in corners:
            col, row = inverse @ (lon + shift[0], lat + shift[1])
            pts.append((int(round(col)), int(round(row))))
        cv2.polylines(image, [np.array(pts, np.int32)], True, (0, 0, 0), 1)
        return image

    monkeypatch.setattr(burgener.cf, 'draw_coastline', fake_coastline)
    meta = _meta(-70.0, 15.0, 800, 700)
    params, before, after = burgener.fit_shoreline('105', rgb, frame, meta)
    assert params[0] == pytest.approx(0.0, abs=0.5)
    assert params[1] == pytest.approx(shift[0], abs=0.4)
    assert params[2] == pytest.approx(shift[1], abs=0.4)
    assert after < before


def test_viewer_template_has_the_burgener_slot():
    html = render_map.INDEX_TEMPLATE
    assert 'id="burgenerBtn"' in html
    assert 'setReportSlot(burgenerBtn, entry.burgener)' in html
    assert '#burgenerBtn.is-absent' in html
    assert '>Comparison Burgener (2023)</button>' in html
    assert '>Comparison Hay &amp; Flögel (2012)</button>' in html
