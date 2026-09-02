"""UI/UX of the generated Folium maps (viewer iframe and standalone pages)."""

from __future__ import annotations

import pytest

from tests.helpers import (
    LAYER_NAMES,
    MAP_STATE,
    goto_viewer,
    idw_html_files,
    knn_html_files,
    layer_labels,
    load_maps_catalog,
    map_frame,
    open_first_point_popup,
    overlay_checked,
    representative_ages,
    repo_url,
    toggle_overlay,
    wait_leaflet_ready,
)


def _color_stats_percents(frame) -> list[float]:
    return frame.evaluate(
        """() => {
          const cells = document.querySelectorAll(
            '.color-stats-control table tr td:last-child'
          );
          return Array.from(cells).map(cell => parseFloat(cell.textContent));
        }"""
    )


@pytest.mark.parametrize('age', representative_ages())
def test_map_controls_and_layers(page, base_url, age):
    goto_viewer(page, base_url, age=age)
    frame = map_frame(page)

    assert frame.locator('.leaflet-container').is_visible()
    assert frame.locator('.leaflet-control-zoom').is_visible()
    assert frame.locator('.leaflet-control-layers').is_visible()
    assert frame.locator(
        '.leaflet-control-zoom-fullscreen, .leaflet-control-fullscreen, a[title="Full Screen"]'
    ).count() >= 1
    assert frame.locator('.leaflet-control-measure').count() >= 1
    assert frame.locator('.basin-filter-control').is_visible()
    assert frame.locator('.color-stats-control').is_visible()
    assert frame.locator('.graticule-label').count() >= 2

    labels = layer_labels(frame)
    for name in LAYER_NAMES:
        assert any(name in label for label in labels), f'{name} missing from {labels}'
        assert overlay_checked(frame, name) is True

    state = frame.evaluate(MAP_STATE)
    assert state is not None
    assert state['width'] > 200 and state['height'] > 200
    assert state['east'] > state['west']
    assert state['north'] > state['south']
    assert frame.evaluate('() => !!window.PCVS && typeof window.PCVS.exportSize === "function"')


@pytest.mark.parametrize('age', representative_ages())
def test_layer_toggles_change_what_is_on_screen(page, base_url, age):
    goto_viewer(page, base_url, age=age)
    frame = map_frame(page)

    overlay_count = frame.locator('.leaflet-overlay-pane img, .leaflet-overlay-pane canvas').count()
    assert overlay_count >= 1, 'Interpolated raster overlay is missing'

    toggle_overlay(frame, 'Color stats')
    frame.wait_for_function(
        "() => !document.querySelector('.color-stats-control')"
    )
    toggle_overlay(frame, 'Color stats')
    frame.wait_for_selector('.color-stats-control')

    toggle_overlay(frame, 'Raster')
    assert overlay_checked(frame, 'Raster') is False
    toggle_overlay(frame, 'Raster')
    assert overlay_checked(frame, 'Raster') is True


@pytest.mark.parametrize('age', representative_ages())
def test_basin_filter_search_clear_and_restore(page, base_url, age):
    goto_viewer(page, base_url, age=age)
    frame = map_frame(page)
    panel = frame.locator('.basin-filter-control')
    assert panel.is_visible()

    if not panel.evaluate("el => el.classList.contains('open')"):
        panel.locator('.basin-filter-header').click()
    assert panel.evaluate("el => el.classList.contains('open')")

    checkboxes = frame.locator('.basin-filter-list input[type="checkbox"]')
    assert checkboxes.count() >= 1
    assert frame.locator('.basin-filter-search').get_attribute('placeholder')

    badge = frame.locator('.basin-filter-count').inner_text().strip()
    assert re_match_count(badge)

    frame.locator('.basin-filter-search').fill('zzzz-no-such-basin')
    assert frame.locator('.basin-filter-empty').is_visible()
    hidden = frame.locator('.basin-filter-list li.hidden').count()
    assert hidden == checkboxes.count()

    frame.locator('.basin-filter-search').fill('')
    assert not frame.locator('.basin-filter-empty').is_visible()

    frame.locator('.basin-filter-actions button', has_text='Clear').click()
    cleared = frame.locator('.basin-filter-count').inner_text().strip()
    assert cleared.startswith('0 /')

    frame.locator('.basin-filter-actions button', has_text='Select all').click()
    restored = frame.locator('.basin-filter-count').inner_text().strip()
    shown, total = [int(part) for part in restored.split('/')]
    assert shown == total
    assert shown > 0

    frame.locator('.basin-filter-search').press('Escape')
    frame.wait_for_timeout(200)
    assert not panel.evaluate("el => el.classList.contains('open')")


def re_match_count(badge: str) -> bool:
    parts = badge.replace(' ', '').split('/')
    assert len(parts) == 2, badge
    shown, total = int(parts[0]), int(parts[1])
    assert 0 <= shown <= total
    assert total > 0
    return True


@pytest.mark.parametrize('age', representative_ages())
def test_color_stats_cover_the_three_climate_classes(page, base_url, age):
    goto_viewer(page, base_url, age=age)
    frame = map_frame(page)
    panel = frame.locator('.color-stats-control')
    text = panel.inner_text()
    lowered = text.lower()
    for label in ('dry', 'semi-arid', 'humid', 'raster coverage'):
        assert label in lowered, f'{label} missing from color stats'
    percents = _color_stats_percents(frame)
    assert len(percents) == 3
    assert all(value >= 0 for value in percents)
    assert 99.0 <= sum(percents) <= 101.0


@pytest.mark.parametrize('age', representative_ages())
def test_data_point_popup_describes_a_formation(page, base_url, age):
    goto_viewer(page, base_url, age=age)
    frame = map_frame(page)
    assert open_first_point_popup(frame)
    popup = frame.locator('.leaflet-popup, .pcvs-popup')
    popup.first.wait_for(state='visible')
    text = popup.first.inner_text()
    assert 'Basin' in text
    assert 'Climate' in text
    assert any(word in text for word in ('Humid', 'Dry', 'Semi-arid', 'Data point'))


@pytest.mark.parametrize('age', representative_ages())
def test_zoom_and_export_api_do_not_break_the_map(page, base_url, age):
    goto_viewer(page, base_url, age=age)
    frame = map_frame(page)
    before = frame.evaluate(MAP_STATE)
    frame.locator('.leaflet-control-zoom-in').click()
    frame.wait_for_timeout(400)
    after_zoom = frame.evaluate(MAP_STATE)
    assert after_zoom['zoom'] >= before['zoom']

    size = frame.evaluate("scope => window.PCVS.exportSize(scope)", 'raster')
    assert size['width'] >= 640
    assert size['height'] >= 200

    frame.evaluate("() => window.PCVS.beginExport({scope: 'full', resize: false, veil: false})")
    frame.evaluate("() => window.PCVS.endExport()")
    restored = frame.evaluate(MAP_STATE)
    assert abs(restored['lat'] - before['lat']) < 2
    assert abs(restored['lng'] - before['lng']) < 2
    assert restored['width'] > 200 and restored['height'] > 200
    assert not frame.evaluate(
        "() => document.documentElement.classList.contains('pcvs-exporting')"
    )
    assert frame.locator('.leaflet-control-zoom').is_visible()


def test_measure_and_fullscreen_controls_are_usable(page, base_url):
    maps = load_maps_catalog()
    goto_viewer(page, base_url, age=int(maps[len(maps) // 2]['age']))
    frame = map_frame(page)

    measure = frame.locator('.leaflet-control-measure').first
    assert measure.is_visible()
    measure.click()
    frame.wait_for_timeout(200)

    fullscreen = frame.locator(
        '.leaflet-control-zoom-fullscreen, .leaflet-control-fullscreen, a[title="Full Screen"]'
    )
    assert fullscreen.count() >= 1
    assert fullscreen.first.is_visible()


def test_standalone_knn_and_idw_pages_boot(page, base_url):
    knn = knn_html_files()
    idw = idw_html_files()
    assert knn and idw
    samples = [knn[0], knn[len(knn) // 2], knn[-1], idw[len(idw) // 2]]
    for path in samples:
        page.goto(
            repo_url(base_url, f'{path.parent.name}/{path.name}'),
            wait_until='domcontentloaded',
            timeout=60_000,
        )
        wait_leaflet_ready(page)
        assert page.locator('.leaflet-container').is_visible()
        assert page.evaluate('() => !!window.PCVS')
        labels = layer_labels(page)
        assert any('Raster' in label for label in labels)
        assert page.locator('.basin-filter-control').is_visible()


def test_every_knn_map_file_loads_leaflet(page, base_url):
    files = knn_html_files()
    assert files
    for path in files:
        page.goto(
            repo_url(base_url, f'{path.parent.name}/{path.name}'),
            wait_until='domcontentloaded',
            timeout=60_000,
        )
        wait_leaflet_ready(page)
        state = page.evaluate(MAP_STATE)
        assert state and state['width'] > 80 and state['height'] > 80
        assert page.locator('.leaflet-tile-pane, .leaflet-overlay-pane').count() >= 1


MARKER_GEOMETRY = """
() => {
  const circles = [];
  const icons = [];
  for (const key of Object.keys(window)) {
    const map = window[key];
    if (!(map && typeof map.eachLayer === 'function' && map._container)) {
      continue;
    }
    map.eachLayer(function(layer) {
      if (!layer.eachLayer) return;
      layer.eachLayer(function(child) {
        if (child.options && typeof child.options.radius === 'number') {
          const node = child.getElement && child.getElement();
          const box = node ? node.getBoundingClientRect() : null;
          circles.push({
            radius: child.options.radius,
            weight: child.options.weight,
            className: child.options.className || '',
            width: box ? box.width : null,
            height: box ? box.height : null,
          });
        }
        if (child._icon) {
          const svg = child._icon.querySelector('svg');
          if (svg) {
            const box = svg.getBoundingClientRect();
            icons.push({
              width: parseFloat(svg.getAttribute('width')),
              height: parseFloat(svg.getAttribute('height')),
              box: box.width,
            });
          }
        }
      });
    });
  }
  return {circles, icons};
}
"""


def _assert_published_point_size(geometry):
    import render_map as renderer

    circles = geometry['circles']
    icons = geometry['icons']
    assert circles, 'No CircleMarker data points on the map'
    for marker in circles:
        assert marker['radius'] == renderer.POINT_RADIUS_PX
        assert marker['weight'] == renderer.POINT_WEIGHT_PX
        assert 'pcvs-point' in marker['className']
        if marker['width'] is not None:
            assert abs(marker['width'] - renderer.POINT_OUTER_PX) <= 2.5
            assert abs(marker['height'] - renderer.POINT_OUTER_PX) <= 2.5
    for icon in icons:
        assert icon['width'] == renderer.POINT_OUTER_PX
        assert icon['height'] == renderer.POINT_OUTER_PX
        if icon['box']:
            assert abs(icon['box'] - renderer.POINT_OUTER_PX) <= 2.5


def test_basin_filter_shows_accented_names(page, base_url):
    goto_viewer(page, base_url)
    frame = map_frame(page)
    panel = frame.locator('.basin-filter-control')
    if not panel.evaluate("el => el.classList.contains('open')"):
        panel.locator('.basin-filter-header').click()
    names = [
        text.strip()
        for text in frame.locator('.basin-filter-list label').all_inner_texts()
    ]
    assert names
    broken = [name for name in names if '?' in name]
    assert not broken, f'Basin select still shows broken accents: {broken}'
    joined = ' '.join(names)
    assert any(mark in joined for mark in 'áãéíóúçêâñèïü'), (
        'Basin select has no accented letters'
    )


def test_basin_select_filters_points(page, base_url):
    goto_viewer(page, base_url)
    frame = map_frame(page)
    panel = frame.locator('.basin-filter-control')
    if not panel.evaluate("el => el.classList.contains('open')"):
        panel.locator('.basin-filter-header').click()

    labels = frame.locator('.basin-filter-list label')
    assert labels.count() >= 2

    target = labels.first
    for index in range(labels.count()):
        text = labels.nth(index).inner_text().strip()
        if any(mark in text for mark in 'áàâãéêíóôõúçñèïü'):
            target = labels.nth(index)
            break

    checkbox = target.locator('input[type="checkbox"]')
    name = checkbox.get_attribute('value')
    assert name
    assert '?' not in name

    before = frame.locator('.basin-filter-count').inner_text().strip()
    shown_before, total = [int(part) for part in before.replace(' ', '').split('/')]
    assert shown_before == total
    assert total > 0

    checkbox.uncheck()
    after = frame.locator('.basin-filter-count').inner_text().strip()
    shown_after, total_after = [int(part) for part in after.replace(' ', '').split('/')]
    assert total_after == total
    assert shown_after < shown_before

    checkbox.check()
    restored = frame.locator('.basin-filter-count').inner_text().strip()
    shown_restored, _ = [int(part) for part in restored.replace(' ', '').split('/')]
    assert shown_restored == shown_before


def test_data_point_size_on_html_and_pdf_export(page, base_url):
    goto_viewer(page, base_url)
    frame = map_frame(page)
    geometry = frame.evaluate(MARKER_GEOMETRY)
    _assert_published_point_size(geometry)

    frame.evaluate("() => window.PCVS.beginExport({scope: 'full', resize: false, veil: false})")
    exported = frame.evaluate(MARKER_GEOMETRY)
    _assert_published_point_size(exported)
    frame.evaluate("() => window.PCVS.endExport()")

    frame.evaluate("() => window.PCVS.beginExport({scope: 'raster', resize: false, veil: false})")
    raster = frame.evaluate(MARKER_GEOMETRY)
    _assert_published_point_size(raster)
    frame.evaluate("() => window.PCVS.endExport()")


def test_pre_rendered_pdfs_are_served(page, base_url):
    maps = load_maps_catalog()
    entry = maps[len(maps) // 2]
    assert entry.get('pdf'), f'{entry["path"]} has no PDF catalog entries'
    for scope, href in entry['pdf'].items():
        response = page.request.get(repo_url(base_url, href))
        assert response.ok, f'{scope} PDF is not served: {href}'
        body = response.body()
        assert body.startswith(b'%PDF-'), href
        assert len(body) > 1000, href


@pytest.mark.slow
def test_live_pdf_export_from_viewer(page, base_url):
    goto_viewer(page, base_url)
    page.locator('#rasterOnly').uncheck()
    with page.expect_download(timeout=90_000) as download_info:
        page.locator('#pdfBtn').click()
    download = download_info.value
    assert download.suggested_filename.endswith('.pdf')
    body = download.path()
    if body:
        assert body.stat().st_size > 1000
        header = body.read_bytes()[:5]
        assert header == b'%PDF-'
