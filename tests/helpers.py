"""Shared helpers for catalog parsing and Playwright UI checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = REPO_ROOT / 'index.html'
KNN_DIR = REPO_ROOT / 'GENERATED_KNN_IDW_MAPS'
IDW_DIR = REPO_ROOT / 'GENERATED_IDW_MAPS'
COMPARISON_DIR = REPO_ROOT / 'COMPARISON'

LAYER_NAMES = ('Raster', 'Coastlines', 'Data points', 'Color stats')
COMPARISON_AGES = {105, 115}
STORAGE_KEY = 'pcvs:age'

FIND_LEAFLET_MAP = """
() => {
  for (const key of Object.keys(window)) {
    const value = window[key];
    if (value && typeof value.getZoom === 'function'
        && typeof value.getContainer === 'function'
        && value._container) {
      return key;
    }
  }
  return null;
}
"""

MAP_STATE = """
() => {
  for (const key of Object.keys(window)) {
    const value = window[key];
    if (value && typeof value.getZoom === 'function'
        && typeof value.getContainer === 'function'
        && value._container) {
      const center = value.getCenter();
      const size = value.getSize();
      const bounds = value.getBounds();
      return {
        zoom: value.getZoom(),
        lat: center.lat,
        lng: center.lng,
        width: size.x,
        height: size.y,
        south: bounds.getSouth(),
        north: bounds.getNorth(),
        west: bounds.getWest(),
        east: bounds.getEast(),
      };
    }
  }
  return null;
}
"""


def load_maps_catalog(index_path: Path | None = None) -> list[dict]:
    """Parse the ``MAPS`` array embedded in the viewer."""
    path = index_path or INDEX_HTML
    if not path.is_file():
        raise FileNotFoundError(f'Viewer not found: {path}')
    text = path.read_text(encoding='utf-8')
    match = re.search(r'var MAPS = (\[.*?\]);', text, flags=re.S)
    if not match:
        raise ValueError('Could not parse MAPS from index.html')
    maps = json.loads(match.group(1))
    if not maps:
        raise ValueError('MAPS catalog is empty')
    return maps


def catalog_ages(maps: list[dict] | None = None) -> list[int]:
    entries = maps if maps is not None else load_maps_catalog()
    return [int(entry['age']) for entry in entries]


def entry_for_age(age: int, maps: list[dict] | None = None) -> dict:
    entries = maps if maps is not None else load_maps_catalog()
    for entry in entries:
        if int(entry['age']) == int(age):
            return entry
    raise KeyError(f'Age {age} is not in the MAPS catalog')


def representative_ages(maps: list[dict] | None = None) -> list[int]:
    """Youngest, a mid age, oldest, plus every age that has a comparison report."""
    entries = maps if maps is not None else load_maps_catalog()
    ages = [int(entry['age']) for entry in entries]
    picks = {ages[0], ages[len(ages) // 2], ages[-1]}
    for entry in entries:
        if entry.get('comparison'):
            picks.add(int(entry['age']))
    return sorted(picks)


def knn_html_files() -> list[Path]:
    if not KNN_DIR.is_dir():
        return []
    return sorted(
        path for path in KNN_DIR.glob('map_*.html')
        if path.is_file()
    )


def idw_html_files() -> list[Path]:
    if not IDW_DIR.is_dir():
        return []
    return sorted(
        path for path in IDW_DIR.glob('map_*_idw_*.html')
        if path.is_file()
    )


def overlay_png_for(html_path: Path) -> Path:
    name = html_path.name
    if not name.startswith('map_'):
        raise ValueError(html_path)
    return html_path.with_name('raster_overlay_' + name[len('map_'):]).with_suffix('.png')


def pdf_path_for(html_path: Path, scope: str) -> Path:
    return html_path.with_name(f'{html_path.stem}_{scope}.pdf')


def pdf_files_for(html_path: Path) -> dict[str, Path]:
    return {scope: pdf_path_for(html_path, scope) for scope in ('full', 'raster')}


def is_valid_pdf(path: Path, min_bytes: int = 1000) -> bool:
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    with path.open('rb') as handle:
        return handle.read(5) == b'%PDF-'


def embedded_basins(html_text: str) -> list[str]:
    """Parse the basin list Folium embeds in a generated map."""
    match = re.search(r'var allBasins = (\[.*?]);', html_text)
    if not match:
        raise ValueError('Could not parse var allBasins from map HTML')
    basins = json.loads(match.group(1))
    if not isinstance(basins, list) or not basins:
        raise ValueError('allBasins is empty')
    return [str(name) for name in basins]


def repo_url(base_url: str, relative: str) -> str:
    return f"{base_url.rstrip('/')}/{relative.lstrip('/')}"


def _clear_page_errors(page) -> None:
    errors = getattr(page, '_pcvs_errors', None)
    if errors is not None:
        errors.clear()


def wait_viewer_ready(page, timeout: int = 60_000) -> None:
    """Wait until the iframe map has Leaflet, PCVS, and the PDF button enabled."""
    page.wait_for_function(
        """() => {
          const frame = document.getElementById('mapFrame');
          const pdf = document.getElementById('pdfBtn');
          const loader = document.getElementById('loader');
          if (!frame || !frame.contentWindow || !pdf || !loader) return false;
          const child = frame.contentWindow;
          return !!(child.L && child.PCVS && !pdf.disabled
                    && !loader.classList.contains('on'));
        }""",
        timeout=timeout,
    )
    _clear_page_errors(page)


def wait_leaflet_ready(page, timeout: int = 60_000) -> None:
    """Wait until a standalone Folium map page is interactive."""
    page.wait_for_function(
        """() => {
          if (!window.L || !window.PCVS) return false;
          const container = document.querySelector('.leaflet-container');
          if (!container) return false;
          const rect = container.getBoundingClientRect();
          return rect.width > 80 && rect.height > 80;
        }""",
        timeout=timeout,
    )
    _clear_page_errors(page)


def map_frame(page):
    """Return the Playwright frame that hosts the generated map."""
    wait_viewer_ready(page)
    for frame in page.frames:
        url = frame.url or ''
        if 'GENERATED_KNN_IDW_MAPS' in url or 'GENERATED_IDW_MAPS' in url:
            return frame
    raise RuntimeError('Map iframe did not load a generated map')


def current_viewer_age(page) -> int:
    label = page.locator('#ageBtnLabel').inner_text().strip()
    match = re.match(r'(\d+)', label)
    if not match:
        raise AssertionError(f'Could not read current age from {label!r}')
    return int(match.group(1))


def select_age_via_combo(page, age: int) -> None:
    combo = page.locator('#ageCombo')
    if not combo.evaluate("el => el.classList.contains('open')"):
        page.locator('#ageBtn').click()
    option = page.locator('#ageMenu .combo-opt').filter(
        has_text=re.compile(rf'^{age} Ma$')
    )
    option.evaluate("el => el.scrollIntoView({block: 'nearest'})")
    option.click()
    wait_viewer_ready(page)


def goto_viewer(page, base_url: str, age: int | None = None) -> None:
    url = repo_url(base_url, 'index.html')
    if age is not None:
        url = f'{url}#age={age}'
    last_error = None
    for _ in range(3):
        _clear_page_errors(page)
        page.goto(url, wait_until='domcontentloaded', timeout=60_000)
        try:
            wait_viewer_ready(page)
            return
        except Exception as exc:
            last_error = exc
            page.wait_for_timeout(400)
    raise last_error


def boxes_overlap(a: dict, b: dict, slack: float = 1.0) -> bool:
    return not (
        a['x'] + a['width'] <= b['x'] + slack
        or b['x'] + b['width'] <= a['x'] + slack
        or a['y'] + a['height'] <= b['y'] + slack
        or b['y'] + b['height'] <= a['y'] + slack
    )


def layer_labels(frame) -> list[str]:
    return frame.evaluate(
        """() => Array.from(
             document.querySelectorAll('.leaflet-control-layers-overlays label')
           ).map(node => node.textContent.replace(/\\s+/g, ' ').trim())"""
    )


def overlay_checked(frame, name: str) -> bool:
    return frame.evaluate(
        """(name) => {
          const labels = document.querySelectorAll('.leaflet-control-layers-overlays label');
          for (const label of labels) {
            const text = label.textContent.replace(/\\s+/g, ' ').trim();
            if (text === name || text.endsWith(name)) {
              const input = label.querySelector('input');
              return !!(input && input.checked);
            }
          }
          return null;
        }""",
        name,
    )


def toggle_overlay(frame, name: str) -> None:
    labels = frame.locator('.leaflet-control-layers-overlays label')
    count = labels.count()
    for index in range(count):
        label = labels.nth(index)
        text = re.sub(r'\s+', ' ', label.inner_text()).strip()
        if text == name or text.endswith(name):
            label.locator('input').click()
            return
    raise AssertionError(f'Layer {name!r} not found in {layer_labels(frame)}')


def open_first_point_popup(frame) -> bool:
    return frame.evaluate(
        """() => {
          for (const key of Object.keys(window)) {
            const map = window[key];
            if (!(map && typeof map.eachLayer === 'function' && map._container)) {
              continue;
            }
            let opened = false;
            map.eachLayer(function(layer) {
              if (opened || !layer.eachLayer) return;
              layer.eachLayer(function(child) {
                if (opened) return;
                if (child.getLatLng && child.getPopup && child.openPopup) {
                  child.openPopup();
                  opened = true;
                }
              });
            });
            return opened;
          }
          return false;
        }"""
    )
