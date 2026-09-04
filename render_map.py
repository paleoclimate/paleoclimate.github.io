"""
Render paleogeographic maps from 110 million years ago using Folium.
This script:
1. Loads GeoJSON files with points and coastlines
2. Applies KNN smoothing on point data, then runs IDW interpolation
3. Generates two maps:
   - Map 1: Original data (points + coastlines + original raster if exists)
   - Map 2: KNN + IDW interpolated data (points + coastlines + KNN+IDW raster)
"""

import argparse
import sys
import threading
import time
from collections import Counter, OrderedDict
import folium
from folium.raster_layers import ImageOverlay
from branca.element import MacroElement, Template
import json
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
import numpy as np
from folium import plugins
import os
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from PIL import Image
import cv2

try:
    import resource
except ImportError:  # not available on Windows
    resource = None

try:
    import psutil
except ImportError:  # optional; /proc is used instead on Linux
    psutil = None

# Neighbor-search backends for KNN smoothing and IDW interpolation.
# Both produce the same weighted averages; they differ only in how the
# nearest neighbors are found (exhaustive distance matrix vs. k-d tree).
METHOD_BRUTE = 'brute'
METHOD_KDTREE = 'kdtree'

METHOD_LABELS = {
    METHOD_BRUTE: 'Brute Force',
    METHOD_KDTREE: 'k-d Tree',
}

# Layer names. They are shown in the map's layer control and are also the
# handles the PDF exporter uses to keep a layer visible or hidden, so both
# sides must agree on the exact strings.
LAYER_RASTER = 'Raster'
LAYER_COASTLINES = 'Coastlines'
LAYER_POINTS = 'Data points'
LAYER_COLOR_STATS = 'Color stats'

# Climate classification palette, shared by markers and the raster ramp.
# Muted cartographic tones: saturated primaries read as a toy map, and the
# raster covers most of the canvas. Hues stay inside the yellow / green / blue
# bands ``classificar_por_intervalos`` counts pixels with.
CLIMATE_COLORS = {
    'H': '#3a7594',   # Humid — dusty ocean
    'S': '#5d8750',   # Semi-arid — sage
    'D': '#c9b05a',   # Dry — ochre
}
CLIMATE_LABELS = {
    'H': 'Humid',
    'S': 'Semi-arid',
    'D': 'Dry',
}

# Neutral canvas behind the coastlines and the interpolated raster.
MAP_BACKGROUND = '#eef0f2'

# Coastlines and the graticule, kept dark enough to read over the raster
# without competing with the data points.
COASTLINE_COLOR = '#334155'
GRATICULE_COLOR = '#94a3b8'

# Opacity of the interpolated surface. High enough for the palette to keep its
# depth, low enough for the coastlines underneath to stay legible.
RASTER_OPACITY = 0.78

# Width, as a fraction of the shorter side, over which the interpolated surface
# fades out at its border.
RASTER_EDGE_FEATHER = 0.02


def _hex_to_rgb(value):
    """Convert ``'#rrggbb'`` to an ``(r, g, b)`` tuple of 0-255 ints."""
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def climate_values_to_rgb(data, valid_mask, gradient_sharp, expected_min=1.0,
                          expected_max=3.0):
    """Map climate values in [1, 3] onto the published Dry/Semi-arid/Humid ramp."""
    data_clamped = np.clip(data, expected_min, expected_max)
    normalized = (data_clamped - expected_min) / (expected_max - expected_min)
    normalized = np.clip((normalized - 0.5) * gradient_sharp + 0.5, 0, 1)

    dry = np.array(_hex_to_rgb(CLIMATE_COLORS['D']), dtype=np.float32)
    semi = np.array(_hex_to_rgb(CLIMATE_COLORS['S']), dtype=np.float32)
    humid = np.array(_hex_to_rgb(CLIMATE_COLORS['H']), dtype=np.float32)

    rgb = np.zeros(data.shape + (3,), dtype=np.float32)
    lower = (normalized <= 0.5) & valid_mask
    upper = (normalized > 0.5) & valid_mask
    rgb[lower] = dry + (semi - dry) * (normalized[lower] * 2.0)[:, None]
    rgb[upper] = semi + (humid - semi) * ((normalized[upper] - 0.5) * 2.0)[:, None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _feather_edges(alpha):
    """Fade an alpha channel out towards the border of the raster.

    A hard-edged rectangle reads as a sticker pasted over the map, and the
    interpolation is least constrained exactly there, where it has points on
    one side only. Only the display image is softened; the GeoTIFF keeps its
    values.
    """
    width = max(4, int(round(min(alpha.shape) * RASTER_EDGE_FEATHER)))
    ramp = np.linspace(0.12, 1.0, width + 1, dtype=np.float32)[1:]
    scale = np.ones(alpha.shape, dtype=np.float32)
    scale[:width, :] = np.minimum(scale[:width, :], ramp[:, None])
    scale[-width:, :] = np.minimum(scale[-width:, :], ramp[::-1, None])
    scale[:, :width] = np.minimum(scale[:, :width], ramp[None, :])
    scale[:, -width:] = np.minimum(scale[:, -width:], ramp[::-1][None, :])
    return (alpha * scale).astype(np.uint8)

# Page geometry for exported PDFs. The page is sized to the aspect ratio of the
# exported region so the map fills it edge to edge with no white borders.
PDF_BASE_HEIGHT_PX = 900
PDF_MIN_WIDTH_PX = 640
PDF_MAX_WIDTH_PX = 2200

# Client-side export dependencies, loaded on demand from a CDN.
# The full-map PDF is drawn as vectors (circles, coastlines) so the points
# stay editable. html-to-image is only used for the raster-only crop, which
# may stay a bitmap, and for the small coverage panel on a vector page.
HTML_TO_IMAGE_URL = 'https://cdn.jsdelivr.net/npm/html-to-image@1.11.13/dist/html-to-image.js'
JSPDF_URL = 'https://cdn.jsdelivr.net/npm/jspdf@2.5.2/dist/jspdf.umd.min.js'


def method_label(method):
    """Human-readable name of a neighbor-search backend."""
    return METHOD_LABELS.get(method, method)

def _format_duration(seconds):
    """Format elapsed seconds for human-readable output."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {secs:.0f}s"


def _format_bytes(num_bytes, signed=False):
    """Format a byte count for human-readable output."""
    if num_bytes is None:
        return "n/a"
    sign = ('-' if num_bytes < 0 else '+') if signed else ''
    value = float(abs(num_bytes))
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if value < 1024 or unit == 'TB':
            precision = 0 if unit == 'B' else 1
            return f"{sign}{value:.{precision}f} {unit}"
        value /= 1024


_PROCESS = psutil.Process() if psutil is not None else None
_PAGE_SIZE = os.sysconf('SC_PAGE_SIZE') if hasattr(os, 'sysconf') else 4096


def _current_rss():
    """Resident set size of this process in bytes, or None if unavailable."""
    if _PROCESS is not None:
        try:
            return _PROCESS.memory_info().rss
        except Exception:
            pass
    try:
        with open('/proc/self/statm', 'r') as f:
            return int(f.read().split()[1]) * _PAGE_SIZE
    except (OSError, IndexError, ValueError):
        return None


def _peak_rss():
    """Peak resident set size of this process in bytes, or None if unavailable."""
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kibibytes; macOS/BSD report bytes.
    return peak if sys.platform == 'darwin' else peak * 1024


class _RssSampler:
    """Background sampler that tracks peak RSS inside open measurement windows.

    Peak memory cannot be read after the fact, so RSS is polled on a daemon
    thread while steps run. Each window (a step, a dataset, the whole script)
    registers a watcher that keeps the highest value seen while it is open.
    """

    def __init__(self, interval=0.02):
        self.interval = interval
        self._watchers = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None or _current_rss() is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.interval):
            self.sample()

    def sample(self):
        """Read RSS now and feed it to every open watcher."""
        rss = _current_rss()
        if rss is None:
            return None
        with self._lock:
            for watcher in self._watchers:
                if rss > watcher['peak']:
                    watcher['peak'] = rss
        return rss

    def watch(self, start_rss=None):
        watcher = {'peak': start_rss or 0}
        with self._lock:
            self._watchers.append(watcher)
        return watcher

    def release(self, watcher):
        """Close a watcher and return the peak RSS observed while it was open."""
        self.sample()
        with self._lock:
            self._watchers = [w for w in self._watchers if w is not watcher]
        return watcher['peak'] or None


_RSS_SAMPLER = _RssSampler()


def _format_memory(delta, peak):
    """Format the RAM part of a metrics line, or '' when RSS is unavailable."""
    parts = []
    if delta is not None:
        parts.append(f"Δ {_format_bytes(delta, signed=True)}")
    if peak is not None:
        parts.append(f"peak {_format_bytes(peak)}")
    return ", ".join(parts)


def _format_metrics(elapsed, rss_delta=None, rss_peak=None):
    """Format elapsed time plus RAM usage for human-readable output."""
    memory = _format_memory(rss_delta, rss_peak)
    if not memory:
        return _format_duration(elapsed)
    return f"{_format_duration(elapsed)} | RAM {memory}"


class StepTimer:
    """Context manager that records and prints step elapsed time and RAM usage."""

    def __init__(self, label, indent=1):
        self.label = label
        self.indent = indent
        self.elapsed = None
        self.rss_start = None
        self.rss_end = None
        self.rss_delta = None
        self.rss_peak = None

    def __enter__(self):
        self.rss_start = _RSS_SAMPLER.sample()
        self._watcher = _RSS_SAMPLER.watch(self.rss_start)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self._start
        self.rss_end = _RSS_SAMPLER.sample()
        self.rss_peak = _RSS_SAMPLER.release(self._watcher)
        if self.rss_start is not None and self.rss_end is not None:
            self.rss_delta = self.rss_end - self.rss_start
        prefix = "  " * self.indent
        print(f"{prefix}[{_format_metrics(self.elapsed, self.rss_delta, self.rss_peak)}] {self.label}")
        return False


PALEO_REFERENCE_FRAME_CORRECTIONS = {
    # Maps a dataset base name (e.g. '100_ma') to an Euler rotation that converts
    # the input coordinates from the GPlates "OptimisedMantleRef" reference frame
    # used by the Zahirovic et al. 2022 rotation file to a paleomagnetic-aligned
    # frame where the equator passes through northern South America at 100 Ma
    # (Amazonas / Roraima / Pará / Amapá), matching standard paleogeographic atlases.
    #
    # Each entry is (pole_lat_deg, pole_lon_deg, angle_deg). The 'default' entry
    # is applied to every dataset that does not have its own override.
    # Calibrated empirically against the 100 Ma reference so that Brazilian basins
    # (Barreirinhas, Pará-Maranhão, Foz do Amazonas) sit on the new equator and
    # the South America Craton coastline straddles the equatorial zone.
    'default': (0.0, 60.0, -13.0),
}


# Shapefile → GeoJSON export replaced every non-ASCII character with '?'.
# Restore the original names (longest matches first) so basin labels, popups
# and the basin filter show Portuguese, Spanish and French accents.
_ACCENT_REPAIRS = {
    'Pernambuco-Para?ba/Pernambuco': 'Pernambuco-Paraíba/Pernambuco',
    'Pernambuco-Para?ba/Para?ba': 'Pernambuco-Paraíba/Paraíba',
    'Neuqu?n (south back-arc basin)': 'Neuquén (south back-arc basin)',
    'Bacia do Norte (=Paran? )': 'Bacia do Norte (=Paraná )',
    'Calcaires sup?rieurs de Berriche': 'Calcaires supérieurs de Berriche',
    'Marne ? gypse inf?rieures': 'Marne à gypse inférieures',
    'Precordillera de Copiap?': 'Precordillera de Copiapó',
    'Ba?ados de Caichig?e': 'Bañados de Caichigüe',
    'Ca?ad?n de la Zorra': 'Cañadón de la Zorra',
    'Ca?ad?n La Orientala': 'Cañadón La Orientala',
    'Lago Colhu? Huapi': 'Lago Colhué Huapi',
    'Ponta do Tubar?o Beds': 'Ponta do Tubarão Beds',
    'Pernambuco-Para?ba': 'Pernambuco-Paraíba',
    'S?o Lu?s-Graja?': 'São Luís-Grajaú',
    'Par?-Maranh?o': 'Pará-Maranhão',
    'Ca?ad?n Asfalto': 'Cañadón Asfalto',
    'Esp?rito Santo': 'Espírito Santo',
    'Bragan?a-Viseu': 'Bragança-Viseu',
    'A?n El Guettar': 'Aïn El Guettar',
    'A?n el Guettar': 'Aïn el Guettar',
    'Anfiteatro de Tic?': 'Anfiteatro de Ticó',
    'Barra de Iti?ba': 'Barra de Itiúba',
    'Ilhas/ Massacar?': 'Ilhas/ Massacará',
    'Pitanga- Carua?u': 'Pitanga- Caruaçu',
    'Salvador/Massacar?': 'Salvador/Massacará',
    'Santo Anast?cio': 'Santo Anastácio',
    'S?o Sebasti?o': 'São Sebastião',
    'Alter do Ch?o': 'Alter do Chão',
    'Ca?adon Matasiete': 'Cañadon Matasiete',
    'Cerro Casta?o': 'Cerro Castaño',
    'Cull?n Grande': 'Cullín Grande',
    'Jaguar?/Sernambi': 'Jaguaré/Sernambi',
    'Miss?o Velha': 'Missão Velha',
    'Pichi Neuqu?n': 'Pichi Neuquén',
    'Pic?n Leuf?': 'Picún Leufú',
    'Rio Tapirap?': 'Rio Tapirapé',
    'Cha?arcillo': 'Chañarcillo',
    'Florian?polis': 'Florianópolis',
    'S?o Lu?s': 'São Luís',
    'S?o Mateus': 'São Mateus',
    'S?o Carlos': 'São Carlos',
    'S?o Jos?': 'São José',
    'Sidi A?ch': 'Sidi Aïch',
    'Pe?as Altas': 'Peñas Altas',
    'Can?da Sol?s': 'Cañada Solís',
    'Can?da Solis': 'Cañada Solis',
    'Rhoundja?a': 'Rhoundjaïa',
    'Goitac?s?': 'Goitacás',
    'Alc?ntara': 'Alcântara',
    'Algod?es': 'Algodões',
    'Atl?ntida': 'Atlântida',
    'Barranqu?n': 'Barranquín',
    'Carm?polis': 'Carmópolis',
    'Celend?n': 'Celendín',
    'Embor?': 'Emboré',
    'Fahd?ne': 'Fahdène',
    'Fah?ne': 'Fahène',
    'Germ?nia': 'Germânia',
    'Guich?n': 'Guichón',
    'Huitr?n': 'Huitrín',
    'Igrapi?na': 'Igrapiúna',
    'Itamarac?': 'Itamaracá',
    'Itanha?m': 'Itanhaém',
    'Janda?ra': 'Jandaíra',
    'Lefip?n': 'Lefipán',
    'Mar?lia': 'Marília',
    'Massarac?': 'Massacará',
    'Massacar?': 'Massacará',
    'Neuqu?n': 'Neuquén',
    'Parna?ba': 'Parnaíba',
    'Pend?ncia': 'Pendência',
    'Pi?arras': 'Piçarras',
    'Po?o Verde': 'Poço Verde',
    'Pregui?as': 'Preguiças',
    'Quissam?': 'Quissamã',
    'Rec?ncavo': 'Recôncavo',
    'Reg?ncia': 'Regência',
    'Serinha?m': 'Serinhaém',
    'Solim?es': 'Solimões',
    'Tacuaremb?': 'Tacuarembó',
    'Tramanda?': 'Tramandaí',
    'Tr?s Barras': 'Três Barras',
    'Ara?atuba': 'Araçatuba',
    'Bragan?a': 'Bragança',
    'Burg?ita': 'Burgüita',
    'Cabi?nas': 'Cabiúnas',
    'Cassipor?': 'Cassiporé',
    'Graja?': 'Grajaú',
    'Guamar?': 'Guamaré',
    'Guaruj?': 'Guarujá',
    'Ilh?us': 'Ilhéus',
    'Ita?nas': 'Itaúnas',
    'Itai?nas': 'Itaúnas',
    'Itacar?': 'Itacaré',
    'Itapag?': 'Itapagí',
    'Ita?pe': 'Itaípe',
    'Jacu?pe': 'Jacuípe',
    'Jaguar?': 'Jaguaré',
    'Macei?': 'Maceió',
    'Maraj?': 'Marajó',
    'Mara?on': 'Marañón',
    'Munda?': 'Mundaú',
    'Pabell?n': 'Pabellón',
    'Paran?': 'Paraná',
    'Po??o': 'Poção',
    'Potos?': 'Potosí',
    'Quiric?': 'Quiricó',
    'Rio ?vila': 'Rio Ávila',
    'R?o Mayer': 'Río Mayer',
    'R?o Chico': 'Río Chico',
    'Tinhar?': 'Tinharé',
    'Trair?': 'Trairí',
    'Tut?ia': 'Tutóia',
    'Alian?a': 'Aliança',
    'Anaj?s': 'Anajás',
    'Ap?n': 'Apón',
    'Aur?s': 'Aurès',
    'Avil?': 'Avilé',
    'Azil?': 'Azilé',
    'Baquer?': 'Baqueró',
    'Caiu?': 'Caiuá',
    'Caju?': 'Cajuá',
    'Can?rias': 'Canárias',
    'Cear?': 'Ceará',
    'Chim?': 'Chimú',
    'Cod?': 'Codó',
    'Col?n': 'Colón',
    'Cricar?': 'Cricaré',
    'Garc?a': 'García',
    'Garga?': 'Gargaú',
    'Goio Er?': 'Goio Erê',
    'Guar?': 'Guará',
    'Huar?n': 'Huarón',
    'Imb?': 'Imbé',
    'Jag?el': 'Jagüel',
    'Jatob?': 'Jatobá',
    'Maca?': 'Macaé',
    'Malarg?e': 'Malargüe',
    'Mut?': 'Mutá',
    'Oy?n': 'Oyón',
    'Peri?': 'Periá',
    'Pia?abu?u': 'Piaçabuçu',
    'Pich?': 'Piché',
    'Rembou?': 'Remboué',
    'Rinc?n': 'Rincón',
    'Tau?': 'Tauá',
    'Tib?': 'Tibí',
    'Vi?ita': 'Viñita',
    'A?u': 'Açu',
    '?gua Grande': 'Água Grande',
    'Travossas?': 'Travessas',
    'Para?ba': 'Paraíba',
    'Maranh?o': 'Maranhão',
    'Esp?rito': 'Espírito',
    'Ca?ad?n': 'Cañadón',
    'Ca?adon': 'Cañadon',
    'Ch?o': 'Chão',
    'S?o': 'São',
    'Lu?s': 'Luís',
    'Jos?': 'José',
    'Par?': 'Pará',
    'R?o ': 'Río ',
}

_ACCENT_REPAIR_ITEMS = tuple(
    sorted(_ACCENT_REPAIRS.items(), key=lambda item: len(item[0]), reverse=True)
)


def restore_lost_accents(text):
    """Put back accents that the GeoJSON export stored as '?'."""
    if not isinstance(text, str) or '?' not in text:
        return text
    repaired = text
    for broken, fixed in _ACCENT_REPAIR_ITEMS:
        if broken in repaired:
            repaired = repaired.replace(broken, fixed)
    return repaired


def restore_geojson_accents(data):
    """Repair accented text on every string property of a GeoJSON object."""
    for feature in data.get('features', []):
        props = feature.get('properties')
        if not isinstance(props, dict):
            continue
        for key, value in props.items():
            if isinstance(value, str) and '?' in value:
                props[key] = restore_lost_accents(value)
    return data


def load_geojson(filepath):
    """Load a GeoJSON file and restore lost accents on text fields."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return restore_geojson_accents(data)


def apply_euler_rotation(lon, lat, pole_lat_deg, pole_lon_deg, angle_deg):
    """Rotate a single (lon, lat) coordinate by an Euler rotation.

    Uses Rodrigues' rotation formula on the unit sphere. The Euler pole is
    given as (pole_lat_deg, pole_lon_deg) and the rotation magnitude as
    angle_deg (positive = counter-clockwise looking from outside the pole).
    Returns a (lon, lat) tuple in degrees.
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    pole_lat_rad = np.radians(pole_lat_deg)
    pole_lon_rad = np.radians(pole_lon_deg)
    angle_rad = np.radians(angle_deg)

    p = np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ])
    k = np.array([
        np.cos(pole_lat_rad) * np.cos(pole_lon_rad),
        np.cos(pole_lat_rad) * np.sin(pole_lon_rad),
        np.sin(pole_lat_rad),
    ])

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    p_rot = (
        p * cos_a
        + np.cross(k, p) * sin_a
        + k * np.dot(k, p) * (1.0 - cos_a)
    )

    new_lat = float(np.degrees(np.arcsin(np.clip(p_rot[2], -1.0, 1.0))))
    new_lon = float(np.degrees(np.arctan2(p_rot[1], p_rot[0])))
    return new_lon, new_lat


def _rotate_coord(coord, pole_lat, pole_lon, angle):
    new_lon, new_lat = apply_euler_rotation(coord[0], coord[1], pole_lat, pole_lon, angle)
    return [new_lon, new_lat] + list(coord[2:])


def _rotate_coord_list(coords, pole_lat, pole_lon, angle):
    return [_rotate_coord(c, pole_lat, pole_lon, angle) for c in coords]


def _split_line_at_antimeridian(coords):
    """Split a rotated coordinate list wherever it crosses the antimeridian.

    Rotation wraps longitudes back into [-180, 180], so a line that ends up
    straddling ±180° contains a segment that jumps ~360° in longitude. Leaflet
    would draw that segment the long way round, as a horizontal streak across
    the whole map. Each crossing is cut at ±180°, interpolating the latitude of
    the crossing so both parts still reach the edge of the map.
    """
    if len(coords) < 2:
        return [list(coords)] if coords else []

    parts = []
    current = [coords[0]]
    for previous, point in zip(coords, coords[1:]):
        delta = point[0] - previous[0]
        if abs(delta) > 180:
            # Longitude of the edge this segment leaves through (its mirror is
            # where the continuation re-enters on the other side of the map).
            exit_lon = 180.0 if delta < 0 else -180.0
            unwrapped_lon = point[0] + (360.0 if delta < 0 else -360.0)
            span = unwrapped_lon - previous[0]
            t = (exit_lon - previous[0]) / span if span else 0.0
            lat_crossing = previous[1] + t * (point[1] - previous[1])
            current.append([exit_lon, lat_crossing])
            parts.append(current)
            current = [[-exit_lon, lat_crossing], point]
        else:
            current.append(point)
    if len(current) > 1:
        parts.append(current)
    return parts


def _rotate_geometry(geom, pole_lat, pole_lon, angle):
    """Rotate every coordinate inside a GeoJSON geometry in place.

    Supports Point, MultiPoint, LineString, MultiLineString, Polygon and
    MultiPolygon. Other types are left untouched. Line geometries are also
    split at the antimeridian, since rotation can push them across ±180°.
    """
    gtype = geom.get('type')
    coords = geom.get('coordinates')
    if coords is None:
        return geom

    if gtype == 'Point':
        geom['coordinates'] = _rotate_coord(coords, pole_lat, pole_lon, angle)
    elif gtype == 'MultiPoint':
        geom['coordinates'] = _rotate_coord_list(coords, pole_lat, pole_lon, angle)
    elif gtype == 'LineString':
        rotated = _rotate_coord_list(coords, pole_lat, pole_lon, angle)
        if len(rotated) < 2:
            geom['coordinates'] = rotated
        else:
            parts = _split_line_at_antimeridian(rotated)
            if len(parts) == 1:
                geom['coordinates'] = parts[0]
            else:
                # Also covers the degenerate case of no drawable part left.
                geom['type'] = 'MultiLineString'
                geom['coordinates'] = parts
    elif gtype == 'MultiLineString':
        split_lines = []
        for line in coords:
            rotated = _rotate_coord_list(line, pole_lat, pole_lon, angle)
            if len(rotated) < 2:
                split_lines.append(rotated)
            else:
                split_lines.extend(_split_line_at_antimeridian(rotated))
        geom['coordinates'] = split_lines
    elif gtype == 'Polygon':
        geom['coordinates'] = [
            _rotate_coord_list(line, pole_lat, pole_lon, angle) for line in coords
        ]
    elif gtype == 'MultiPolygon':
        geom['coordinates'] = [
            [_rotate_coord_list(ring, pole_lat, pole_lon, angle) for ring in poly]
            for poly in coords
        ]
    return geom


def apply_paleo_reference_frame_correction(geojson_data, base_name):
    """Apply the configured Euler rotation to all geometries in a GeoJSON.

    Looks up the rotation for ``base_name`` in PALEO_REFERENCE_FRAME_CORRECTIONS,
    falling back to the 'default' entry. Returns a new GeoJSON dict with the
    rotated coordinates; the input is left unmodified. If no correction is
    configured (or the configured angle is 0), the input is returned as-is.
    """
    correction = PALEO_REFERENCE_FRAME_CORRECTIONS.get(
        base_name,
        PALEO_REFERENCE_FRAME_CORRECTIONS.get('default'),
    )
    if not correction:
        return geojson_data
    pole_lat, pole_lon, angle = correction
    if angle == 0:
        return geojson_data

    rotated = json.loads(json.dumps(geojson_data))
    for feature in rotated.get('features', []):
        geom = feature.get('geometry')
        if geom:
            _rotate_geometry(geom, pole_lat, pole_lon, angle)

    print(
        f"Applied paleo reference frame correction for '{base_name}': "
        f"pole=({pole_lat}°, {pole_lon}°), angle={angle}° "
        f"({len(rotated.get('features', []))} features)"
    )
    return rotated

def get_geojson_bounds(geojson_data):
    """Calculate bounds from GeoJSON features."""
    all_coords = []
    for feature in geojson_data.get('features', []):
        geom = feature.get('geometry', {})
        if geom.get('type') == 'Point':
            coords = geom.get('coordinates', [])
            if coords:
                all_coords.append(coords)
        elif geom.get('type') in ['LineString', 'MultiLineString']:
            coords = geom.get('coordinates', [])
            if not coords or not coords[0]:
                continue
            if isinstance(coords[0][0], list):  # MultiLineString
                for line in coords:
                    all_coords.extend(line)
            else:  # LineString
                all_coords.extend(coords)
    
    if not all_coords:
        return None
    
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]

def get_geotiff_bounds(geotiff_path):
    """Get bounds from GeoTIFF file."""
    with rasterio.open(geotiff_path) as src:
        # Transform bounds to WGS84 if needed
        bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
        return [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

def climate_to_numeric(climate):
    """Convert climate classification to numeric value for interpolation."""
    climate_map = {
        'H': 3.0,  # Humid
        'S': 2.0,  # Semi-arid
        'D': 1.0   # Dry
    }
    return climate_map.get(climate, 2.0)

def get_climate_class(props, default=None):
    """Read the climate classification ('H', 'S' or 'D') from feature properties.

    Shapefile-to-GeoJSON exports of the source data have shipped this field as
    both 'Climate_Cl' and 'Climate_cl', so the property name is matched
    case-insensitively. Returns ``default`` when no climate field is present.
    """
    if not props:
        return default
    for key, value in props.items():
        if key.lower() == 'climate_cl':
            if value is None:
                return default
            value = str(value).strip().upper()
            return value or default
    return default

# Stable pie-slice order for multi-climate markers (Humid, Semi-arid, Dry).
_CLIMATE_DISPLAY_ORDER = ('H', 'S', 'D')

# Fill and stroke shared by solid CircleMarkers and split (multi-climate)
# icons. The thin dark edge keeps a point crisp; the white halo around it
# (``path.pcvs-point`` in the theme) is what separates the climate colour from
# a raster painted in that very same colour.
MARKER_FILL_OPACITY = 1.0
MARKER_STROKE_COLOR = '#16253a'
MARKER_HALO = ('drop-shadow(0 0 1.3px rgba(255, 255, 255, 0.95)) '
               'drop-shadow(0 0.5px 1px rgba(15, 23, 42, 0.3))')

# Pixel geometry of every data point (CircleMarker and split SVG icons).
# Leaflet CircleMarker radius is in CSS pixels and does not change with zoom;
# PDF export prints the same markers, so HTML and PDF stay in lockstep.
POINT_RADIUS_PX = 3.5
POINT_WEIGHT_PX = 1.15
POINT_OUTER_PX = 2 * POINT_RADIUS_PX + POINT_WEIGHT_PX


def _climate_name(code):
    """Spell out a climate code for popups, e.g. 'H' -> 'Humid (H)'."""
    if not code or code == 'N/A':
        return 'N/A'
    label = CLIMATE_LABELS.get(code)
    return f'{label} ({code})' if label else code


def resolve_marker_climates(climates):
    """Pick climate class(es) that win by prevalence at a coincident location.

    Only the highest count matters. A single clear winner yields one color; a
    two-way or three-way tie yields two or three colors for a split marker.
    Less-prevalent climates are ignored.
    """
    counts = Counter(c for c in climates if c in _CLIMATE_DISPLAY_ORDER)
    if not counts:
        return []
    max_count = max(counts.values())
    return [c for c in _CLIMATE_DISPLAY_ORDER if counts.get(c) == max_count]


def climate_marker_icon_html(climates_tied, color_map, radius_px=POINT_RADIUS_PX,
                             weight_px=POINT_WEIGHT_PX,
                             opacity=MARKER_FILL_OPACITY):
    """Build a multi-color DivIcon SVG matching Folium CircleMarker geometry.

    Used only for 2/3-way climate ties. Size/stroke match `CircleMarker` with
    the same `radius_px` and `weight_px`. Returns `(html, outer_size_px)`.
    """
    import math

    n = len(climates_tied)
    if n < 2:
        raise ValueError('climate_marker_icon_html is for multi-color ties only')

    # Leaflet draws stroke centered on the path, so outer size is 2*r + weight.
    outer = 2 * radius_px + weight_px
    cx = cy = outer / 2.0
    r = float(radius_px)

    def _pt(deg):
        rad = math.radians(deg)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    slices = []
    slice_deg = 360.0 / n
    for i, climate in enumerate(climates_tied):
        # Start at top (-90°) so slices match the previous conic-gradient layout.
        a0 = -90.0 + i * slice_deg
        a1 = -90.0 + (i + 1) * slice_deg
        x0, y0 = _pt(a0)
        x1, y1 = _pt(a1)
        large = 1 if slice_deg > 180 else 0
        color = color_map.get(climate, '#94a3b8')
        slices.append(
            f'<path d="M {cx:.2f},{cy:.2f} L {x0:.2f},{y0:.2f} '
            f'A {r:.2f},{r:.2f} 0 {large} 1 {x1:.2f},{y1:.2f} Z" '
            f'fill="{color}" fill-opacity="{opacity}"/>'
        )
    stroke = (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" '
        f'fill="none" stroke="{MARKER_STROKE_COLOR}" stroke-width="{weight_px}"/>'
    )
    html = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{outer}" height="{outer}" '
        f'viewBox="0 0 {outer} {outer}" style="display:block;'
        f'filter:{MARKER_HALO};">'
        + ''.join(slices) + stroke + '</svg>'
    )
    return html, outer

def extract_points_and_values(points_data):
    """Extract point coordinates and numeric values from GeoJSON."""
    points = []
    values = []
    for feature in points_data.get('features', []):
        geom = feature.get('geometry', {})
        props = feature.get('properties', {})
        if geom.get('type') == 'Point':
            coords = geom.get('coordinates', [])
            if coords:
                points.append([coords[0], coords[1]])  # [lon, lat]
                climate = get_climate_class(props, 'S')
                values.append(climate_to_numeric(climate))
    return np.array(points), np.array(values)

def knn_smooth_values(points, values, k=8, power=2.0, exclude_self=True,
                      method=METHOD_BRUTE):
    """
    Apply KNN regression to smooth point values using spatial neighbors.

    Parameters:
    -----------
    method : str
        Neighbor search backend: METHOD_BRUTE (full distance matrix) or
        METHOD_KDTREE (scipy cKDTree queries).
    """
    if len(points) == 0:
        raise ValueError("No points available for KNN")
    if len(points) == 1:
        return values.copy()

    k = min(k, len(points) - 1 if exclude_self else len(points))
    if k <= 0:
        return values.copy()

    if method == METHOD_KDTREE:
        return _knn_smooth_values_kdtree(points, values, k, power, exclude_self)

    distances = cdist(points, points)
    if exclude_self:
        np.fill_diagonal(distances, np.inf)

    smoothed = np.zeros_like(values, dtype=np.float64)
    for i in range(len(points)):
        # Stable sort so coincident points are always picked in index order,
        # keeping the result identical to the k-d tree backend
        nearest_idx = np.argsort(distances[i], kind='stable')[:k]
        nearest_dist = distances[i][nearest_idx].astype(np.float64)
        nearest_dist[nearest_dist == 0] = 1e-10
        weights = 1.0 / (nearest_dist ** power)
        weights = weights / np.sum(weights)
        smoothed[i] = np.sum(weights * values[nearest_idx])

    return smoothed


def _coincident_margin(points):
    """How many extra neighbors a k-d tree query needs to cover distance ties.

    Datasets contain several points sharing the exact same coordinates, so the
    neighbor at position k is often tied with others. Querying this many extra
    neighbors guarantees the whole tied group comes back, which is what makes
    the index-based tie-break below reproduce the brute-force selection.
    """
    _, counts = np.unique(points, axis=0, return_counts=True)
    return int(counts.max()) - 1


def _nearest_neighbors_kdtree(tree, query_points, k, margin, n_points):
    """Query the k nearest neighbors, ordered by (distance, point index)."""
    k_query = min(k + margin, n_points)
    distances, indices = tree.query(query_points, k=k_query)
    distances = distances.reshape(len(query_points), k_query).astype(np.float64)
    indices = indices.reshape(len(query_points), k_query)

    # cKDTree sorts by distance only, leaving coincident points in arbitrary
    # order; re-sorting by (distance, index) matches the brute-force backend.
    order = np.lexsort((indices, distances), axis=1)
    rows = np.arange(len(query_points))[:, None]
    return distances[rows, order], indices[rows, order]


def _knn_smooth_values_kdtree(points, values, k, power, exclude_self):
    """KNN smoothing using a k-d tree for the neighbor queries."""
    n_points = len(points)
    tree = cKDTree(points)
    margin = _coincident_margin(points)

    k_wanted = k + 1 if exclude_self else k
    distances, indices = _nearest_neighbors_kdtree(tree, points, k_wanted, margin, n_points)

    if exclude_self:
        # Discard the query point itself. With coincident points the query may
        # not return its own index, so those rows drop their farthest neighbor
        # instead to keep the row width uniform.
        drop = indices == np.arange(n_points)[:, None]
        missing_self = ~drop.any(axis=1)
        if np.any(missing_self):
            drop[missing_self, -1] = True
        keep = ~drop
        n_kept = distances.shape[1] - 1
        distances = distances[keep].reshape(n_points, n_kept)
        indices = indices[keep].reshape(n_points, n_kept)

    distances = distances[:, :k]
    indices = indices[:, :k]

    distances = np.where(distances == 0, 1e-10, distances)
    weights = 1.0 / (distances ** power)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return np.sum(weights * values[indices], axis=1)

def idw_interpolation(points, values, grid_lons, grid_lats, power=2, n_neighbors=12, 
                      preserve_points=True, point_radius=0.15, method=METHOD_BRUTE):
    """
    Perform Inverse Distance Weighting (IDW) interpolation using N nearest neighbors.
    
    This implementation mimics ArcGIS's IDW with "VARIABLE N" search radius,
    which uses only the N nearest points for each grid cell interpolation.
    
    Parameters:
    -----------
    points : array-like, shape (n_points, 2)
        Point coordinates [lon, lat]
    values : array-like, shape (n_points,)
        Values at points
    grid_lons : array-like, shape (n_lons,)
        Longitude grid
    grid_lats : array-like, shape (n_lats,)
        Latitude grid
    power : float
        Power parameter for IDW (default: 2)
    n_neighbors : int
        Number of nearest neighbors to use for interpolation (default: 12)
        This matches ArcGIS's "VARIABLE 12" setting from original raster.
    preserve_points : bool
        If True, grid cells very close to data points will use the exact point value
        instead of interpolated value (default: True)
    point_radius : float
        Distance threshold (in degrees) for preserving point values (default: 0.15)
    method : str
        Neighbor search backend: METHOD_BRUTE (full distance matrix) or
        METHOD_KDTREE (scipy cKDTree queries).
    
    Returns:
    --------
    grid_values : array, shape (n_lats, n_lons)
        Interpolated values on grid
    """
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    
    # Flatten grid
    grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    n_grid_points = len(grid_points)
    
    # Limit to n_neighbors nearest points
    n_neighbors = min(n_neighbors, len(points))
    
    if method == METHOD_KDTREE:
        grid_values = _idw_interpolation_kdtree(
            points, values, grid_points, power, n_neighbors,
            preserve_points, point_radius
        )
        return grid_values.reshape(lon_grid.shape)
    
    # Calculate distances from all grid points to all data points
    distances = cdist(grid_points, points)
    
    # Initialize output array
    grid_values = np.zeros(n_grid_points)
    
    # For each grid point, find the N nearest neighbors and interpolate
    for i in range(n_grid_points):
        # Get distances to all data points for this grid cell
        dist_i = distances[i, :]
        
        # Find the nearest point
        min_dist_idx = np.argmin(dist_i)
        min_dist = dist_i[min_dist_idx]
        
        # If very close to a data point and preserve_points is True, use that value directly
        if preserve_points and min_dist < point_radius:
            grid_values[i] = values[min_dist_idx]
        else:
            # Find indices of N nearest neighbors (stable: ties by point index)
            nearest_idx = np.argsort(dist_i, kind='stable')[:n_neighbors]
            nearest_dist = dist_i[nearest_idx]
            nearest_values = values[nearest_idx]
            
            # Avoid division by zero (point exactly on data point)
            nearest_dist[nearest_dist == 0] = 1e-10
            
            # Calculate weights: w = 1 / d^power
            weights = 1.0 / (nearest_dist ** power)
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Weighted average
            grid_values[i] = np.sum(weights * nearest_values)
    
    # Reshape to grid
    grid_values = grid_values.reshape(lon_grid.shape)
    
    return grid_values


def _idw_interpolation_kdtree(points, values, grid_points, power, n_neighbors,
                              preserve_points, point_radius):
    """IDW interpolation using a k-d tree for the neighbor queries.

    Returns a flat array of interpolated values, one per grid point.
    """
    tree = cKDTree(points)
    margin = _coincident_margin(points)
    distances, indices = _nearest_neighbors_kdtree(
        tree, grid_points, n_neighbors, margin, len(points)
    )
    distances = distances[:, :n_neighbors]
    indices = indices[:, :n_neighbors]

    safe_distances = np.where(distances == 0, 1e-10, distances)
    weights = 1.0 / (safe_distances ** power)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    grid_values = np.sum(weights * values[indices], axis=1)

    if preserve_points:
        near = distances[:, 0] < point_radius
        grid_values[near] = values[indices[near, 0]]

    return grid_values

def create_idw_raster(points_data, output_path, resolution=0.1, power=2, n_neighbors=12,
                      preserve_points=True, point_radius=0.15, points=None, values=None,
                      method=METHOD_BRUTE):
    """
    Create a GeoTIFF raster from IDW interpolation of point data.
    
    Parameters:
    -----------
    points_data : dict
        GeoJSON data with point features
    points : array-like, optional
        Point coordinates [lon, lat] to use instead of extracting from GeoJSON
    values : array-like, optional
        Numeric values for each point (e.g., after KNN smoothing)
    output_path : str
        Path to save GeoTIFF
    resolution : float
        Grid resolution in degrees (default: 0.1)
    power : float
        IDW power parameter (default: 2)
    n_neighbors : int
        Number of nearest neighbors for IDW (default: 12, matches ArcGIS settings)
    preserve_points : bool
        If True, preserve exact values at data point locations (default: True)
    point_radius : float
        Distance threshold for preserving point values (default: 0.15 degrees)
    method : str
        Neighbor search backend: METHOD_BRUTE or METHOD_KDTREE
    """
    # Extract points and values
    if points is None or values is None:
        points, values = extract_points_and_values(points_data)

    if len(points) == 0:
        raise ValueError("No point data found")
    
    points = np.array(points)
    values = np.array(values)
    
    # Calculate bounds with padding
    min_lon, max_lon = points[:, 0].min() - 1, points[:, 0].max() + 1
    min_lat, max_lat = points[:, 1].min() - 1, points[:, 1].max() + 1
    
    # Create grid
    grid_lons = np.arange(min_lon, max_lon + resolution, resolution)
    grid_lats = np.arange(min_lat, max_lat + resolution, resolution)
    
    print(f"Creating IDW raster with resolution {resolution}°")
    print(f"Grid size: {len(grid_lats)} x {len(grid_lons)}")
    print(f"Bounds: [{min_lat:.2f}, {min_lon:.2f}] to [{max_lat:.2f}, {max_lon:.2f}]")
    print(f"Using {n_neighbors} nearest neighbors for interpolation")
    print(f"Preserve point values: {preserve_points} (radius: {point_radius}°)")
    print(f"Neighbor search: {method_label(method)}")
    
    # Perform IDW interpolation using N nearest neighbors
    grid_values = idw_interpolation(points, values, grid_lons, grid_lats, 
                                    power=power, n_neighbors=n_neighbors,
                                    preserve_points=preserve_points, 
                                    point_radius=point_radius,
                                    method=method)
    
    # Create GeoTIFF
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, 
                           len(grid_lons), len(grid_lats))
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=len(grid_lats),
        width=len(grid_lons),
        count=1,
        dtype=grid_values.dtype,
        crs='EPSG:4326',
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(grid_values, 1)
    
    print(f"IDW raster saved to: {output_path}")
    return output_path

def create_raster_overlay(geotiff_path, map_obj, raster_img_path='raster_overlay.png', layer_name='Raster',
                          points_data=None, preserve_points=True, point_radius=0.15,
                          point_values_override=None, gradient_sharp=2.5):
    """Create a raster overlay from GeoTIFF using ImageOverlay.
    
    Parameters:
    -----------
    geotiff_path : str
        Path to the GeoTIFF file
    map_obj : folium.Map
        Folium map object to add the overlay to
    raster_img_path : str
        Path to save the PNG image
    layer_name : str
        Name for the layer in the layer control
    points_data : dict, optional
        GeoJSON data with point features (for point value preservation)
    preserve_points : bool
        If True, override raster values near point locations with exact point values
    point_radius : float
        Distance threshold in degrees for point preservation
    point_values_override : array-like, optional
        Override values at point locations (e.g. after KNN smoothing)
    gradient_sharp : float
        Factor for sharper color transitions (higher = more abrupt). Applied as
        normalized = clip((normalized - 0.5) * gradient_sharp + 0.5, 0, 1).
    """
    try:
        print(f"Opening GeoTIFF: {geotiff_path}")
        with rasterio.open(geotiff_path) as src:
            print(f"GeoTIFF CRS: {src.crs}")
            print(f"GeoTIFF bounds (native): {src.bounds}")
            
            # Read the raster data
            data = src.read(1)  # Read first band
            raster_transform = src.transform
            print(f"Raster shape: {data.shape}")
            print(f"Data range: {np.nanmin(data)} to {np.nanmax(data)}")
            
            # If points_data is provided and preserve_points is True, override values at point locations
            # Strategy: Each point colors its EXACT pixel + a small surrounding area
            # When points conflict, we use the MAXIMUM value (preferring Humid over Dry)
            if points_data is not None and preserve_points:
                bounds = src.bounds
                pixel_size_x = (bounds.right - bounds.left) / data.shape[1]
                pixel_size_y = (bounds.top - bounds.bottom) / data.shape[0]
                
                # Track which pixels have been set by points and their values
                # We'll use maximum value when multiple points affect same pixel
                point_values_matrix = np.full(data.shape, np.nan)
                
                points_overridden = 0
                override_index = 0
                override_count = len(point_values_override) if point_values_override is not None else 0
                for feature in points_data.get('features', []):
                    geom = feature.get('geometry', {})
                    props = feature.get('properties', {})
                    
                    if geom.get('type') == 'Point':
                        coords = geom.get('coordinates', [])
                        if coords:
                            lon, lat = coords[0], coords[1]
                            if point_values_override is not None and override_index < override_count:
                                point_value = float(point_values_override[override_index])
                            else:
                                climate = get_climate_class(props, 'S')
                                point_value = climate_to_numeric(climate)
                            override_index += 1
                            
                            # Check if within raster bounds
                            if bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top:
                                # Get pixel coordinates for this point
                                col, row = ~raster_transform * (lon, lat)
                                col, row = int(col), int(row)
                                
                                if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                                    # Set the exact pixel for this point
                                    # Use max to prefer Humid (3) over Semi-arid (2) over Dry (1)
                                    if np.isnan(point_values_matrix[row, col]) or point_value > point_values_matrix[row, col]:
                                        point_values_matrix[row, col] = point_value
                                    
                                    # Also set a small surrounding area (1 pixel radius) with same logic
                                    for dr in range(-1, 2):
                                        for dc in range(-1, 2):
                                            r, c = row + dr, col + dc
                                            if 0 <= r < data.shape[0] and 0 <= c < data.shape[1]:
                                                if np.isnan(point_values_matrix[r, c]) or point_value > point_values_matrix[r, c]:
                                                    point_values_matrix[r, c] = point_value
                
                # Apply point values to data
                mask = ~np.isnan(point_values_matrix)
                data[mask] = point_values_matrix[mask]
                points_overridden = np.sum(mask)
                
                print(f"Point value preservation: {points_overridden} pixels overridden")
            
            # Get bounds in WGS84
            bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            print(f"GeoTIFF bounds (WGS84): {bounds}")
            # bounds format: (minx, miny, maxx, maxy) -> (west, south, east, north)
            # ImageOverlay needs: [[south, west], [north, east]]
            image_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
            print(f"ImageOverlay bounds: {image_bounds}")
            
            # Handle NaN values - create mask for valid data
            valid_mask = ~np.isnan(data)
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            
            print(f"Data range (cleaned): {data_min} to {data_max}")
            print(f"Valid pixels: {np.sum(valid_mask)} / {data.size}")

            rgb_uint8 = climate_values_to_rgb(data, valid_mask, gradient_sharp)
            # Pixels without data stay fully transparent instead of turning black
            alpha = np.where(valid_mask, 255.0, 0.0).astype(np.float32)
            rgba = np.dstack([rgb_uint8, _feather_edges(alpha)])

            channel_ranges = ', '.join(
                f'{name}: [{rgb_uint8[..., i][valid_mask].min()}, '
                f'{rgb_uint8[..., i][valid_mask].max()}]'
                for i, name in enumerate('RGB')
            )
            print(f"Color range - {channel_ranges}")

            # Flip image vertically because raster origin is top-left but geographic is bottom-left
            rgba = np.flipud(rgba)

            img_colored = Image.fromarray(rgba, mode='RGBA')
            
            # Save to a permanent location
            img_colored.save(raster_img_path)
            print(f"Raster image saved to: {raster_img_path}")
            
            # Add ImageOverlay to map
            # pixelated=False: the default renders the 0.1° grid nearest-neighbour,
            # which turns every class boundary into a staircase of blocks.
            image_overlay = ImageOverlay(
                image=raster_img_path,
                bounds=image_bounds,
                opacity=RASTER_OPACITY,
                name=layer_name,
                interactive=True,
                cross_origin=False,
                pixelated=False,
                zindex=1
            )
            image_overlay.add_to(map_obj)
            print("ImageOverlay added to map successfully")
            
            return raster_img_path
    except Exception as e:
        import traceback
        print(f"Error loading GeoTIFF: {e}")
        print(traceback.format_exc())
        return None

def classificar_por_intervalos(caminho_imagem):
    """Classifica pixels da imagem em amarelo/verde/azul e retorna contagens e percentuais."""
    img = cv2.imread(caminho_imagem)
    if img is None:
        print(f"Erro: Imagem não encontrada: {caminho_imagem}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Amarelo (H: 20-34)
    mask_yellow = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([34, 255, 255]))

    # Verde (H: 35-89)
    mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([89, 255, 255]))

    # Azul (H: 90-130)
    mask_blue = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))

    total_pixels = img.shape[0] * img.shape[1]

    count_yellow = cv2.countNonZero(mask_yellow)
    count_green = cv2.countNonZero(mask_green)
    count_blue = cv2.countNonZero(mask_blue)

    total_classificado = count_yellow + count_green + count_blue
    nao_classificado = total_pixels - total_classificado

    def pct(x): return (x / total_pixels) * 100 if total_pixels else 0.0

    return {
        "total_pixels": total_pixels,
        "count_yellow": count_yellow,
        "count_green": count_green,
        "count_blue": count_blue,
        "total_classificado": total_classificado,
        "nao_classificado": nao_classificado,
        "pct_yellow": pct(count_yellow),
        "pct_green": pct(count_green),
        "pct_blue": pct(count_blue),
        "pct_classificado": pct(total_classificado),
        "pct_nao_classificado": pct(nao_classificado),
    }


def _add_graticule(map_obj, interval=30):
    """Add a lat/lon graticule with labels that reposition at viewport edges on pan/zoom.

    The graticule is always visible and not toggleable via LayerControl.
    Grid lines use a custom pane (z-index 250) so they render behind data overlays.
    """
    graticule_css = MacroElement()
    graticule_css._template = Template("""
        {% macro header(this, kwargs) %}
        <style>
            .graticule-label {
                background: none !important;
                border: none !important;
                box-shadow: none !important;
            }
            .graticule-label span {
                font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI",
                             Roboto, "Helvetica Neue", Arial, sans-serif;
                font-size: 9.5px;
                font-weight: 600;
                letter-spacing: 0.04em;
                color: #55647a;
                text-shadow:
                    0 0 3px rgba(255, 255, 255, 0.95),
                    1px 0 0 rgba(255, 255, 255, 0.85), -1px 0 0 rgba(255, 255, 255, 0.85),
                    0 1px 0 rgba(255, 255, 255, 0.85), 0 -1px 0 rgba(255, 255, 255, 0.85);
                white-space: nowrap;
            }
        </style>
        {% endmacro %}
    """)
    map_obj.get_root().add_child(graticule_css)

    graticule_js = f"""
        {{% macro script(this, kwargs) %}}
        (function() {{
            var map = {{{{ this._parent.get_name() }}}};
            var interval = {interval};

            var pane = map.createPane('graticule');
            pane.style.zIndex = 250;
            pane.style.pointerEvents = 'none';

            var labelPane = map.createPane('graticuleLabels');
            labelPane.style.zIndex = 255;
            labelPane.style.pointerEvents = 'none';

            var lineStyle = {{
                color: '{GRATICULE_COLOR}',
                weight: 0.6,
                opacity: 0.55,
                dashArray: '1 5',
                lineCap: 'round',
                interactive: false,
                pane: 'graticule'
            }};
            var majorLineStyle = {{
                color: '{GRATICULE_COLOR}',
                weight: 0.9,
                opacity: 0.7,
                dashArray: null,
                interactive: false,
                pane: 'graticule'
            }};

            for (var lat = -90; lat <= 90; lat += interval) {{
                var style = (lat === 0) ? majorLineStyle : lineStyle;
                L.polyline([[lat, -180], [lat, 180]], style).addTo(map);
            }}
            for (var lon = -180; lon < 180; lon += interval) {{
                var style = (lon === 0) ? majorLineStyle : lineStyle;
                L.polyline([[-90, lon], [90, lon]], style).addTo(map);
            }}

            var labelsGroup = L.layerGroup().addTo(map);

            function fmtLat(lat) {{
                if (lat === 0) return '0°';
                return Math.abs(lat) + '°' + (lat > 0 ? 'N' : 'S');
            }}
            function fmtLon(lon) {{
                if (lon === 0) return '0°';
                if (Math.abs(lon) === 180) return '180°';
                return Math.abs(lon) + '°' + (lon > 0 ? 'E' : 'W');
            }}

            function updateLabels() {{
                labelsGroup.clearLayers();
                var b = map.getBounds();
                var west = b.getWest(), east = b.getEast();
                var south = b.getSouth(), north = b.getNorth();
                var mx = (east - west) * 0.01;
                var my = (north - south) * 0.03;

                for (var lat = -90; lat <= 90; lat += interval) {{
                    if (lat > south && lat < north) {{
                        L.marker([lat, west + mx], {{
                            icon: L.divIcon({{
                                className: 'graticule-label',
                                html: '<span>' + fmtLat(lat) + '</span>',
                                iconSize: [35, 14],
                                iconAnchor: [0, 7]
                            }}),
                            interactive: false,
                            pane: 'graticuleLabels'
                        }}).addTo(labelsGroup);
                    }}
                }}

                for (var lon = -180; lon < 180; lon += interval) {{
                    if (lon > west && lon < east) {{
                        L.marker([south + my, lon], {{
                            icon: L.divIcon({{
                                className: 'graticule-label',
                                html: '<span>' + fmtLon(lon) + '</span>',
                                iconSize: [35, 14],
                                iconAnchor: [15, -2]
                            }}),
                            interactive: false,
                            pane: 'graticuleLabels'
                        }}).addTo(labelsGroup);
                    }}
                }}
            }}

            map.on('moveend', updateLabels);
            map.whenReady(function() {{ setTimeout(updateLabels, 300); }});
            updateLabels();
        }})();
        {{% endmacro %}}
    """
    macro = MacroElement()
    macro._template = Template(graticule_js)
    map_obj.add_child(macro)


# ---------------------------------------------------------------------------
# Map look and feel
# ---------------------------------------------------------------------------

# Every floating panel on the map (layer control, legend, basin filter, colour
# stats, title card) shares these tokens so the map reads as one interface.
MAP_THEME_CSS = """
    {% macro header(this, kwargs) %}
    <style>
        .leaflet-container {
            --pcvs-ink: #0f172a;
            --pcvs-muted: #64748b;
            --pcvs-line: rgba(15, 23, 42, 0.10);
            --pcvs-surface: rgba(255, 255, 255, 0.94);
            --pcvs-shadow: 0 1px 2px rgba(15, 23, 42, 0.06),
                           0 10px 26px -12px rgba(15, 23, 42, 0.35);
            --pcvs-radius: 10px;
            font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI",
                         Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 12px;
            color: var(--pcvs-ink);
            /* Flat fill plus a light wash from above and a soft edge below, so
               the canvas reads as paper rather than a slab of grey. */
            background:
                radial-gradient(1200px 820px at 50% -12%,
                                rgba(255, 255, 255, 0.9),
                                rgba(255, 255, 255, 0) 62%),
                radial-gradient(1500px 1100px at 50% 118%,
                                rgba(15, 23, 42, 0.07),
                                rgba(15, 23, 42, 0) 60%),
                __BACKGROUND__;
            -webkit-font-smoothing: antialiased;
            overflow: hidden;
        }
        html, body { overflow: hidden; }

        .pcvs-panel {
            background: var(--pcvs-surface);
            -webkit-backdrop-filter: saturate(160%) blur(10px);
            backdrop-filter: saturate(160%) blur(10px);
            border: 1px solid var(--pcvs-line);
            border-radius: var(--pcvs-radius);
            box-shadow: var(--pcvs-shadow);
            color: var(--pcvs-ink);
        }
        .pcvs-panel-title {
            font-size: 9.5px;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: var(--pcvs-muted);
        }

        /* Zoom / fullscreen / measure buttons */
        .leaflet-bar,
        .leaflet-control-layers,
        .leaflet-control-measure {
            border: 1px solid var(--pcvs-line) !important;
            border-radius: var(--pcvs-radius) !important;
            box-shadow: var(--pcvs-shadow) !important;
            background: var(--pcvs-surface) !important;
            -webkit-backdrop-filter: saturate(160%) blur(10px);
            backdrop-filter: saturate(160%) blur(10px);
            overflow: hidden;
        }
        /* Only the background colour is themed: the shorthand would wipe out
           the sprite the fullscreen plugin draws its icon with. Sizes stay at
           26px for the same reason. */
        .leaflet-bar a,
        .leaflet-bar a:hover {
            width: 26px;
            height: 26px;
            line-height: 26px;
            color: var(--pcvs-ink);
            background-color: transparent;
            border-bottom: 1px solid var(--pcvs-line);
            font-size: 15px;
            font-weight: 500;
            transition: background-color 0.15s ease;
        }
        .leaflet-bar a:last-child { border-bottom: none; }
        .leaflet-bar a:hover { background-color: rgba(15, 23, 42, 0.06); }
        .leaflet-bar a.leaflet-disabled { color: #cbd5e1; background-color: transparent; }

        /* Layer control */
        .leaflet-control-layers-expanded {
            padding: 9px 11px 9px 10px !important;
            min-width: 148px;
        }
        .leaflet-control-layers-list::before {
            content: "Layers";
            display: block;
            margin-bottom: 7px;
            font-size: 9.5px;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: var(--pcvs-muted);
        }
        .leaflet-control-layers-overlays label {
            display: block;
            margin: 0 -5px !important;
            padding: 3px 5px !important;
            border-radius: 6px;
            font-size: 12px;
            line-height: 1.35 !important;
            cursor: pointer;
            transition: background 0.12s ease;
        }
        .leaflet-control-layers-overlays label:hover { background: rgba(15, 23, 42, 0.05); }
        .leaflet-control-layers-selector {
            margin: 0 7px 0 0 !important;
            accent-color: #2f6fed;
            vertical-align: -2px;
        }
        .leaflet-control-layers-separator { display: none !important; }

        /* Popups and tooltips */
        .leaflet-popup-content-wrapper {
            border-radius: var(--pcvs-radius);
            box-shadow: var(--pcvs-shadow);
            border: 1px solid var(--pcvs-line);
        }
        .leaflet-popup-content {
            margin: 11px 13px;
            font-size: 12px;
            line-height: 1.5;
        }
        .leaflet-tooltip {
            border: 1px solid var(--pcvs-line);
            border-radius: 7px;
            box-shadow: var(--pcvs-shadow);
            font-size: 11px;
            padding: 3px 8px;
            color: var(--pcvs-ink);
            background: var(--pcvs-surface);
        }
        .leaflet-control-attribution {
            background: rgba(255, 255, 255, 0.78) !important;
            border-radius: 6px 0 0 0;
            font-size: 10px;
            color: var(--pcvs-muted);
        }

        /* Popup body used by data point markers */
        .pcvs-popup { font-size: 12px; color: var(--pcvs-ink); }
        .pcvs-popup-head {
            font-size: 9.5px;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: var(--pcvs-muted);
            margin-bottom: 6px;
        }
        .pcvs-popup-name { font-weight: 600; margin-bottom: 4px; }
        .pcvs-popup dl {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 2px 10px;
            margin: 0;
        }
        .pcvs-popup dt { color: var(--pcvs-muted); }
        .pcvs-popup dd { margin: 0; }
        .pcvs-popup-item + .pcvs-popup-item {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid var(--pcvs-line);
        }
        .pcvs-dot {
            display: inline-block;
            width: 9px;
            height: 9px;
            border-radius: 50%;
            border: 1px solid rgba(15, 23, 42, 0.55);
            vertical-align: -1px;
            margin-right: 5px;
        }
        .pcvs-scroll { max-height: 260px; overflow-y: auto; }

        /* Slim scrollbars for the panels that can overflow */
        .pcvs-scroll,
        .basin-filter-list {
            scrollbar-width: thin;
            scrollbar-color: rgba(15, 23, 42, 0.2) transparent;
        }
        .pcvs-scroll::-webkit-scrollbar,
        .basin-filter-list::-webkit-scrollbar { width: 9px; }
        .pcvs-scroll::-webkit-scrollbar-track,
        .basin-filter-list::-webkit-scrollbar-track { background: transparent; }
        .pcvs-scroll::-webkit-scrollbar-thumb,
        .basin-filter-list::-webkit-scrollbar-thumb {
            background: rgba(15, 23, 42, 0.18);
            background-clip: content-box;
            border: 3px solid transparent;
            border-radius: 999px;
        }
        .pcvs-scroll::-webkit-scrollbar-thumb:hover,
        .basin-filter-list::-webkit-scrollbar-thumb:hover {
            background: rgba(15, 23, 42, 0.32);
            background-clip: content-box;
        }

        /* Data points: a white halo lifts the climate colour off the raster */
        path.pcvs-point { filter: __MARKER_HALO__; }

        /* Caption kept out of the map and out of exports: the viewer header
           already names the age and the interpolation. */
        .pcvs-title { display: none !important; }

        /* Colour statistics */
        .color-stats-control { padding: 9px 11px 10px; }
        .color-stats-control table {
            border-collapse: collapse;
            font-size: 11.5px;
            font-variant-numeric: tabular-nums;
            margin-top: 5px;
        }
        .color-stats-control th {
            font-size: 9.5px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--pcvs-muted);
            padding-bottom: 3px;
        }
        .color-stats-control td { padding: 1px 0; }
        .color-stats-control td + td,
        .color-stats-control th + th { padding-left: 14px; }

        .climate-point-icon { background: transparent !important; border: none !important; }

        /* Cover shown while an export reframes the map (see the export API) */
        .pcvs-export-veil {
            position: fixed;
            inset: 0;
            z-index: 100000;
            display: grid;
            place-items: center;
            background: __BACKGROUND__;
            font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI",
                         Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748b;
            opacity: 0;
            transition: opacity 0.18s ease;
        }
        .pcvs-export-veil.on { opacity: 1; }
        .pcvs-export-veil div { text-align: center; }
        .pcvs-export-veil i {
            display: block;
            width: 18px;
            height: 18px;
            margin: 0 auto 11px;
            border: 2px solid rgba(15, 23, 42, 0.13);
            border-top-color: #2f6fed;
            border-radius: 50%;
            animation: pcvs-spin 0.8s linear infinite;
        }
        @keyframes pcvs-spin { to { transform: rotate(360deg); } }

        /* Covers the reserved footer of a raster-only export so coastlines
           south of the surface do not run through the coverage card. */
        .pcvs-export-stats-band {
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 700;
            pointer-events: none;
            background: __BACKGROUND__;
        }

        /* Everything that only exists for on-screen interaction is stripped
           from exports, so a PDF shows the map and its read-only panels only. */
        .pcvs-exporting .leaflet-control-zoom,
        .pcvs-exporting .leaflet-control-layers,
        .pcvs-exporting .leaflet-control-attribution,
        .pcvs-exporting .leaflet-control-fullscreen,
        .pcvs-exporting .leaflet-control-measure,
        .pcvs-exporting .leaflet-measure-resultpopup,
        .pcvs-exporting .basin-filter-control,
        .pcvs-exporting .leaflet-popup,
        .pcvs-exporting .leaflet-tooltip {
            display: none !important;
        }
        /* Backdrop blur cannot be rasterised, and renderers that try leave an
           opaque block behind each panel. The panels are near-opaque anyway. */
        .pcvs-exporting .pcvs-panel,
        .pcvs-exporting .leaflet-tooltip,
        .pcvs-exporting .leaflet-control-layers,
        .pcvs-exporting .leaflet-bar {
            -webkit-backdrop-filter: none !important;
            backdrop-filter: none !important;
            background: #ffffff !important;
            box-shadow: none !important;
        }
        /* CSS filters flatten every data point into a small bitmap when
           Chromium prints. The halo is a screen affordance; the PDF keeps
           the dark stroke so the circles stay editable paths. */
        .pcvs-exporting path.pcvs-point,
        .pcvs-exporting .climate-point-icon svg {
            filter: none !important;
        }
        .pcvs-exporting .graticule-label span {
            text-shadow: none !important;
        }
    </style>
    {% endmacro %}
"""


def _add_map_theme(map_obj):
    """Apply the shared visual language to the Leaflet controls of a map."""
    theme = MacroElement()
    theme._template = Template(MAP_THEME_CSS
                               .replace('__BACKGROUND__', MAP_BACKGROUND)
                               .replace('__MARKER_HALO__', MARKER_HALO))
    map_obj.get_root().add_child(theme)


def _script_macro(map_obj, body, **substitutions):
    """Attach a ``{% macro script %}`` block to a map, filling in placeholders.

    ``body`` is plain JavaScript containing ``__NAME__`` placeholders, replaced
    by the matching keyword argument. Values are inserted verbatim, so callers
    pass JSON. Keeping the JavaScript free of Python formatting syntax avoids
    escaping every brace and percent sign it contains.
    """
    for name, value in substitutions.items():
        body = body.replace(f'__{name.upper()}__', value)
    macro = MacroElement()
    macro._template = Template(
        '{% macro script(this, kwargs) %}\n' + body + '\n{% endmacro %}'
    )
    map_obj.add_child(macro)


def _add_export_api(map_obj, raster_bounds, full_bounds, export_basename):
    """Expose ``window.PCVS``, the export entry point used by page and toolbar.

    The API frames the map on a chosen region and strips interactive chrome.
    Both export paths reuse that framing: Playwright then prints the page
    (vector points, raster surface as an image) and the viewer button draws
    the same layers into a jsPDF document so a live download stays editable.
    """
    config = json.dumps({
        'rasterBounds': raster_bounds,
        'fullBounds': full_bounds,
        'basename': export_basename,
        'baseHeight': PDF_BASE_HEIGHT_PX,
        'minWidth': PDF_MIN_WIDTH_PX,
        'maxWidth': PDF_MAX_WIDTH_PX,
        'background': MAP_BACKGROUND,
        'htmlToImage': HTML_TO_IMAGE_URL,
        'jspdf': JSPDF_URL,
        'pointRadius': POINT_RADIUS_PX,
        'pointWeight': POINT_WEIGHT_PX,
    }, ensure_ascii=False)

    _script_macro(map_obj, """
        (function() {
            var map = {{ this._parent.get_name() }};
            var CFG = __CONFIG__;

            function boundsFor(scope) {
                var b = (scope === 'raster' && CFG.rasterBounds) ? CFG.rasterBounds
                                                                 : CFG.fullBounds;
                return b ? L.latLngBounds(b) : map.getBounds();
            }

            /* When a raster-only export keeps the coverage panel, the fitted
               region sits above it so the panel does not cover the surface.
               Measured from the live control so an unchecked layer costs no
               extra margin. */
            function colorStatsPadPx() {
                var panel = map.getContainer().querySelector('.color-stats-control');
                if (!panel) return 0;
                var mapRect = map.getContainer().getBoundingClientRect();
                var panelRect = panel.getBoundingClientRect();
                if (panelRect.height < 1) return 0;
                return Math.max(0, Math.ceil(mapRect.bottom - panelRect.top) + 12);
            }

            /* Page geometry follows the region being exported, so the map fills
               it completely instead of sitting inside white margins. */
            function exportSize(scope) {
                var b = boundsFor(scope);
                var lonSpan = Math.abs(b.getEast() - b.getWest());
                var latSpan = Math.abs(b.getNorth() - b.getSouth());
                var ratio = (latSpan > 0) ? lonSpan / latSpan : 1.4;
                var height = CFG.baseHeight;
                var width = Math.round(height * ratio);
                if (width < CFG.minWidth) {
                    height = Math.round(CFG.minWidth / ratio);
                    width = CFG.minWidth;
                } else if (width > CFG.maxWidth) {
                    height = Math.round(CFG.maxWidth / ratio);
                    width = CFG.maxWidth;
                }
                var padBottom = (scope === 'raster') ? colorStatsPadPx() : 0;
                return {width: width, height: height + padBottom, padBottom: padBottom};
            }

            var saved = null;
            var veil = null;

            /* Framing an export moves and resizes the map, which on screen
               reads as it jumping to another zoom level. The veil hides that
               from view; it lives outside the captured container, so it never
               reaches the rendered image. */
            function showVeil() {
                if (veil) return;
                veil = L.DomUtil.create('div', 'pcvs-export-veil', document.body);
                veil.innerHTML = '<div><i></i>Rendering PDF</div>';
                requestAnimationFrame(function() {
                    if (veil) L.DomUtil.addClass(veil, 'on');
                });
            }

            function hideVeil() {
                if (!veil) return;
                var node = veil;
                veil = null;
                L.DomUtil.removeClass(node, 'on');
                setTimeout(function() {
                    if (node.parentNode) node.parentNode.removeChild(node);
                }, 220);
            }

            function beginExport(options) {
                options = options || {};
                var scope = options.scope === 'raster' ? 'raster' : 'full';
                var container = map.getContainer();
                if (options.veil) showVeil();
                var size = exportSize(scope);
                saved = {
                    center: map.getCenter(),
                    zoom: map.getZoom(),
                    cssText: container.style.cssText,
                    bodyOverflow: document.body.style.overflow,
                    statsBand: null
                };
                document.documentElement.classList.add('pcvs-exporting');
                map.closePopup();
                if (options.resize !== false) {
                    document.body.style.overflow = 'hidden';
                    container.style.position = 'absolute';
                    container.style.top = '0';
                    container.style.left = '0';
                    container.style.width = size.width + 'px';
                    container.style.height = size.height + 'px';
                }
                map.invalidateSize({animate: false, pan: false});
                var fitOpts = {animate: false, padding: [0, 0]};
                if (scope === 'raster' && size.padBottom) {
                    fitOpts = {
                        animate: false,
                        paddingTopLeft: [0, 0],
                        paddingBottomRight: [0, size.padBottom]
                    };
                    var band = L.DomUtil.create('div', 'pcvs-export-stats-band', container);
                    band.style.height = size.padBottom + 'px';
                    saved.statsBand = band;
                }
                map.fitBounds(boundsFor(scope), fitOpts);
                return size;
            }

            function endExport() {
                document.documentElement.classList.remove('pcvs-exporting');
                hideVeil();
                if (!saved) return;
                if (saved.statsBand && saved.statsBand.parentNode) {
                    saved.statsBand.parentNode.removeChild(saved.statsBand);
                }
                var container = map.getContainer();
                container.style.cssText = saved.cssText;
                document.body.style.overflow = saved.bodyOverflow;
                map.invalidateSize({animate: false, pan: false});
                map.setView(saved.center, saved.zoom, {animate: false});
                saved = null;
            }

            function settled() {
                return new Promise(function(resolve) {
                    map.whenReady(function() {
                        var pending = Array.prototype.filter.call(
                            map.getContainer().querySelectorAll('img'),
                            function(img) { return !img.complete; }
                        ).map(function(img) {
                            return new Promise(function(done) {
                                img.addEventListener('load', done, {once: true});
                                img.addEventListener('error', done, {once: true});
                            });
                        });
                        Promise.all(pending).then(function() {
                            requestAnimationFrame(function() {
                                setTimeout(resolve, 350);
                            });
                        });
                    });
                });
            }

            function loadScript(src) {
                return new Promise(function(resolve, reject) {
                    var s = document.createElement('script');
                    s.src = src;
                    s.onload = resolve;
                    s.onerror = function() { reject(new Error('Failed to load ' + src)); };
                    document.head.appendChild(s);
                });
            }

            function libs() {
                var chain = Promise.resolve();
                if (!window.htmlToImage) {
                    chain = chain.then(function() { return loadScript(CFG.htmlToImage); });
                }
                if (!window.jspdf) {
                    chain = chain.then(function() { return loadScript(CFG.jspdf); });
                }
                return chain;
            }

            var PX = 0.75;

            function hexRgb(color) {
                if (!color) return [0, 0, 0];
                color = String(color).trim();
                var rgb = color.match(/^rgba?\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)/i);
                if (rgb) return [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])];
                var hex = color.replace('#', '');
                if (hex.length === 3) {
                    hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
                }
                if (hex.length < 6) return [0, 0, 0];
                return [
                    parseInt(hex.slice(0, 2), 16),
                    parseInt(hex.slice(2, 4), 16),
                    parseInt(hex.slice(4, 6), 16)
                ];
            }

            function newPdf(size) {
                var ptW = size.width * PX;
                var ptH = size.height * PX;
                return new window.jspdf.jsPDF({
                    orientation: ptW >= ptH ? 'landscape' : 'portrait',
                    unit: 'pt',
                    format: [ptW, ptH],
                    compress: true
                });
            }

            function collectLayers() {
                var rasters = [];
                var lines = [];
                var circles = [];
                var pies = [];
                var labels = [];
                function walk(layer) {
                    if (layer.eachLayer) {
                        layer.eachLayer(walk);
                        return;
                    }
                    if (!layer._map) return;
                    if (layer instanceof L.ImageOverlay) {
                        rasters.push(layer);
                        return;
                    }
                    if (layer instanceof L.CircleMarker) {
                        circles.push(layer);
                        return;
                    }
                    if (layer instanceof L.Polyline) {
                        lines.push(layer);
                        return;
                    }
                    if (!layer._icon) return;
                    if (layer._icon.classList.contains('graticule-label')) {
                        labels.push(layer);
                        return;
                    }
                    if (layer._icon.querySelector('svg')) pies.push(layer);
                }
                map.eachLayer(walk);
                return {
                    rasters: rasters,
                    lines: lines,
                    circles: circles,
                    pies: pies,
                    labels: labels
                };
            }

            function flattenRings(latlngs) {
                if (!latlngs || !latlngs.length) return [];
                var first = latlngs[0];
                if (first && first.lat !== undefined) return [latlngs];
                var rings = [];
                for (var i = 0; i < latlngs.length; i++) {
                    rings = rings.concat(flattenRings(latlngs[i]));
                }
                return rings;
            }

            function setOpacity(doc, opacity) {
                if (typeof doc.GState !== 'function') return;
                doc.setGState(new doc.GState({opacity: opacity}));
            }

            function imageDataUrl(img) {
                if (!img) return Promise.resolve(null);
                var src = img.currentSrc || img.src || '';
                if (src.indexOf('data:image/') === 0) return Promise.resolve(src);
                try {
                    var canvas = document.createElement('canvas');
                    canvas.width = img.naturalWidth || img.width;
                    canvas.height = img.naturalHeight || img.height;
                    if (!canvas.width || !canvas.height) return Promise.resolve(null);
                    canvas.getContext('2d').drawImage(img, 0, 0);
                    return Promise.resolve(canvas.toDataURL('image/png'));
                } catch (err) {
                    return Promise.resolve(null);
                }
            }

            function drawRasters(doc, rasters) {
                var chain = Promise.resolve();
                rasters.forEach(function(layer) {
                    chain = chain.then(function() {
                        return imageDataUrl(layer.getElement && layer.getElement());
                    }).then(function(url) {
                        if (!url || !layer.getBounds) return;
                        var bounds = layer.getBounds();
                        var nw = map.latLngToContainerPoint(bounds.getNorthWest());
                        var se = map.latLngToContainerPoint(bounds.getSouthEast());
                        var opacity = (layer.options && layer.options.opacity != null)
                            ? layer.options.opacity : 1;
                        setOpacity(doc, opacity);
                        doc.addImage(
                            url, 'PNG',
                            nw.x * PX, nw.y * PX,
                            (se.x - nw.x) * PX, (se.y - nw.y) * PX
                        );
                        setOpacity(doc, 1);
                    });
                });
                return chain;
            }

            function drawPolyline(doc, layer) {
                var opt = layer.options || {};
                var color = hexRgb(opt.color || '#334155');
                var weight = (opt.weight != null ? opt.weight : 1) * PX;
                var opacity = opt.opacity != null ? opt.opacity : 1;
                var closed = (typeof L.Polygon === 'function' && layer instanceof L.Polygon);
                doc.setDrawColor(color[0], color[1], color[2]);
                doc.setLineWidth(Math.max(0.15, weight));
                if (doc.setLineCap) doc.setLineCap(opt.lineCap || 'round');
                if (doc.setLineJoin) doc.setLineJoin(opt.lineJoin || 'round');
                if (doc.setLineDashPattern) {
                    if (opt.dashArray) {
                        var dash = String(opt.dashArray).split(/[\\s,]+/).map(Number)
                            .filter(function(n) { return isFinite(n); })
                            .map(function(n) { return n * PX; });
                        doc.setLineDashPattern(dash, 0);
                    } else {
                        doc.setLineDashPattern([], 0);
                    }
                }
                setOpacity(doc, opacity);
                flattenRings(layer.getLatLngs()).forEach(function(ring) {
                    if (ring.length < 2) return;
                    var start = map.latLngToContainerPoint(ring[0]);
                    var deltas = [];
                    var prev = start;
                    for (var i = 1; i < ring.length; i++) {
                        var point = map.latLngToContainerPoint(ring[i]);
                        deltas.push([(point.x - prev.x) * PX, (point.y - prev.y) * PX]);
                        prev = point;
                    }
                    doc.lines(deltas, start.x * PX, start.y * PX, [1, 1], 'S', closed);
                });
                setOpacity(doc, 1);
                if (doc.setLineDashPattern) doc.setLineDashPattern([], 0);
            }

            function drawCircle(doc, layer) {
                var latlng = layer.getLatLng();
                var point = map.latLngToContainerPoint(latlng);
                var radius = (layer.options.radius || CFG.pointRadius) * PX;
                var weight = (layer.options.weight != null ? layer.options.weight : CFG.pointWeight) * PX;
                var fill = hexRgb(layer.options.fillColor || layer.options.color || '#94a3b8');
                var stroke = hexRgb(layer.options.color || '#16253a');
                var fillOp = layer.options.fillOpacity != null ? layer.options.fillOpacity : 1;
                setOpacity(doc, fillOp);
                doc.setFillColor(fill[0], fill[1], fill[2]);
                doc.setDrawColor(stroke[0], stroke[1], stroke[2]);
                doc.setLineWidth(Math.max(0.15, weight));
                if (doc.setLineDashPattern) doc.setLineDashPattern([], 0);
                doc.circle(point.x * PX, point.y * PX, radius, 'FD');
                setOpacity(doc, 1);
            }

            function drawPie(doc, layer) {
                var svg = layer._icon && layer._icon.querySelector('svg');
                if (!svg) return;
                var paths = svg.querySelectorAll('path');
                var fills = [];
                for (var i = 0; i < paths.length; i++) {
                    fills.push(hexRgb(paths[i].getAttribute('fill') || '#94a3b8'));
                }
                if (!fills.length) return;
                var circle = svg.querySelector('circle');
                var radiusPx = circle ? parseFloat(circle.getAttribute('r')) : CFG.pointRadius;
                var weightPx = circle ? parseFloat(circle.getAttribute('stroke-width')) : CFG.pointWeight;
                var stroke = hexRgb(
                    (circle && circle.getAttribute('stroke')) || '#16253a'
                );
                var point = map.latLngToContainerPoint(layer.getLatLng());
                var x = point.x * PX;
                var y = point.y * PX;
                var radius = radiusPx * PX;
                var start = -Math.PI / 2;
                var slice = (Math.PI * 2) / fills.length;
                for (var s = 0; s < fills.length; s++) {
                    var a0 = start + s * slice;
                    var a1 = start + (s + 1) * slice;
                    var steps = Math.max(8, Math.ceil(Math.abs(a1 - a0) / (Math.PI / 12)));
                    var originX = x;
                    var originY = y;
                    var deltas = [];
                    var prevX = originX;
                    var prevY = originY;
                    for (var t = 0; t <= steps; t++) {
                        var angle = a0 + (a1 - a0) * (t / steps);
                        var nx = x + radius * Math.cos(angle);
                        var ny = y + radius * Math.sin(angle);
                        deltas.push([nx - prevX, ny - prevY]);
                        prevX = nx;
                        prevY = ny;
                    }
                    doc.setFillColor(fills[s][0], fills[s][1], fills[s][2]);
                    doc.lines(deltas, originX, originY, [1, 1], 'F', true);
                }
                doc.setDrawColor(stroke[0], stroke[1], stroke[2]);
                doc.setLineWidth(Math.max(0.15, weightPx * PX));
                doc.circle(x, y, radius, 'S');
            }

            function drawLabel(doc, layer) {
                var node = layer._icon && layer._icon.querySelector('span');
                var text = node ? String(node.textContent || '').trim() : '';
                if (!text || !layer._icon) return;
                var box = layer._icon.getBoundingClientRect();
                var origin = map.getContainer().getBoundingClientRect();
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(9.5 * PX);
                doc.setTextColor(85, 100, 122);
                doc.text(text, (box.left - origin.left) * PX,
                         (box.top - origin.top + box.height * 0.72) * PX);
            }

            function drawColorStats(doc) {
                var panel = map.getContainer().querySelector('.color-stats-control');
                if (!panel || !panel.getClientRects().length || !window.htmlToImage) {
                    return Promise.resolve();
                }
                var box = panel.getBoundingClientRect();
                var origin = map.getContainer().getBoundingClientRect();
                return window.htmlToImage.toPng(panel, {
                    backgroundColor: '#ffffff',
                    pixelRatio: 2,
                    cacheBust: false
                }).then(function(url) {
                    doc.addImage(
                        url, 'PNG',
                        (box.left - origin.left) * PX,
                        (box.top - origin.top) * PX,
                        box.width * PX,
                        box.height * PX
                    );
                }).catch(function() { return null; });
            }

            function composeVectorPdf(scope, size) {
                var doc = newPdf(size);
                var bg = hexRgb(CFG.background);
                doc.setFillColor(bg[0], bg[1], bg[2]);
                doc.rect(0, 0, size.width * PX, size.height * PX, 'F');
                var layers = collectLayers();
                return drawRasters(doc, layers.rasters).then(function() {
                    layers.lines.forEach(function(layer) { drawPolyline(doc, layer); });
                    layers.circles.forEach(function(layer) { drawCircle(doc, layer); });
                    layers.pies.forEach(function(layer) { drawPie(doc, layer); });
                    layers.labels.forEach(function(layer) { drawLabel(doc, layer); });
                    return drawColorStats(doc);
                }).then(function() {
                    return {
                        buffer: doc.output('arraybuffer'),
                        filename: CFG.basename + '_' + scope + '.pdf'
                    };
                });
            }

            function composeBitmapPdf(scope, size) {
                var ratio = Math.min(3, Math.max(1.5, 3600 / size.width));
                return window.htmlToImage.toCanvas(map.getContainer(), {
                    backgroundColor: CFG.background,
                    width: size.width,
                    height: size.height,
                    pixelRatio: ratio,
                    cacheBust: false
                }).then(function(canvas) {
                    var doc = newPdf(size);
                    doc.addImage(
                        canvas.toDataURL('image/jpeg', 0.95), 'JPEG',
                        0, 0, size.width * PX, size.height * PX, undefined, 'FAST'
                    );
                    return {
                        buffer: doc.output('arraybuffer'),
                        filename: CFG.basename + '_' + scope + '.pdf'
                    };
                });
            }

            /* Full-map downloads stay vector so points can be edited. The
               raster-only crop may stay a bitmap, matching the published pair. */
            function exportPdf(options) {
                options = options || {};
                var scope = options.scope === 'raster' ? 'raster' : 'full';
                var size;
                showVeil();
                return libs().then(function() {
                    size = beginExport({scope: scope, veil: true});
                    return settled();
                }).then(function() {
                    return scope === 'raster'
                        ? composeBitmapPdf(scope, size)
                        : composeVectorPdf(scope, size);
                }).then(function(result) {
                    endExport();
                    return result;
                }).catch(function(err) {
                    endExport();
                    throw err;
                });
            }

            window.PCVS = {
                exportSize: exportSize,
                beginExport: beginExport,
                endExport: endExport,
                settled: settled,
                exportPdf: exportPdf,
                basename: CFG.basename
            };

            /* The shell page drives exports from its toolbar over postMessage,
               which keeps working even when the iframe is not same-origin. */
            window.addEventListener('message', function(event) {
                var data = event.data;
                if (!data || data.pcvs !== 'export' || !event.source) return;
                exportPdf({scope: data.scope}).then(function(result) {
                    event.source.postMessage({
                        pcvs: 'export-result',
                        id: data.id,
                        filename: result.filename,
                        buffer: result.buffer
                    }, '*', [result.buffer]);
                }).catch(function(err) {
                    event.source.postMessage({
                        pcvs: 'export-error',
                        id: data.id,
                        message: String(err && err.message || err)
                    }, '*');
                });
            });
        })();
    """, config=config)


def create_map(points_data, coastline_data, geotiff_path=None, output_file='map.html', 
               raster_img_path='raster_overlay.png',
               point_values_override=None, raster_layer_name=LAYER_RASTER,
               gradient_sharp=2.5,
               color_stats_img_path=None, color_stats_name=None,
               method=None, age_label='', map_subtitle=''):
    """Create a Folium map with points, coastlines, and optional raster.

    ``age_label`` and ``map_subtitle`` are accepted for caller compatibility;
    the viewer header is what names the reconstruction.
    """
    
    # Calculate combined bounds
    print("Calculating map bounds...")
    bounds_list = []
    
    points_bounds = get_geojson_bounds(points_data)
    if points_bounds:
        bounds_list.append(points_bounds)
    
    coastline_bounds = get_geojson_bounds(coastline_data)
    if coastline_bounds:
        bounds_list.append(coastline_bounds)
    
    # Try to get bounds from GeoTIFF
    if geotiff_path and os.path.exists(geotiff_path):
        try:
            geotiff_bounds = get_geotiff_bounds(geotiff_path)
            if geotiff_bounds:
                bounds_list.append(geotiff_bounds)
        except:
            pass
    
    # Calculate overall bounds
    if bounds_list:
        all_lats = [b[0][0] for b in bounds_list] + [b[1][0] for b in bounds_list]
        all_lons = [b[0][1] for b in bounds_list] + [b[1][1] for b in bounds_list]
        full_bounds = [[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]]
        center_lat = (min(all_lats) + max(all_lats)) / 2
        center_lon = (min(all_lons) + max(all_lons)) / 2
    else:
        # Default center (South America region)
        full_bounds = None
        center_lat = -20
        center_lon = -20
    
    # Create base map. Fractional zoom (zoomSnap 0) makes fit_bounds land on the
    # exact framing asked for; with the default integer snapping, datasets whose
    # extent falls just over a power-of-two boundary drop a whole zoom level and
    # come out framed very differently from their neighbours.
    print("Creating Folium map...")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=3,
        tiles=None,
        crs='EPSG4326',
        zoom_snap=0,
        zoom_delta=0.5,
    )
    
    # Add OpenStreetMap hidden and without checkbox in LayerControl
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap', overlay=True, 
                     control=False, show=False).add_to(m)

    # Added early so it sits with the zoom buttons in the top-left stack,
    # above the basin filter panel.
    plugins.Fullscreen().add_to(m)
    
    # Add GeoTIFF raster overlay if provided
    if geotiff_path and os.path.exists(geotiff_path):
        print(f"Adding GeoTIFF raster overlay...")
        try:
            create_raster_overlay(geotiff_path, m, raster_img_path=raster_img_path, 
                                 layer_name=raster_layer_name,
                                 points_data=points_data, preserve_points=True, point_radius=0.3,
                                 point_values_override=point_values_override, gradient_sharp=gradient_sharp)
        except Exception as e:
            import traceback
            print(f"Could not add GeoTIFF: {e}")
            print(traceback.format_exc())
    
    # Add GeoJSON layers
    print("Adding GeoJSON layers...")
    
    # Add coastline layer
    folium.GeoJson(
        coastline_data,
        name=LAYER_COASTLINES,
        smooth_factor=1.3,
        style_function=lambda feature: {
            'color': COASTLINE_COLOR,
            'weight': 1.05,
            'opacity': 0.88,
            'lineCap': 'round',
            'lineJoin': 'round',
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME', 'TIME'],
            aliases=['Location:', 'Age (Ma):'],
            sticky=True
        )
    ).add_to(m)
    
    # Marker geometry. The same radius and stroke drive solid CircleMarkers, the
    # multi-climate SVG icons and the PDF export, so every point on the map is
    # the same size no matter how many records share the location.
    point_radius_px = POINT_RADIUS_PX
    point_weight = POINT_WEIGHT_PX
    color_map = CLIMATE_COLORS
    
    # Group features by coordinate so overlapping points share a single marker
    coord_groups = OrderedDict()
    for feature in points_data.get('features', []):
        geom = feature.get('geometry', {})
        if geom.get('type') == 'Point':
            coords = geom.get('coordinates', [])
            if coords:
                key = (coords[0], coords[1])
                coord_groups.setdefault(key, []).append(feature)
    
    # Create a FeatureGroup for points
    points_group = folium.FeatureGroup(name=LAYER_POINTS)
    
    marker_basins_list = []
    for (lon, lat), features in coord_groups.items():
        climates = [get_climate_class(f.get('properties', {}), '') for f in features]
        basins = list(OrderedDict.fromkeys(
            f.get('properties', {}).get('Basin_Sub_') or 'N/A' for f in features
        ))
        marker_basins_list.append(basins)

        # Prevalence wins; ties (2 or 3) become a split-color marker.
        marker_climates = resolve_marker_climates(climates)

        if len(features) == 1:
            props = features[0].get('properties', {})
            formation = props.get('Formation') or 'N/A'
            basin = props.get('Basin_Sub_') or 'N/A'
            country = props.get('Country') or 'N/A'
            climate_val = get_climate_class(props) or 'N/A'
            age = props.get('TIME') or 'N/A'
            popup_html = (
                '<div class="pcvs-popup">'
                '<div class="pcvs-popup-head">Data point</div>'
                f'<div class="pcvs-popup-name">{formation}</div>'
                '<dl>'
                f'<dt>Basin</dt><dd>{basin}</dd>'
                f'<dt>Country</dt><dd>{country}</dd>'
                f'<dt>Climate</dt><dd>{_climate_name(climate_val)}</dd>'
                f'<dt>Age</dt><dd>{age} Ma</dd>'
                '</dl></div>'
            )
            tooltip_text = f'{formation} ({basin})'
        else:
            parts = []
            for feat in features:
                props = feat.get('properties', {})
                climate = get_climate_class(props) or ''
                dot_color = color_map.get(climate, '#94a3b8')
                formation = props.get('Formation') or 'N/A'
                basin = props.get('Basin_Sub_') or 'N/A'
                country = props.get('Country') or 'N/A'
                age = props.get('TIME') or 'N/A'
                parts.append(
                    '<div class="pcvs-popup-item">'
                    f'<div class="pcvs-popup-name">'
                    f'<span class="pcvs-dot" style="background:{dot_color}"></span>'
                    f'{formation}</div>'
                    '<dl>'
                    f'<dt>Basin</dt><dd>{basin}</dd>'
                    f'<dt>Country</dt><dd>{country}</dd>'
                    f'<dt>Climate</dt><dd>{_climate_name(climate)}</dd>'
                    f'<dt>Age</dt><dd>{age} Ma</dd>'
                    '</dl></div>'
                )
            counts = Counter(c for c in climates if c in _CLIMATE_DISPLAY_ORDER)
            count_bits = [f'{_climate_name(c)} {counts[c]}'
                          for c in _CLIMATE_DISPLAY_ORDER if c in counts]
            popup_html = (
                '<div class="pcvs-popup">'
                f'<div class="pcvs-popup-head">{len(features)} points at this location</div>'
                f'<div class="pcvs-popup-head" style="margin-bottom:8px">'
                f'{" · ".join(count_bits)}</div>'
                '<div class="pcvs-scroll">' + ''.join(parts) + '</div></div>'
            )
            names = list(OrderedDict.fromkeys(
                f.get('properties', {}).get('Formation') or 'N/A' for f in features
            ))
            tooltip_text = f'{len(features)} points: {", ".join(names)}'

        # Solid markers keep the original CircleMarker size/stroke; only ties
        # use a DivIcon SVG pie with the same radius and weight.
        if len(marker_climates) <= 1:
            fill_color = color_map.get(marker_climates[0], '#94a3b8') if marker_climates else '#94a3b8'
            folium.CircleMarker(
                location=[lat, lon],
                radius=point_radius_px,
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=tooltip_text,
                color=MARKER_STROKE_COLOR,
                fillColor=fill_color,
                fillOpacity=MARKER_FILL_OPACITY,
                weight=point_weight,
                class_name='pcvs-point',
            ).add_to(points_group)
        else:
            icon_html, icon_outer_px = climate_marker_icon_html(
                marker_climates, color_map,
                radius_px=point_radius_px, weight_px=point_weight,
            )
            # Fractional anchor: the SVG is an odd number of pixels wide, so
            # rounding here would offset split markers from solid ones.
            icon_anchor = icon_outer_px / 2.0
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=tooltip_text,
                icon=folium.DivIcon(
                    html=icon_html,
                    icon_size=(icon_outer_px, icon_outer_px),
                    icon_anchor=(icon_anchor, icon_anchor),
                    class_name='climate-point-icon',
                ),
            ).add_to(points_group)
    
    points_group.add_to(m)

    # Basin filter control (topleft, next to zoom)
    basins = sorted(set(
        feature.get('properties', {}).get('Basin_Sub_') or 'N/A'
        for feature in points_data.get('features', [])
        if feature.get('geometry', {}).get('type') == 'Point'
           and feature.get('geometry', {}).get('coordinates')
    ))
    marker_basins = marker_basins_list

    if basins:
        pg_name = points_group.get_name()
        basins_json = json.dumps(basins, ensure_ascii=False)
        marker_basins_json = json.dumps(marker_basins, ensure_ascii=False)

        basin_css = MacroElement()
        basin_css._template = Template("""
            {% macro header(this, kwargs) %}
            <style>
                /* Sits at the top of the top-left stack, above the zoom
                   buttons; the open list floats over them instead of pushing
                   them down. */
                .basin-filter-control {
                    position: relative;
                    width: 216px;
                    overflow: visible;
                    z-index: 900;
                }
                .basin-filter-control.open { z-index: 1100; }
                .basin-filter-header {
                    padding: 7px 10px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 8px;
                    cursor: pointer;
                    user-select: none;
                    border-radius: var(--pcvs-radius);
                }
                .basin-filter-header:hover { background: rgba(15, 23, 42, 0.04); }
                .basin-filter-count {
                    margin-left: auto;
                    padding: 1px 6px;
                    border-radius: 999px;
                    background: rgba(47, 111, 237, 0.12);
                    color: #2f6fed;
                    font-size: 10px;
                    font-weight: 650;
                    font-variant-numeric: tabular-nums;
                }
                .basin-filter-header .arrow {
                    font-size: 8px;
                    color: #64748b;
                    transition: transform 0.2s ease;
                }
                .basin-filter-control.open .basin-filter-header .arrow {
                    transform: rotate(180deg);
                }
                .basin-filter-body {
                    display: none;
                    position: absolute;
                    top: calc(100% + 6px);
                    left: 0;
                    width: 100%;
                    max-height: 56vh;
                    flex-direction: column;
                    padding: 9px 10px 10px;
                    background: var(--pcvs-surface);
                    border: 1px solid var(--pcvs-line);
                    border-radius: var(--pcvs-radius);
                    box-shadow: var(--pcvs-shadow);
                    -webkit-backdrop-filter: saturate(160%) blur(10px);
                    backdrop-filter: saturate(160%) blur(10px);
                }
                .basin-filter-control.open .basin-filter-body { display: flex; }
                .basin-filter-search {
                    flex: none;
                    width: 100%;
                    padding: 5px 8px;
                    margin-bottom: 7px;
                    font: inherit;
                    font-size: 11.5px;
                    color: inherit;
                    border: 1px solid var(--pcvs-line);
                    border-radius: 7px;
                    background: rgba(255, 255, 255, 0.7);
                    outline: none;
                }
                .basin-filter-search:focus {
                    border-color: #2f6fed;
                    box-shadow: 0 0 0 3px rgba(47, 111, 237, 0.14);
                }
                .basin-filter-actions { flex: none; display: flex; gap: 6px; margin-bottom: 7px; }
                .basin-filter-actions button {
                    flex: 1;
                    padding: 4px 6px;
                    font: inherit;
                    font-size: 10px;
                    font-weight: 600;
                    color: #475569;
                    border: 1px solid var(--pcvs-line);
                    border-radius: 7px;
                    background: rgba(255, 255, 255, 0.7);
                    cursor: pointer;
                    transition: background 0.12s ease, color 0.12s ease;
                }
                .basin-filter-actions button:hover { background: rgba(15, 23, 42, 0.06); color: #0f172a; }
                .basin-filter-list {
                    list-style: none;
                    padding: 0;
                    margin: 0;
                    overflow-y: auto;
                    min-height: 0;
                }
                .basin-filter-list li { border-radius: 6px; }
                .basin-filter-list li:hover { background: rgba(15, 23, 42, 0.05); }
                .basin-filter-list li.hidden { display: none; }
                .basin-filter-list label {
                    display: flex;
                    align-items: center;
                    gap: 7px;
                    padding: 3px 5px;
                    font-size: 11.5px;
                    line-height: 1.3;
                    cursor: pointer;
                }
                .basin-filter-list input[type="checkbox"] {
                    margin: 0;
                    flex: none;
                    accent-color: #2f6fed;
                }
                .basin-filter-empty {
                    padding: 6px 5px;
                    font-size: 11px;
                    color: #94a3b8;
                }
            </style>
            {% endmacro %}
        """)
        m.get_root().add_child(basin_css)

        _script_macro(m, """
            (function() {
                var map = {{ this._parent.get_name() }};
                var pointsGroup = __GROUP__;
                var allBasins = __BASINS__;
                var markerBasins = __MARKER_BASINS__;

                var allMarkers = [];
                pointsGroup.eachLayer(function(layer) {
                    allMarkers.push(layer);
                });

                var selectedBasins = new Set(allBasins);
                var countBadge, emptyHint;

                var filterControl = L.control({position: 'topleft'});
                filterControl.onAdd = function() {
                    var container = L.DomUtil.create('div', 'pcvs-panel basin-filter-control');
                    L.DomEvent.disableClickPropagation(container);
                    L.DomEvent.disableScrollPropagation(container);

                    var header = L.DomUtil.create('div', 'basin-filter-header', container);
                    header.innerHTML =
                        '<span class="pcvs-panel-title">Basins</span>' +
                        '<span class="basin-filter-count"></span>' +
                        '<span class="arrow">&#9660;</span>';
                    countBadge = header.querySelector('.basin-filter-count');

                    var body = L.DomUtil.create('div', 'basin-filter-body', container);

                    var search = L.DomUtil.create('input', 'basin-filter-search', body);
                    search.type = 'search';
                    search.placeholder = 'Search basins';

                    var actions = L.DomUtil.create('div', 'basin-filter-actions', body);
                    var selectAllBtn = L.DomUtil.create('button', '', actions);
                    selectAllBtn.textContent = 'Select all';
                    var deselectAllBtn = L.DomUtil.create('button', '', actions);
                    deselectAllBtn.textContent = 'Clear';

                    var list = L.DomUtil.create('ul', 'basin-filter-list', body);
                    emptyHint = L.DomUtil.create('div', 'basin-filter-empty', body);
                    emptyHint.textContent = 'No basin matches this search.';
                    emptyHint.style.display = 'none';

                    var rows = [];
                    allBasins.forEach(function(basin) {
                        var li = L.DomUtil.create('li', '', list);
                        var label = L.DomUtil.create('label', '', li);
                        var cb = document.createElement('input');
                        cb.type = 'checkbox';
                        cb.checked = true;
                        cb.value = basin;
                        label.appendChild(cb);
                        label.appendChild(document.createTextNode(basin));
                        rows.push({item: li, checkbox: cb, key: basin.toLowerCase()});
                        cb.addEventListener('change', function() {
                            if (this.checked) {
                                selectedBasins.add(this.value);
                            } else {
                                selectedBasins.delete(this.value);
                            }
                            applyFilter();
                        });
                    });

                    header.addEventListener('click', function() {
                        container.classList.toggle('open');
                        if (container.classList.contains('open')) search.focus();
                    });

                    map.on('click', function() { container.classList.remove('open'); });
                    document.addEventListener('keydown', function(event) {
                        if (event.key === 'Escape') container.classList.remove('open');
                    });

                    search.addEventListener('input', function() {
                        var q = this.value.trim().toLowerCase();
                        var visible = 0;
                        rows.forEach(function(row) {
                            var match = !q || row.key.indexOf(q) !== -1;
                            row.item.classList.toggle('hidden', !match);
                            if (match) visible++;
                        });
                        emptyHint.style.display = visible ? 'none' : 'block';
                    });

                    function setAll(checked) {
                        rows.forEach(function(row) {
                            if (row.item.classList.contains('hidden')) return;
                            row.checkbox.checked = checked;
                            if (checked) {
                                selectedBasins.add(row.checkbox.value);
                            } else {
                                selectedBasins.delete(row.checkbox.value);
                            }
                        });
                        applyFilter();
                    }
                    selectAllBtn.addEventListener('click', function() { setAll(true); });
                    deselectAllBtn.addEventListener('click', function() { setAll(false); });

                    return container;
                };
                filterControl.addTo(map);

                /* Leaflet stacks a corner in insertion order, and this control
                   is built after the zoom and fullscreen buttons; moving it to
                   the front puts it at the top of the top-left column. */
                var panel = filterControl.getContainer();
                var corner = panel.parentNode;
                if (corner && corner.firstChild !== panel) {
                    corner.insertBefore(panel, corner.firstChild);
                }

                function applyFilter() {
                    var shown = 0;
                    for (var i = 0; i < allMarkers.length; i++) {
                        var show = markerBasins[i].some(function(b) { return selectedBasins.has(b); });
                        if (show) {
                            shown++;
                            if (!pointsGroup.hasLayer(allMarkers[i])) {
                                pointsGroup.addLayer(allMarkers[i]);
                            }
                        } else if (pointsGroup.hasLayer(allMarkers[i])) {
                            pointsGroup.removeLayer(allMarkers[i]);
                        }
                    }
                    if (countBadge) countBadge.textContent = shown + ' / ' + allMarkers.length;
                }
                applyFilter();
            })();
        """, group=pg_name, basins=basins_json, marker_basins=marker_basins_json)

    # Add color stats as a fixed Leaflet control (bottom-left) toggled via LayerControl checkbox
    if color_stats_img_path and os.path.exists(color_stats_img_path):
        stats = classificar_por_intervalos(color_stats_img_path)
        if stats is not None:
            stats_checkbox_name = color_stats_name or LAYER_COLOR_STATS
            rows = [
                ('D', 'Dry', stats['count_yellow'], stats['pct_yellow']),
                ('S', 'Semi-arid', stats['count_green'], stats['pct_green']),
                ('H', 'Humid', stats['count_blue'], stats['pct_blue']),
            ]
            body = ''.join(
                '<tr>'
                f'<td><span class="pcvs-dot" style="background:'
                f'{CLIMATE_COLORS.get(code, "#cbd5e1")}"></span>{label}</td>'
                f'<td align="right">{count:,}</td>'
                f'<td align="right">{pct:.1f}%</td>'
                '</tr>'
                for code, label, count, pct in rows
            )
            stats_html = (
                '<div class="pcvs-panel-title">Raster coverage</div>'
                '<table>'
                '<tr><th align="left">Class</th><th align="right">Pixels</th>'
                '<th align="right">Share</th></tr>'
                + body +
                '</table>'
            )
            # Empty FeatureGroup just to get a checkbox in the LayerControl
            stats_group = folium.FeatureGroup(name=stats_checkbox_name, show=True)
            stats_group.add_to(m)
            _script_macro(m, """
                (function() {
                    var map = {{ this._parent.get_name() }};
                    var name = __NAME__;
                    var colorStatsControl = L.control({position: 'bottomleft'});
                    colorStatsControl.onAdd = function() {
                        var div = L.DomUtil.create('div', 'pcvs-panel color-stats-control');
                        div.innerHTML = __HTML__;
                        L.DomEvent.disableClickPropagation(div);
                        return div;
                    };
                    colorStatsControl.addTo(map);
                    map.on('overlayremove', function(e) {
                        if (e.name === name) map.removeControl(colorStatsControl);
                    });
                    map.on('overlayadd', function(e) {
                        if (e.name === name) colorStatsControl.addTo(map);
                    });
                })();
            """, name=json.dumps(stats_checkbox_name, ensure_ascii=False),
                 html=json.dumps(stats_html, ensure_ascii=False))

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    _add_map_theme(m)

    # Add measure tool
    plugins.MeasureControl().add_to(m)

    # Fit bounds to raster extent (tightest framing around interpolated area)
    raster_bounds = None
    if geotiff_path and os.path.exists(geotiff_path):
        try:
            raster_bounds = get_geotiff_bounds(geotiff_path)
        except:
            pass
    if raster_bounds:
        m.fit_bounds(raster_bounds)
    elif full_bounds:
        m.fit_bounds(full_bounds)

    # After fitBounds so the first label pass sees the published framing.
    _add_graticule(m, interval=30)

    _add_export_api(
        m,
        raster_bounds=raster_bounds,
        full_bounds=full_bounds or raster_bounds,
        export_basename=os.path.splitext(os.path.basename(output_file))[0],
    )
    
    print(f"Saving map to {output_file}...")
    _save_map(m, output_file)
    print(f"Map saved successfully! Open {output_file} in your browser.")
    
    return output_file


def _save_map(map_obj, output_file, attempts=8):
    """Write a Folium map, retrying when Windows has the destination locked."""
    tmp = output_file + '.tmp'
    last_err = None
    for attempt in range(attempts):
        try:
            map_obj.save(tmp)
            os.replace(tmp, output_file)
            return
        except OSError as err:
            last_err = err
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            time.sleep(0.45 * (attempt + 1))
    raise last_err

PDF_SCOPES = ('full', 'raster')

PDF_SCOPE_LABELS = {
    'full': 'entire map',
    'raster': 'raster area',
}


class PdfExporter:
    """Render generated maps to PDF, reusing one headless browser.

    Framing, chrome hiding and page geometry all come from ``window.PCVS``, the
    same API the in-browser export button uses. Playwright then prints the
    framed page: CSS filters are off while exporting, so data points stay
    vector circles instead of one small bitmap each.
    """

    def __init__(self, wait_seconds=1.5):
        self.wait_seconds = wait_seconds
        self._playwright = None
        self._browser = None
        self.available = False

    def start(self):
        """Launch the headless browser. Leaves ``available`` False on failure."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print("PDF export skipped (pip install playwright && playwright install chromium)")
            return self
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch()
            self.available = True
        except Exception as e:
            print(f"PDF export skipped (could not start Chromium: {e})")
            self.close()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        self.available = False

    def export(self, html_path, pdf_path, scope='full'):
        """Write one PDF of ``html_path`` framed on ``scope``."""
        if not self.available or not os.path.exists(html_path):
            return False
        pdf_path_abs = os.path.abspath(pdf_path)
        out_dir = os.path.dirname(pdf_path_abs)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        page = None
        try:
            page = self._browser.new_page(viewport={'width': 1280, 'height': 900})
            page.goto('file://' + os.path.abspath(html_path),
                      wait_until='load', timeout=60000)
            page.wait_for_function('window.PCVS !== undefined', timeout=30000)

            # The page is sized to the region being exported, so the map fills
            # it with no margins to trim afterwards.
            size = page.evaluate('scope => window.PCVS.exportSize(scope)', scope)
            page.set_viewport_size({
                'width': int(size['width']),
                'height': int(size['height']),
            })
            page.evaluate(
                'scope => window.PCVS.beginExport({scope: scope, resize: false})',
                scope,
            )
            # Drop CSS filters that Chromium would flatten into one bitmap
            # per CircleMarker. The same rules live in the map theme; this
            # keeps older generated HTML exportable as vector points too.
            page.add_style_tag(content=(
                '.pcvs-exporting path.pcvs-point,'
                '.pcvs-exporting .climate-point-icon svg'
                '{ filter: none !important; }'
                '.pcvs-exporting .graticule-label span'
                '{ text-shadow: none !important; }'
                '.pcvs-exporting .pcvs-panel,'
                '.pcvs-exporting .leaflet-bar'
                '{ box-shadow: none !important; }'
            ))
            page.evaluate('() => window.PCVS.settled()')
            page.wait_for_timeout(int(self.wait_seconds * 1000))

            # Keep screen styling: print media would drop the map background.
            page.emulate_media(media='screen')
            page.pdf(
                path=pdf_path_abs,
                width=f"{size['width']}px",
                height=f"{size['height']}px",
                margin={'top': '0', 'right': '0', 'bottom': '0', 'left': '0'},
                print_background=True,
            )
            print(f"  PDF saved ({PDF_SCOPE_LABELS.get(scope, scope)}): {pdf_path}")
            return True
        except Exception as e:
            print(f"  PDF failed for {html_path} [{scope}]: {e}")
            return False
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass

    def export_all(self, html_path, scopes=PDF_SCOPES):
        """Write one PDF per scope next to ``html_path``. Returns paths written."""
        written = []
        for scope in scopes:
            pdf_path = pdf_path_for(html_path, scope)
            if self.export(html_path, pdf_path, scope=scope):
                written.append(pdf_path)
        return written


def pdf_path_for(html_path, scope):
    """Path of the PDF holding ``html_path`` framed on ``scope``."""
    return f"{os.path.splitext(html_path)[0]}_{scope}.pdf"

def discover_geojson_datasets(geojson_dir='GEOJSON'):
    """
    Find all point+costa dataset pairs in geojson_dir.
    - Point files: *.geojson that do NOT end with _costa.geojson
    - Coast files: {base}_costa.geojson
    Returns list of (base_name, points_path, coast_path).
    """
    if not os.path.isdir(geojson_dir):
        return []
    pairs = []
    for f in sorted(os.listdir(geojson_dir)):
        if not f.endswith('.geojson'):
            continue
        if f.endswith('_costa.geojson'):
            continue
        base = f[:-len('.geojson')]
        coast_file = base + '_costa.geojson'
        coast_path = os.path.join(geojson_dir, coast_file)
        if os.path.isfile(coast_path):
            points_path = os.path.join(geojson_dir, f)
            pairs.append((base, points_path, coast_path))
    return pairs

def filter_datasets(datasets, requested):
    """Keep only the datasets named in ``requested``.

    Names may be given with or without the '_ma' suffix, so both '110' and
    '110_ma' select the 110_ma dataset. Returns all datasets when nothing is
    requested.
    """
    if not requested:
        return datasets

    selected = []
    missing = []
    for wanted in requested:
        key = wanted.strip().lower()
        candidates = [key] if key.endswith('_ma') else [key, f'{key}_ma']
        match = next((d for d in datasets if d[0].lower() in candidates), None)
        if match is None:
            missing.append(wanted)
        elif match not in selected:
            selected.append(match)

    if missing:
        available = ', '.join(d[0] for d in datasets)
        print(f"No dataset matches {missing} (available: {available})")
    return selected

def _extract_age_sort_key(filename):
    """Extract numeric age from filename for sorting, e.g. 'map_65_ma_...' -> 65."""
    import re
    m = re.search(r'_(\d+)_ma', filename)
    return int(m.group(1)) if m else 0

def _extract_method_tag(filename):
    """Return the neighbor search backend encoded in a generated filename, if any."""
    for method in (METHOD_KDTREE, METHOD_BRUTE):
        if f'_{method}' in filename:
            return method
    return ''

def _map_caption(filename, method):
    """Caption naming the interpolation a generated map came from."""
    if '_knn_idw' in filename:
        caption = 'KNN + IDW interpolation'
    elif '_idw' in filename:
        caption = 'IDW interpolation'
    else:
        caption = 'Original raster'
    return f'{caption} · {method_label(method)}' if method else caption


def generate_index_html(dir_knn_idw, dir_idw, output='index.html'):
    """Generate the viewer shell that switches between the generated maps."""
    maps_list = []
    folder = dir_knn_idw
    if os.path.isdir(folder):
        htmls = sorted(
            [f for f in os.listdir(folder) if f.endswith('.html')],
            key=_extract_age_sort_key
        )
        for h in htmls:
            age = _extract_age_sort_key(h)
            html_path = f"{folder}/{h}"
            pdfs = {}
            for scope in PDF_SCOPES:
                candidate = pdf_path_for(html_path, scope)
                if os.path.isfile(candidate):
                    pdfs[scope] = candidate
            comparison_path = f"COMPARISON/comparison_report_{age}.html"
            method = _extract_method_tag(h)
            maps_list.append({
                'path': html_path,
                'age': age,
                'label': f"{age} Ma",
                'method': method,
                'caption': _map_caption(h, method),
                'pdf': pdfs,
                'comparison': comparison_path if os.path.isfile(comparison_path) else '',
            })

    if not maps_list:
        return

    html = (INDEX_TEMPLATE
            .replace('__MAPS__', json.dumps(maps_list, ensure_ascii=False))
            .replace('__BACKGROUND__', MAP_BACKGROUND))

    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Index page generated: {output}")


INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PCVS · Paleoclimate Visualization System</title>
<style>
  :root {
    --ink: #0f172a;
    --muted: #64748b;
    --line: rgba(148, 163, 184, 0.28);
    --accent: #2f6fed;
    --header-1: #131c2e;
    --header-2: #0a1120;
    --on-header: #e8edf6;
    --on-header-dim: #93a3bd;
    --radius: 10px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: var(--ink);
    background: __BACKGROUND__;
    display: flex;
    flex-direction: column;
    -webkit-font-smoothing: antialiased;
  }

  header {
    flex: none;
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 8px 16px;
    background: linear-gradient(180deg, var(--header-1), var(--header-2));
    color: var(--on-header);
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    box-shadow: 0 1px 12px rgba(8, 15, 30, 0.28);
    position: relative;
    z-index: 20;
    min-width: 0;
  }

  .brand {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 1px;
    flex: none;
    min-width: 0;
  }
  .brand-mark {
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.14em;
    line-height: 1.1;
  }
  .brand-meta {
    display: flex;
    align-items: baseline;
    gap: 7px;
    min-width: 0;
    max-width: 280px;
  }
  .context-age {
    font-size: 11px;
    font-weight: 650;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
    flex: none;
  }
  .context-sub {
    font-size: 10.5px;
    color: var(--on-header-dim);
    letter-spacing: 0.01em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .divider {
    width: 1px;
    height: 26px;
    background: rgba(255, 255, 255, 0.12);
    flex: none;
  }

  .field { display: flex; align-items: center; gap: 8px; flex: none; }
  .field-label {
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--on-header-dim);
  }

  .btn, .toggle, .combo-btn {
    font: inherit;
    font-size: 12px;
    color: var(--on-header);
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 8px;
    transition: background 0.15s ease, border-color 0.15s ease, opacity 0.15s ease;
  }

  /* Age picker. A native select renders its list with the platform's own
     styling, which has nothing to do with the rest of the interface. */
  .combo { position: relative; flex: none; }
  .combo-btn {
    display: inline-flex;
    align-items: center;
    justify-content: space-between;
    gap: 9px;
    min-width: 96px;
    padding: 5px 9px;
    font-variant-numeric: tabular-nums;
    cursor: pointer;
  }
  .combo-btn:hover { background: rgba(255, 255, 255, 0.13); }
  .combo.open .combo-btn {
    background: rgba(255, 255, 255, 0.15);
    border-color: rgba(255, 255, 255, 0.28);
  }
  .combo-caret {
    width: 0;
    height: 0;
    border-left: 3.5px solid transparent;
    border-right: 3.5px solid transparent;
    border-top: 4px solid var(--on-header-dim);
    transition: transform 0.18s ease;
  }
  .combo.open .combo-caret { transform: rotate(180deg); }

  .combo-menu {
    display: none;
    position: absolute;
    top: calc(100% + 7px);
    left: 0;
    min-width: 128px;
    max-height: 46vh;
    overflow-y: auto;
    padding: 5px;
    background: rgba(255, 255, 255, 0.97);
    border: 1px solid rgba(15, 23, 42, 0.1);
    border-radius: var(--radius);
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06),
                0 18px 40px -16px rgba(15, 23, 42, 0.45);
    -webkit-backdrop-filter: saturate(160%) blur(10px);
    backdrop-filter: saturate(160%) blur(10px);
    z-index: 50;
    scrollbar-width: thin;
    scrollbar-color: rgba(15, 23, 42, 0.2) transparent;
  }
  .combo.open .combo-menu { display: block; }
  .combo-menu::-webkit-scrollbar { width: 9px; }
  .combo-menu::-webkit-scrollbar-track { background: transparent; }
  .combo-menu::-webkit-scrollbar-thumb {
    background: rgba(15, 23, 42, 0.18);
    background-clip: content-box;
    border: 3px solid transparent;
    border-radius: 999px;
  }
  .combo-menu::-webkit-scrollbar-thumb:hover {
    background: rgba(15, 23, 42, 0.32);
    background-clip: content-box;
  }
  .combo-opt {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 9px;
    border-radius: 7px;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    color: var(--ink);
    white-space: nowrap;
    cursor: pointer;
    transition: background 0.12s ease;
  }
  .combo-opt:hover { background: rgba(15, 23, 42, 0.06); }
  .combo-opt[aria-selected="true"] { color: var(--accent); font-weight: 650; }
  .combo-opt-mark {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: currentColor;
    opacity: 0;
    flex: none;
  }
  .combo-opt[aria-selected="true"] .combo-opt-mark { opacity: 1; }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 11px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
  }
  .btn:hover:not(:disabled) { background: rgba(255, 255, 255, 0.14); }
  .btn:disabled { opacity: 0.38; cursor: not-allowed; }
  .btn[hidden] { display: none; }
  .btn-icon { padding: 6px 8px; font-size: 13px; line-height: 1; }
  .btn-primary {
    background: var(--accent);
    border-color: transparent;
    color: #fff;
  }
  .btn-primary:hover:not(:disabled) { background: #2a62d4; }
  .btn.busy { pointer-events: none; }
  :focus-visible { outline: 2px solid #7aa5ff; outline-offset: 2px; }

  /* The export button keeps its label and swaps only the icon, so the header
     layout does not shift while a PDF renders. */
  .btn-ico {
    display: inline-flex;
    width: 13px;
    height: 13px;
    flex: none;
  }
  .btn-ico svg { width: 13px; height: 13px; display: block; }
  .btn-ico i { display: none; }
  .btn.busy .btn-ico svg { display: none; }
  .btn.busy .btn-ico i {
    display: block;
    width: 13px;
    height: 13px;
    box-sizing: border-box;
    border: 2px solid rgba(255, 255, 255, 0.35);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Age timeline. flex-shrink is 0 so showing Comparison on 105/115 Ma
     cannot steal width from the slider. Extra chrome comes out of .spacer. */
  .timeline {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1 0 260px;
    max-width: 460px;
  }
  .timeline input[type="range"] {
    flex: 1;
    min-width: 90px;
    height: 22px;
    accent-color: var(--accent);
    background: transparent;
    cursor: pointer;
  }
  .timeline-hint {
    font-size: 9.5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--on-header-dim);
    white-space: nowrap;
  }

  .toggle {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 10px;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }
  .toggle:hover { background: rgba(255, 255, 255, 0.12); }
  .toggle input { margin: 0; accent-color: var(--accent); cursor: pointer; }

  .spacer { flex: 1 1 0; min-width: 8px; }
  .header-actions {
    display: flex;
    align-items: center;
    gap: 10px;
    flex: none;
  }
  /* Keep the Comparison slot on every age. display:none would hand that
     width back to the timeline when leaving 105/115 Ma. */
  #comparisonBtn.is-absent {
    visibility: hidden;
    pointer-events: none;
  }

  /* Map area */
  main { position: relative; flex: 1; min-height: 0; }
  iframe { width: 100%; height: 100%; border: none; display: block; overflow: hidden; }

  .loader {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    background: __BACKGROUND__;
    color: var(--muted);
    font-size: 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease;
  }
  .loader.on { opacity: 1; }
  .loader .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent);
    margin: 0 auto 10px;
    animation: pulse 1.1s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { transform: scale(0.7); opacity: 0.35; }
    50% { transform: scale(1.25); opacity: 1; }
  }

  /* Toast */
  .toast {
    position: absolute;
    left: 50%;
    bottom: 22px;
    transform: translate(-50%, 14px);
    padding: 9px 16px;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.94);
    color: #f1f5f9;
    font-size: 12px;
    box-shadow: 0 10px 30px -10px rgba(15, 23, 42, 0.6);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease, transform 0.2s ease;
    z-index: 10;
  }
  .toast.on { opacity: 1; transform: translate(-50%, 0); }
  .toast.error { background: #7f1d1d; }

  @media (max-width: 1200px) {
    .timeline-hint { display: none; }
    .context-sub { display: none; }
    .brand-meta { max-width: none; }
  }
  @media (max-width: 900px) {
    header { flex-wrap: wrap; gap: 10px 14px; }
    .divider { display: none; }
    .timeline { order: 10; flex-basis: 100%; max-width: none; }
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span class="brand-mark">PCVS</span>
    <span class="brand-meta">
      <span class="context-age" id="ctxAge"></span>
      <span class="context-sub" id="ctxSub"></span>
    </span>
  </div>

  <div class="divider"></div>

  <div class="field">
    <span class="field-label">Age</span>
    <div class="combo" id="ageCombo">
      <button type="button" class="combo-btn" id="ageBtn" aria-haspopup="listbox"
              aria-expanded="false" aria-label="Reconstruction age">
        <span id="ageBtnLabel"></span>
        <span class="combo-caret" aria-hidden="true"></span>
      </button>
      <div class="combo-menu" id="ageMenu" role="listbox"
           aria-label="Reconstruction age"></div>
    </div>
  </div>

  <div class="timeline">
    <span class="timeline-hint">Recent</span>
    <button class="btn btn-icon" id="prevBtn" title="Younger reconstruction (←)"
            aria-label="Younger age">&#9664;</button>
    <input type="range" id="ageRange" min="0" max="0" step="1" list="ageTicks"
           aria-label="Age timeline">
    <datalist id="ageTicks"></datalist>
    <button class="btn btn-icon" id="nextBtn" title="Older reconstruction (→)"
            aria-label="Older age">&#9654;</button>
    <span class="timeline-hint">Ancient</span>
  </div>

  <div class="spacer"></div>

  <div class="header-actions">
    <button type="button" class="btn is-absent" id="comparisonBtn"
            aria-hidden="true" tabindex="-1">Comparison</button>
    <label class="toggle" id="scopeToggle"
           title="Export only the area covered by the interpolated raster instead of the whole map">
      <input type="checkbox" id="rasterOnly">
      <span>Raster area only</span>
    </label>
    <button class="btn btn-primary" id="pdfBtn" title="Export the map as shown (P)">
      <span class="btn-ico" aria-hidden="true">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M8 2v8M4.6 6.9 8 10.3l3.4-3.4M2.8 13.4h10.4"/>
        </svg>
        <i></i>
      </span><span>PDF</span>
    </button>
  </div>
</header>

<main>
  <iframe id="mapFrame" title="Paleogeographic map"></iframe>
  <div class="loader" id="loader"><div><span class="dot"></span>Loading map</div></div>
  <div class="toast" id="toast" role="status" aria-live="polite"></div>
</main>

<script>
var MAPS = __MAPS__;
var STORAGE_KEY = 'pcvs:age';

var frame = document.getElementById('mapFrame');
var combo = document.getElementById('ageCombo');
var comboBtn = document.getElementById('ageBtn');
var comboLabel = document.getElementById('ageBtnLabel');
var comboMenu = document.getElementById('ageMenu');
var ctxAge = document.getElementById('ctxAge');
var ctxSub = document.getElementById('ctxSub');
var range = document.getElementById('ageRange');
var ticks = document.getElementById('ageTicks');
var prevBtn = document.getElementById('prevBtn');
var nextBtn = document.getElementById('nextBtn');
var pdfBtn = document.getElementById('pdfBtn');
var rasterOnly = document.getElementById('rasterOnly');
var comparisonBtn = document.getElementById('comparisonBtn');
var loader = document.getElementById('loader');
var toast = document.getElementById('toast');

var current = 0;
var toastTimer = null;
var ageOptions = [];

function showToast(message, isError) {
  toast.textContent = message;
  toast.classList.toggle('error', !!isError);
  toast.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(function() { toast.classList.remove('on'); },
                          isError ? 6000 : 3000);
}

function indexForAge(age) {
  for (var i = 0; i < MAPS.length; i++) {
    if (String(MAPS[i].age) === String(age)) return i;
  }
  return -1;
}

function initialIndex() {
  var fromHash = (location.hash.match(/age=(\\d+)/) || [])[1];
  var stored = null;
  try { stored = localStorage.getItem(STORAGE_KEY); } catch (e) {}
  var idx = indexForAge(fromHash);
  if (idx < 0) idx = indexForAge(stored);
  return idx < 0 ? 0 : idx;
}

function clamp(idx) {
  return Math.max(0, Math.min(MAPS.length - 1, idx));
}

/* Update the header for an age without reloading the map, so dragging the
   timeline stays responsive instead of loading every age it passes over. */
function preview(idx) {
  var at = clamp(idx);
  var entry = MAPS[at];
  comboLabel.textContent = entry.label;
  ctxAge.textContent = entry.label;
  ctxSub.textContent = entry.caption || '';
  ctxSub.title = entry.caption || '';
  ageOptions.forEach(function(option, i) {
    option.setAttribute('aria-selected', i === at ? 'true' : 'false');
  });
  range.value = String(at);
  prevBtn.disabled = at === 0;
  nextBtn.disabled = at === MAPS.length - 1;
}

function openCombo() {
  combo.classList.add('open');
  comboBtn.setAttribute('aria-expanded', 'true');
  var selected = ageOptions[current];
  if (selected) selected.scrollIntoView({block: 'nearest'});
}

function closeCombo() {
  combo.classList.remove('open');
  comboBtn.setAttribute('aria-expanded', 'false');
}

function selectMap(idx, pushHash) {
  current = clamp(idx);
  var entry = MAPS[current];
  preview(current);

  loader.classList.add('on');
  pdfBtn.disabled = true;
  frame.src = entry.path;

  var hasComparison = Boolean(entry.comparison);
  comparisonBtn.classList.toggle('is-absent', !hasComparison);
  comparisonBtn.setAttribute('aria-hidden', hasComparison ? 'false' : 'true');
  comparisonBtn.tabIndex = hasComparison ? 0 : -1;
  if (!hasComparison && document.activeElement === comparisonBtn) comparisonBtn.blur();
  if (pushHash !== false) {
    history.replaceState(null, '', '#age=' + entry.age);
  }
  try { localStorage.setItem(STORAGE_KEY, String(entry.age)); } catch (e) {}
}

function currentScope() {
  return rasterOnly.checked ? 'raster' : 'full';
}

function downloadBlob(blob, filename) {
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(function() { URL.revokeObjectURL(url); }, 4000);
}

/* Pre-rendered PDFs are the fallback for browsers that cannot reach into the
   frame (for instance when the site is opened straight from disk). They show
   the map with every layer on, which is how it loads. */
function downloadPrerendered(scope) {
  var entry = MAPS[current];
  var href = (entry.pdf || {})[scope] || (entry.pdf || {}).full;
  if (!href) {
    showToast('No PDF available for this map.', true);
    return false;
  }
  var a = document.createElement('a');
  a.href = href;
  a.download = href.split('/').pop();
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  return true;
}

/* The frame renders itself. A full-map download is drawn as vectors so the
   data points stay editable; the raster-only crop may stay a bitmap. */
function exportLive(scope) {
  var target = frame.contentWindow;
  if (!target) return Promise.reject(new Error('Map frame is not ready'));
  return new Promise(function(resolve, reject) {
    var id = 'export-' + Date.now();
    var timer = setTimeout(function() {
      window.removeEventListener('message', onMessage);
      reject(new Error('export timed out'));
    }, 60000);

    function onMessage(event) {
      var data = event.data;
      if (!data || data.id !== id) return;
      if (data.pcvs === 'export-result') {
        clearTimeout(timer);
        window.removeEventListener('message', onMessage);
        downloadBlob(new Blob([data.buffer], {type: 'application/pdf'}), data.filename);
        resolve();
      } else if (data.pcvs === 'export-error') {
        clearTimeout(timer);
        window.removeEventListener('message', onMessage);
        reject(new Error(data.message));
      }
    }

    window.addEventListener('message', onMessage);
    target.postMessage({pcvs: 'export', scope: scope, id: id}, '*');
  });
}

function exportPdf() {
  if (pdfBtn.disabled || pdfBtn.classList.contains('busy')) return;
  var scope = currentScope();
  pdfBtn.classList.add('busy');
  showToast('Rendering the ' + (scope === 'raster' ? 'raster area' : 'entire map') + '…');

  exportLive(scope).then(function() {
    showToast('PDF exported.');
  }).catch(function(err) {
    if (downloadPrerendered(scope)) {
      showToast('Live export unavailable (' + err.message +
                '); downloaded the pre-rendered PDF instead.', true);
    }
  }).finally(function() {
    pdfBtn.classList.remove('busy');
  });
}

MAPS.forEach(function(entry, i) {
  var option = document.createElement('div');
  option.className = 'combo-opt';
  option.setAttribute('role', 'option');
  option.setAttribute('aria-selected', 'false');
  option.innerHTML = '<span class="combo-opt-mark"></span>';
  option.appendChild(document.createTextNode(entry.label));
  option.addEventListener('click', function() {
    closeCombo();
    selectMap(i);
  });
  comboMenu.appendChild(option);
  ageOptions.push(option);

  var tick = document.createElement('option');
  tick.value = String(i);
  tick.label = entry.label;
  ticks.appendChild(tick);
});
range.max = String(MAPS.length - 1);

comboBtn.addEventListener('click', function(event) {
  event.stopPropagation();
  if (combo.classList.contains('open')) closeCombo(); else openCombo();
});
comboBtn.addEventListener('keydown', function(event) {
  if (event.key === 'Escape') {
    closeCombo();
  } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault();
    event.stopPropagation();
    if (!combo.classList.contains('open')) {
      openCombo();
      return;
    }
    selectMap(current + (event.key === 'ArrowDown' ? 1 : -1));
    ageOptions[current].scrollIntoView({block: 'nearest'});
  }
});
document.addEventListener('click', function(event) {
  if (!combo.contains(event.target)) closeCombo();
});
range.addEventListener('input', function() { preview(Number(this.value)); });
range.addEventListener('change', function() { selectMap(Number(this.value)); });
prevBtn.addEventListener('click', function() { selectMap(current - 1); });
nextBtn.addEventListener('click', function() { selectMap(current + 1); });
pdfBtn.addEventListener('click', exportPdf);
comparisonBtn.addEventListener('click', function() {
  var entry = MAPS[current];
  if (entry.comparison) window.open(entry.comparison, '_blank');
});
frame.addEventListener('load', function() {
  loader.classList.remove('on');
  pdfBtn.disabled = false;
});
window.addEventListener('hashchange', function() {
  var idx = indexForAge((location.hash.match(/age=(\\d+)/) || [])[1]);
  if (idx >= 0 && idx !== current) selectMap(idx, false);
});

document.addEventListener('keydown', function(event) {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  var tag = (event.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
  if (event.key === 'ArrowLeft') { selectMap(current - 1); event.preventDefault(); }
  else if (event.key === 'ArrowRight') { selectMap(current + 1); event.preventDefault(); }
  else if (event.key === 'p' || event.key === 'P') { exportPdf(); }
});

selectMap(initialIndex(), false);
</script>
</body>
</html>"""

def _print_metrics_summary(total_elapsed, dataset_metrics, rss_start, rss_end, rss_peak):
    """Print a summary of execution times and RAM usage per dataset and overall."""
    print("\n" + "=" * 60)
    print("Timing and memory summary")
    print("=" * 60)
    print(f"Total elapsed: {_format_duration(total_elapsed)}")
    if rss_start is not None and rss_end is not None:
        print(f"RAM (RSS) at start: {_format_bytes(rss_start)}")
        print(f"RAM (RSS) at end:   {_format_bytes(rss_end)} "
              f"({_format_bytes(rss_end - rss_start, signed=True)})")
    if rss_peak is not None:
        print(f"RAM (RSS) peak:     {_format_bytes(rss_peak)}")
    else:
        print("RAM (RSS): unavailable on this platform (install psutil for memory metrics)")
    for entry in dataset_metrics:
        print(f"\n{entry['base']} — {_format_metrics(entry['total'], entry['rss_delta'], entry['rss_peak'])}")
        for step in entry['steps']:
            print(f"  [{_format_metrics(step['elapsed'], step['rss_delta'], step['rss_peak'])}] {step['label']}")


def _use_utf8_console():
    """Let progress output print non-ASCII characters on any console.

    Windows terminals default to a legacy code page that cannot encode the
    characters used in the metrics lines, which would abort the run partway
    through with a UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


def main():
    """Generate maps for every point+costa dataset found in GEOJSON/."""
    _use_utf8_console()
    _RSS_SAMPLER.start()
    script_start = time.perf_counter()
    script_rss_start = _RSS_SAMPLER.sample()
    script_watcher = _RSS_SAMPLER.watch(script_rss_start)
    dataset_metrics = []

    parser = argparse.ArgumentParser(description='Generate paleogeographic maps (IDW and KNN+IDW) for all datasets in GEOJSON/.')
    parser.add_argument('--power', type=float, required=True,
                        help='Power parameter for IDW and KNN (e.g. 4.0)')
    parser.add_argument('--gradient-sharp', type=float, default=2.5,
                        help='Gradient sharpening factor for color transitions (default: 2.5, higher = more abrupt)')
    parser.add_argument('--geojson-dir', default='GEOJSON',
                        help='Directory containing point and coastline GeoJSON files (default: GEOJSON)')
    parser.add_argument('--pdf', action='store_true',
                        help='Export each map to PDF, one framed on the entire map and one on '
                             'the raster area (requires playwright)')
    parser.add_argument('--map', dest='maps', nargs='+', metavar='DATASET',
                        help='Render only the given dataset(s), e.g. --map 110 or --map 110 115 (default: all)')
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument('--brute', dest='method', action='store_const', const=METHOD_BRUTE,
                              help='Find KNN/IDW neighbors by brute force (full distance matrix) [default]')
    method_group.add_argument('--kdtree', dest='method', action='store_const', const=METHOD_KDTREE,
                              help='Find KNN/IDW neighbors with a k-d tree')
    parser.set_defaults(method=METHOD_BRUTE)
    args = parser.parse_args()
    power = args.power
    gradient_sharp = args.gradient_sharp
    method = args.method
    params_suffix = f'_power{power}_gradient_sharp{gradient_sharp}_{method}'

    datasets = discover_geojson_datasets(args.geojson_dir)
    if not datasets:
        print(f"No datasets found in {args.geojson_dir}/ (need both X.geojson and X_costa.geojson for each X).")
        return
    datasets = filter_datasets(datasets, args.maps)
    if not datasets:
        return
    print(f"Found {len(datasets)} dataset(s): {[b for b, _, _ in datasets]}")
    print(f"Neighbor search method: {method_label(method)} ({method})")

    dir_geotiffs = 'GENERATED_GEOTIFFS'
    dir_idw_maps = 'GENERATED_IDW_MAPS'
    dir_knn_idw_maps = 'GENERATED_KNN_IDW_MAPS'
    for d in (dir_geotiffs, dir_idw_maps, dir_knn_idw_maps):
        os.makedirs(d, exist_ok=True)

    # One browser serves every export; launching one per PDF dominated the run.
    pdf_exporter = PdfExporter()
    if args.pdf:
        pdf_exporter.start()

    generated = []
    for base, points_path, coast_path in datasets:
        dataset_start = time.perf_counter()
        dataset_rss_start = _RSS_SAMPLER.sample()
        dataset_watcher = _RSS_SAMPLER.watch(dataset_rss_start)
        step_metrics = []

        def _record_step(timer):
            step_metrics.append({
                'label': timer.label,
                'elapsed': timer.elapsed,
                'rss_delta': timer.rss_delta,
                'rss_peak': timer.rss_peak,
            })

        age_label = base.replace('_ma', '').replace('_', ' ') + ' Ma'
        print("\n" + "=" * 60)
        print(f"DATASET: {base}")
        print("=" * 60)

        with StepTimer("Load GeoJSON") as t:
            points_data = load_geojson(points_path)
            coastline_data = load_geojson(coast_path)
        _record_step(t)

        n_pts = len(points_data.get('features', []))
        n_coast = len(coastline_data.get('features', []))
        print(f"Loaded {n_pts} points from {points_path}")
        print(f"Loaded {n_coast} coastline features from {coast_path}")
        if n_pts == 0:
            print(f"Skipping {base}: no point features.")
            _RSS_SAMPLER.release(dataset_watcher)
            continue

        with StepTimer("Paleo reference frame correction") as t:
            points_data = apply_paleo_reference_frame_correction(points_data, base)
            coastline_data = apply_paleo_reference_frame_correction(coastline_data, base)
        _record_step(t)

        original_raster_path = os.path.join('GEOTIFF', f'{base}_idw.tif')
        if not os.path.exists(original_raster_path) and base.endswith('_ma'):
            alt = os.path.join('GEOTIFF', f'{base.replace("_ma", "")}_idw.tif')
            if os.path.exists(alt):
                original_raster_path = alt
        idw_only_raster_path = os.path.join(dir_geotiffs, f'{base}_idw_only{params_suffix}.tif')
        idw_raster_path = os.path.join(dir_geotiffs, f'{base}_knn_idw{params_suffix}.tif')
        map1_file = os.path.join(dir_idw_maps, f'map_{base}_original.html')
        map_idw_file = os.path.join(dir_idw_maps, f'map_{base}_idw{params_suffix}.html')
        map_knn_idw_file = os.path.join(dir_knn_idw_maps, f'map_{base}_knn_idw{params_suffix}.html')
        raster_overlay_idw_png = os.path.join(dir_idw_maps, f'raster_overlay_{base}_idw{params_suffix}.png')
        raster_overlay_knn_idw_png = os.path.join(dir_knn_idw_maps, f'raster_overlay_{base}_knn_idw{params_suffix}.png')

        if os.path.exists(original_raster_path):
            print("\nGenerating Map: Original Data (with original raster)")
            with StepTimer("Map: Original Data") as t:
                create_map(
                    points_data=points_data,
                    coastline_data=coastline_data,
                    geotiff_path=original_raster_path,
                    output_file=map1_file,
                    raster_img_path=os.path.join(dir_idw_maps, f'raster_overlay_{base}_original.png'),
                    gradient_sharp=gradient_sharp,
                    age_label=age_label,
                    map_subtitle='Original raster',
                )
            _record_step(t)
            generated.append(map1_file)
            if pdf_exporter.available:
                with StepTimer("PDF: Original Data") as t:
                    pdf_exporter.export_all(map1_file)
                _record_step(t)
        else:
            print(f"Original raster not found ({original_raster_path}), skipping Original map.")

        print(f"\nExecuting IDW Interpolation (no KNN) — {method_label(method)}")
        points, values = extract_points_and_values(points_data)
        with StepTimer(f"IDW raster (no KNN, {method})") as t:
            create_idw_raster(
                points_data=points_data,
                points=points,
                values=values,
                output_path=idw_only_raster_path,
                resolution=0.1,
                power=power,
                method=method
            )
        _record_step(t)

        print(f"Generating Map: IDW only ({map_idw_file})")
        with StepTimer("Map: IDW only") as t:
            create_map(
                points_data=points_data,
                coastline_data=coastline_data,
                geotiff_path=idw_only_raster_path,
                output_file=map_idw_file,
                raster_img_path=raster_overlay_idw_png,
                gradient_sharp=gradient_sharp,
                color_stats_img_path=raster_overlay_idw_png,
                method=method,
                age_label=age_label,
                map_subtitle='IDW interpolation',
            )
        _record_step(t)
        generated.append(map_idw_file)
        if pdf_exporter.available:
            with StepTimer("PDF: IDW only") as t:
                pdf_exporter.export_all(map_idw_file)
            _record_step(t)

        print(f"Executing KNN Smoothing + IDW Interpolation — {method_label(method)}")
        with StepTimer(f"KNN smoothing ({method})") as t:
            knn_values = knn_smooth_values(points, values, k=8, power=power,
                                           exclude_self=True, method=method)
        _record_step(t)

        with StepTimer(f"IDW raster (KNN + IDW, {method})") as t:
            create_idw_raster(
                points_data=points_data,
                points=points,
                values=knn_values,
                output_path=idw_raster_path,
                resolution=0.1,
                power=power,
                method=method
            )
        _record_step(t)

        print(f"Generating Map: KNN + IDW ({map_knn_idw_file})")
        with StepTimer("Map: KNN + IDW") as t:
            create_map(
                points_data=points_data,
                coastline_data=coastline_data,
                geotiff_path=idw_raster_path,
                output_file=map_knn_idw_file,
                raster_img_path=raster_overlay_knn_idw_png,
                point_values_override=knn_values,
                gradient_sharp=gradient_sharp,
                color_stats_img_path=raster_overlay_knn_idw_png,
                method=method,
                age_label=age_label,
                map_subtitle='KNN + IDW interpolation',
            )
        _record_step(t)
        generated.append(map_knn_idw_file)
        if pdf_exporter.available:
            with StepTimer("PDF: KNN + IDW") as t:
                pdf_exporter.export_all(map_knn_idw_file)
            _record_step(t)

        dataset_total = time.perf_counter() - dataset_start
        dataset_rss_end = _RSS_SAMPLER.sample()
        dataset_rss_peak = _RSS_SAMPLER.release(dataset_watcher)
        dataset_rss_delta = (dataset_rss_end - dataset_rss_start
                             if dataset_rss_start is not None and dataset_rss_end is not None
                             else None)
        dataset_metrics.append({
            'base': base,
            'total': dataset_total,
            'rss_delta': dataset_rss_delta,
            'rss_peak': dataset_rss_peak,
            'steps': step_metrics,
        })
        print(f"  Dataset total: {_format_metrics(dataset_total, dataset_rss_delta, dataset_rss_peak)}")

    pdf_exporter.close()
    total_elapsed = time.perf_counter() - script_start

    print("\n" + "=" * 60)
    print("All maps generated successfully!")
    print("=" * 60)
    print(f"Power: {power}, Gradient sharp: {gradient_sharp}")
    print(f"Neighbor search method: {method_label(method)} ({method})")
    print(f"Datasets processed: {len(datasets)}")
    print(f"GeoTIFFs: {dir_geotiffs}/")
    print(f"IDW-only maps (HTML/PDF): {dir_idw_maps}/")
    print(f"KNN+IDW maps (HTML/PDF): {dir_knn_idw_maps}/")
    for f in generated:
        print(f"  {f}")

    with StepTimer("Generate index.html", indent=0) as t:
        generate_index_html(dir_knn_idw_maps, dir_idw_maps)

    script_rss_end = _RSS_SAMPLER.sample()
    # The kernel updates its high-water mark lazily, so a sampled value may top it.
    observed_peak = _RSS_SAMPLER.release(script_watcher)
    script_rss_peak = max([p for p in (_peak_rss(), observed_peak) if p is not None],
                          default=None)
    _RSS_SAMPLER.stop()
    _print_metrics_summary(total_elapsed, dataset_metrics,
                           script_rss_start, script_rss_end, script_rss_peak)

if __name__ == '__main__':
    main()
