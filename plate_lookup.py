"""PlateID for a present-day coordinate.

The add-points page has the same even-odd test in templates/add-point.html.
Both read plates/zahirovic2022-static.json. The smaller plate wins when two
polygons contain the point.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
POLYGONS = REPO_ROOT / 'plates' / 'zahirovic2022-static.json'


def _contains(rings: list, longitude: float, latitude: float) -> bool:
    crossings = 0
    for ring in rings:
        for index in range(len(ring) - 1):
            x1, y1 = ring[index]
            x2, y2 = ring[index + 1]
            if abs(x1 - x2) > 180:
                if x1 > x2:
                    x2 += 360
                else:
                    x1 += 360
                sample_x = longitude + 360 if longitude < 0 else longitude
            else:
                sample_x = longitude
            if (y1 > latitude) == (y2 > latitude):
                continue
            crossed_at = (x2 - x1) * (latitude - y1) / (y2 - y1) + x1
            if sample_x < crossed_at:
                crossings += 1
    return crossings % 2 == 1


def _bbox(rings: list) -> tuple[float, float, float, float]:
    longitudes = [point[0] for ring in rings for point in ring]
    latitudes = [point[1] for ring in rings for point in ring]
    west, east = min(longitudes), max(longitudes)
    for ring in rings:
        for index in range(len(ring) - 1):
            if abs(ring[index][0] - ring[index + 1][0]) > 180:
                west, east = -180.0, 180.0
                break
    return west, min(latitudes), east, max(latitudes)


def load_plates(path: Path | None = None) -> list[dict]:
    payload = json.loads((path or POLYGONS).read_text(encoding='utf-8'))
    plates = []
    for plate_id, area, rings in payload['plates']:
        plates.append({
            'id': plate_id,
            'area': area,
            'rings': rings,
            'bbox': _bbox(rings),
        })
    return plates


def plate_id_at(longitude: float, latitude: float, plates: list[dict] | None = None) -> int | None:
    """Return the PlateID whose present-day polygon contains this coordinate."""
    if plates is None:
        plates = load_plates()
    hits = []
    for plate in plates:
        west, south, east, north = plate['bbox']
        if longitude < west or longitude > east or latitude < south or latitude > north:
            continue
        if _contains(plate['rings'], longitude, latitude):
            hits.append(plate)
    if not hits:
        return None
    hits.sort(key=lambda plate: plate['area'])
    return hits[0]['id']
