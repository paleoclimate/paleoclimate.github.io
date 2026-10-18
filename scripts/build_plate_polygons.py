#!/usr/bin/env python3
"""Build the present-day plate polygons the add-points screen looks up.

Downloads the static polygons shipped with Zahirovic et al. 2022, dissolves
them by PLATEID1, and writes plates/zahirovic2022-static.json. Needs pyshp
and shapely, which the published site does not import:

    python -m pip install pyshp shapely
    python scripts/build_plate_polygons.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import shapefile
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / 'plates' / 'zahirovic2022-static.json'
SOURCE = 'https://repo.gplates.org/webdav/pmm/zahirovic2022/StaticPolygons.zip'
SIMPLIFY_DEGREES = 0.05
ROUND_DIGITS = 4

# Present-day localities whose plate is fixed by the static polygons.
ANCHORS = (
    (-40.25, -10.5, 201),
    (10.0, 5.5, 701),
    (1.57, 33.06, 714),
    (-65.732, -20.0, 201),
    (-72.6, 8.65, 2030),
    (28.9484, 28.0, 715),
    (-72.099, -49.044, 28800),
    (0.0, 0.0, 701),
    (151.0, -33.0, 801),
)


def _download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(SOURCE, destination)


def _dissolved(shp_path: Path) -> dict[int, object]:
    reader = shapefile.Reader(str(shp_path), encoding='latin1')
    try:
        grouped: dict[int, list] = defaultdict(list)
        for record in reader.shapeRecords():
            plate_id = int(record.record['PLATEID1'])
            geometry = shape(record.shape.__geo_interface__)
            if geometry.is_empty:
                continue
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            grouped[plate_id].append(geometry)
    finally:
        reader.close()
    dissolved = {}
    for plate_id, parts in grouped.items():
        geometry = unary_union(parts)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        dissolved[plate_id] = geometry
    return dissolved


def _rings(geometry) -> list[list[list[float]]]:
    if geometry.geom_type == 'Polygon':
        polygons = [geometry]
    elif geometry.geom_type == 'MultiPolygon':
        polygons = list(geometry.geoms)
    else:
        return []
    rings = []
    for polygon in polygons:
        for ring in [polygon.exterior, *polygon.interiors]:
            rounded = [
                [round(x, ROUND_DIGITS), round(y, ROUND_DIGITS)]
                for x, y in ring.coords
            ]
            compact = [rounded[0]]
            for point in rounded[1:]:
                if point != compact[-1]:
                    compact.append(point)
            if len(compact) >= 4:
                rings.append(compact)
    return rings


def _covers(dissolved: dict[int, object], longitude: float, latitude: float) -> int | None:
    point = Point(longitude, latitude)
    geometries = list(dissolved.values())
    plate_ids = list(dissolved.keys())
    tree = STRtree(geometries)
    hits = [plate_ids[index] for index in tree.query(point) if geometries[index].covers(point)]
    if not hits:
        return None
    hits.sort(key=lambda plate_id: dissolved[plate_id].area)
    return hits[0]


def build(zip_path: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary = Path(temporary_dir)
        if zip_path is None:
            print(f'Downloading {SOURCE}')
            zip_path = temporary / 'StaticPolygons.zip'
            _download(zip_path)
        extract_dir = temporary / 'extracted'
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        shapefile_path = next(extract_dir.rglob('*.shp'))
        dissolved = _dissolved(shapefile_path)
    for longitude, latitude, expected in ANCHORS:
        found = _covers(dissolved, longitude, latitude)
        if found != expected:
            raise SystemExit(
                f'{latitude}, {longitude} is plate {found}, expected {expected}'
            )
    plates = []
    for plate_id in sorted(dissolved):
        geometry = dissolved[plate_id].simplify(SIMPLIFY_DEGREES, preserve_topology=True)
        rings = _rings(geometry)
        if not rings:
            continue
        plates.append([plate_id, round(dissolved[plate_id].area, 4), rings])
    payload = {
        'model': 'Zahirovic et al. 2022 static polygons',
        'source': SOURCE,
        'plates': plates,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(',', ':')), encoding='utf-8')
    print(f'Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes, {len(plates)} plates)')
    sys.path.insert(0, str(REPO_ROOT))
    from plate_lookup import load_plates, plate_id_at

    loaded = load_plates(OUTPUT)
    for longitude, latitude, expected in ANCHORS:
        found = plate_id_at(longitude, latitude, loaded)
        if found != expected:
            raise SystemExit(
                f'lookup {latitude}, {longitude} is plate {found}, expected {expected}'
            )


if __name__ == '__main__':
    given = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    build(given)
