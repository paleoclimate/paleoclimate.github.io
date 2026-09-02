"""Static checks: generated files, viewer catalog, and renderer contracts."""

from __future__ import annotations

import re

from tests.helpers import (
    COMPARISON_AGES,
    COMPARISON_DIR,
    INDEX_HTML,
    KNN_DIR,
    LAYER_NAMES,
    REPO_ROOT,
    catalog_ages,
    embedded_basins,
    entry_for_age,
    idw_html_files,
    is_valid_pdf,
    knn_html_files,
    load_maps_catalog,
    overlay_png_for,
    pdf_files_for,
)


def test_viewer_exists():
    assert INDEX_HTML.is_file(), 'index.html is missing; run python render_map.py'


def test_maps_catalog_is_well_formed():
    maps = load_maps_catalog()
    ages = catalog_ages(maps)
    assert ages == sorted(ages), 'MAPS must be ordered from youngest to oldest'
    assert ages == sorted(set(ages)), 'MAPS must not repeat an age'
    assert ages[0] < ages[-1]

    for entry in maps:
        for key in ('path', 'age', 'label', 'method', 'caption', 'pdf', 'comparison'):
            assert key in entry, f'{entry} is missing {key}'
        assert entry['label'] == f"{entry['age']} Ma"
        assert entry['path'].startswith('GENERATED_KNN_IDW_MAPS/')
        assert entry['path'].endswith('.html')
        assert (REPO_ROOT / entry['path']).is_file(), f"Missing map {entry['path']}"
        assert isinstance(entry['pdf'], dict)
        for scope in ('full', 'raster'):
            href = entry['pdf'].get(scope)
            if href:
                assert (REPO_ROOT / href).is_file(), f'Missing PDF {href}'
                assert href.endswith(f'_{scope}.pdf')


def test_catalog_ages_match_knn_html_files():
    maps = load_maps_catalog()
    catalog_paths = {entry['path'] for entry in maps}
    disk_paths = {path.relative_to(REPO_ROOT).as_posix() for path in knn_html_files()}
    assert catalog_paths == disk_paths, (
        'index.html MAPS and GENERATED_KNN_IDW_MAPS/*.html drifted apart. '
        'Regenerate the viewer with python render_map.py'
    )


def test_every_knn_map_has_overlay_png():
    missing = []
    for html in knn_html_files():
        png = overlay_png_for(html)
        if not png.is_file() or png.stat().st_size < 1000:
            missing.append(str(png))
    assert not missing, 'Missing or tiny raster overlays:\n' + '\n'.join(missing)


def test_idw_maps_cover_the_same_ages():
    knn_ages = set(catalog_ages())
    idw_ages = set()
    for path in idw_html_files():
        match = re.search(r'map_(\d+)_ma', path.name)
        assert match, path.name
        idw_ages.add(int(match.group(1)))
    assert knn_ages, 'No KNN maps in the catalog'
    assert knn_ages <= idw_ages, f'IDW maps missing ages {sorted(knn_ages - idw_ages)}'


def test_comparison_links_only_for_supported_ages():
    maps = load_maps_catalog()
    for entry in maps:
        age = int(entry['age'])
        report = COMPARISON_DIR / f'comparison_report_{age}.html'
        if age in COMPARISON_AGES:
            assert entry['comparison'] == f'COMPARISON/comparison_report_{age}.html'
            assert report.is_file(), f'Missing comparison report for {age} Ma'
        else:
            assert entry['comparison'] in ('', None)
            if report.is_file():
                raise AssertionError(
                    f'{report.name} exists but index.html does not link it for {age} Ma'
                )


def test_generated_html_embeds_map_chrome_and_export_api():
    html_files = knn_html_files()
    assert html_files, 'No KNN maps to inspect'
    required = (
        'leaflet-container',
        'window.PCVS',
        'basin-filter-control',
        'color-stats-control',
        'leaflet-control-layers',
        'leaflet-control-zoom',
        'leaflet-control-measure',
        'leaflet-control-fullscreen',
        'graticule',
        'pcvs-export',
        *LAYER_NAMES,
    )
    broken = []
    for path in html_files:
        text = path.read_text(encoding='utf-8')
        missing = [token for token in required if token not in text]
        if missing:
            broken.append(f'{path.name}: {missing}')
    assert not broken, 'Generated maps are missing UI chrome:\n' + '\n'.join(broken)


def test_layer_names_match_renderer_constants():
    import render_map as renderer

    html = knn_html_files()[0].read_text(encoding='utf-8')
    for name in (
        renderer.LAYER_RASTER,
        renderer.LAYER_COASTLINES,
        renderer.LAYER_POINTS,
        renderer.LAYER_COLOR_STATS,
    ):
        assert name in html
        assert name in LAYER_NAMES


def test_idw_maps_also_embed_export_api():
    files = idw_html_files()
    assert files, 'No IDW maps found'
    for path in files:
        text = path.read_text(encoding='utf-8')
        assert 'window.PCVS' in text, path.name
        assert 'leaflet-container' in text, path.name


def test_viewer_shell_has_expected_controls():
    html = INDEX_HTML.read_text(encoding='utf-8')
    for token in (
        'id="ageCombo"',
        'id="ageRange"',
        'id="prevBtn"',
        'id="nextBtn"',
        'id="pdfBtn"',
        'id="rasterOnly"',
        'id="comparisonBtn"',
        'id="mapFrame"',
        'id="loader"',
        'id="toast"',
        "STORAGE_KEY = 'pcvs:age'",
    ):
        assert token in html, f'Viewer is missing {token}'


def test_comparison_assets_exist_for_linked_ages():
    for age in COMPARISON_AGES:
        entry = entry_for_age(age)
        assert (REPO_ROOT / entry['comparison']).is_file()
        assert (COMPARISON_DIR / f'metrics_{age}.csv').is_file()
        assert (COMPARISON_DIR / f'gcp_{age}.json').is_file()
    assert (COMPARISON_DIR / 'index_comparison.html').is_file()
    assert (COMPARISON_DIR / 'gcp_picker.html').is_file()


def test_geotiffs_exist_for_catalog_ages():
    geotiff_dir = REPO_ROOT / 'GENERATED_GEOTIFFS'
    if not geotiff_dir.is_dir():
        raise AssertionError('GENERATED_GEOTIFFS/ is missing')
    missing = []
    for age in catalog_ages():
        knn = list(geotiff_dir.glob(f'{age}_ma_knn_idw_*.tif'))
        idw = list(geotiff_dir.glob(f'{age}_ma_idw_only_*.tif'))
        if not knn:
            missing.append(f'{age} Ma KNN GeoTIFF')
        if not idw:
            missing.append(f'{age} Ma IDW-only GeoTIFF')
    assert not missing, 'Missing interpolated rasters:\n' + '\n'.join(missing)


ACCENTED_BASINS = (
    'Ceará',
    'Espírito Santo',
    'Jacuípe',
    'Marajó',
    'Pará-Maranhão',
    'Paraná',
    'Parnaíba',
    'Pernambuco-Paraíba',
    'Recôncavo',
    'São Luís',
    'Solimões',
)

BROKEN_BASIN_FRAGMENTS = (
    'Cear?',
    'Esp?rito Santo',
    'Jacu?pe',
    'Maraj?',
    'Par?-Maranh?o',
    'Paran?',
    'Parna?ba',
    'Para?ba',
    'Rec?ncavo',
    'S?o Lu?s',
    'Solim?es',
)


def test_restore_lost_accents_repairs_basin_names():
    import render_map as renderer

    for broken, fixed in (
        ('Cear?', 'Ceará'),
        ('Par?-Maranh?o', 'Pará-Maranhão'),
        ('S?o Lu?s', 'São Luís'),
        ('S?o Lu?s-Graja?', 'São Luís-Grajaú'),
        ('Pernambuco-Para?ba', 'Pernambuco-Paraíba'),
        ('Rec?ncavo', 'Recôncavo'),
        ('Solim?es', 'Solimões'),
        ('Esp?rito Santo', 'Espírito Santo'),
        ('Paran?', 'Paraná'),
        ('Jatob?', 'Jatobá'),
        ('Maraj?', 'Marajó'),
        ('Neuqu?n', 'Neuquén'),
        ('Ca?ad?n Asfalto', 'Cañadón Asfalto'),
        ('Aur?s', 'Aurès'),
        ('Santos', 'Santos'),
        ('Potiguar', 'Potiguar'),
    ):
        assert renderer.restore_lost_accents(broken) == fixed


def test_generated_maps_keep_basin_accents():
    files = knn_html_files() + idw_html_files()
    assert files, 'No generated maps to inspect'
    seen = set()
    broken = []
    for path in files:
        basins = embedded_basins(path.read_text(encoding='utf-8'))
        seen.update(basins)
        bad = [name for name in basins if '?' in name]
        if bad:
            broken.append(f'{path.name}: {bad}')
    assert not broken, 'Basin names still lost their accents:\n' + '\n'.join(broken)
    for name in ACCENTED_BASINS:
        assert name in seen, f'{name} never appears in the generated maps'
    for fragment in BROKEN_BASIN_FRAGMENTS:
        assert fragment not in seen, f'Broken basin label still published: {fragment}'


def test_every_html_and_pdf_map_was_generated():
    knn = knn_html_files()
    idw = idw_html_files()
    assert knn, 'No KNN HTML maps; run python render_map.py --pdf'
    assert idw, 'No IDW HTML maps; run python render_map.py --pdf'
    catalog_paths = {entry['path'] for entry in load_maps_catalog()}
    knn_paths = {path.relative_to(REPO_ROOT).as_posix() for path in knn}
    assert catalog_paths == knn_paths

    missing = []
    for html in knn + idw:
        if html.stat().st_size < 10_000:
            missing.append(f'tiny HTML {html.relative_to(REPO_ROOT)}')
        png = overlay_png_for(html)
        if not png.is_file() or png.stat().st_size < 1000:
            missing.append(f'missing overlay {png.name}')
        for scope, pdf in pdf_files_for(html).items():
            if not is_valid_pdf(pdf):
                missing.append(f'missing/invalid {scope} PDF for {html.name}')
    assert not missing, 'Maps were not generated successfully:\n' + '\n'.join(missing)


def test_html_markers_use_the_published_point_size():
    import render_map as renderer

    files = knn_html_files() + idw_html_files()
    assert files
    radius = renderer.POINT_RADIUS_PX
    weight = renderer.POINT_WEIGHT_PX
    outer = renderer.POINT_OUTER_PX
    broken = []
    for path in files:
        text = path.read_text(encoding='utf-8')
        if f'"radius": {radius}' not in text:
            broken.append(f'{path.name}: CircleMarker radius is not {radius}')
        if f'"weight": {weight}' not in text:
            broken.append(f'{path.name}: CircleMarker weight is not {weight}')
        if 'climate-point-icon' in text:
            icon_size = f'"iconSize": [{outer}, {outer}]'
            icon_size_2 = f'"iconSize": [{outer:.2f}, {outer:.2f}]'
            if icon_size not in text and icon_size_2 not in text:
                broken.append(f'{path.name}: split-marker SVG is not {outer}px')
    assert not broken, 'Point size drifted in generated HTML:\n' + '\n'.join(broken)
