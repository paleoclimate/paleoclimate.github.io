"""
Hay & Floegel (2012) comparison for the ages without hand-picked control points.

Hay, W.W. & Floegel, S. (2012), New thoughts about the Cretaceous climate and
oceans, Earth-Science Reviews 115, 262-272, reprint the climate-zone maps of
Chumakov et al. (1995) in Figs. 2-7. The Albian (Fig. 4) and Aptian (Fig. 3)
maps are the 105 and 115 Ma Floegel comparison already scored with control
points (``compare_floegel.py``). This script adds the other four, at the PCVS
age nearest the middle of each stage:

  Fig. 2 Berriasian    -> 140 Ma      Fig. 5 Cenomanian    -> 95 Ma
  Fig. 6 Santonian     -> 85 Ma       Fig. 7 Maastrichtian -> 70 Ma

The figures are vector drawings in an equirectangular frame with a 30 degree
graticule and no coordinate labels. ``extract`` (run once, needs the PDF and
``pymupdf``) renders a figure, takes the scale from the graticule spacing,
finds the offset of the map by matching its shoreline to our coastline over
the whole figure, and keeps only a crop around the raster window in
``COMPARISON/inputs/{age}_chumakov.png``, with the georeference in
``COMPARISON/inputs/chumakov_georef.json``. The article is not open access,
so the repository keeps the crops, as it does for the Floegel scans.

``compare`` refits the rotation and shift on the crop (as for Burgener),
classifies it with the Floegel legend and de-para, and writes the same
reports as ``compare_floegel.py`` (``comparison_report_{age}.html``), so the
viewer's "Comparison Hay & Floegel (2012)" button covers these ages too.
``--check`` also scores Albian and Aptian this way, to compare the automatic
georeference with the control points at 105 and 115 Ma; it never overwrites
those reports.

    python compare_chumakov.py extract --pdf "<Hay e Floegel 2012>.pdf"
    python compare_chumakov.py compare [--check]
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np

import compare_burgener as cb
import compare_floegel as cf

GEOREF_PATH = os.path.join(cf.INPUT_DIR, 'chumakov_georef.json')
SUMMARY_NAME = 'chumakov_summary.csv'
DPI = 200
GRATICULE_DEG = 30.0
CROP_MARGIN = 25.0          # degrees of figure kept around the raster window

# PCVS age: (figure, stage, page, frame rectangle in PDF points)
FIGURES = {
    '140': (2, 'Berriasian', 3, (37, 55, 550, 314)),
    '95': (5, 'Cenomanian', 4, (46, 468, 558, 727)),
    '85': (6, 'Santonian', 5, (37, 56, 549, 313)),
    '70': (7, 'Maastrichtian', 5, (37, 460, 550, 718)),
}
# Scored with control points by compare_floegel.py; the automatic score is a check only.
CHECK_FIGURES = {
    '105': (4, 'Albian', 4, (45, 56, 557, 318)),
    '115': (3, 'Aptian', 3, (36, 484, 550, 726)),
}
AGES = tuple(FIGURES)
LAND_CLOSE = 9              # px; closes symbols, labels and present-day coastlines on land
FRAME_GAP = 15              # px; graticule candidates this close to the frame are the frame
# The full-globe Berriasian map has 180 W at its left frame, and the cropped ones are
# drawn on the same centring, so the corner longitude is taken as 180 W and the
# coarse scan stays within +-45 degrees of it (+-30 in latitude).
COARSE_SCAN = ((-45.0, 45.0), (-30.0, 30.0))


def image_path(age):
    return os.path.join(cf.INPUT_DIR, f'{age}_chumakov.png')


def figure_info(age):
    return FIGURES.get(age) or CHECK_FIGURES[age]


def render_figure(pdf_path, age):
    """RGB render of one figure, frame plus a 4 pt margin, at DPI."""
    import pymupdf  # only the one-off extraction needs it

    _fig, _stage, page, rect = figure_info(age)
    with pymupdf.open(pdf_path) as doc:
        clip = pymupdf.Rect(rect[0] - 4, rect[1] - 4, rect[2] + 4, rect[3] + 4)
        pix = doc[page - 1].get_pixmap(dpi=DPI, clip=clip)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return np.ascontiguousarray(rgb[..., :3])


def graticule_spacing(rgb, frame):
    """Pixels between 30 degree graticule lines (neutral gray rows and columns).

    Lines within FRAME_GAP px of the frame are the frame itself and are skipped.
    """
    img = rgb.astype(np.int32)
    gray = (np.ptp(img, axis=2) < 12) & (img.mean(axis=2) > 100) & (img.mean(axis=2) < 190)
    x0, x1, y0, y1 = frame
    centers = []
    for axis, (lo, hi) in ((0, (x0, x1)), (1, (y0, y1))):
        lines = np.where(gray.mean(axis=axis) > 0.25)[0]
        runs, start = [], None
        for i, v in enumerate(lines):
            start = v if start is None else start
            if i + 1 == len(lines) or lines[i + 1] != v + 1:
                center = (start + v) / 2.0
                if lo + FRAME_GAP < center < hi - FRAME_GAP:
                    runs.append(center)
                start = None
        centers.append(np.array(runs))
    gaps = np.concatenate([np.diff(c) for c in centers if len(c) > 1])
    gaps = gaps[gaps > 40]
    if gaps.size == 0:
        raise RuntimeError('graticule not found')
    unit = gaps.min()
    steps = np.maximum(np.round(gaps / unit), 1)
    return float(gaps.sum() / steps.sum())


# The vector figures fill some zones with paler tints than the scanned legend
# (``legenda.png``): pale pink humid zone, pale yellow arid zone, light greens.
# Without them the pale pink sits as close to paper white as to the legend pink.
FIGURE_TINTS = [
    ((228, 188, 212), cf.HUMID),   # humid zone, pale
    ((208, 132, 176), cf.HUMID),   # humid zone
    ((252, 248, 140), cf.DRY),     # arid zone, pale
    ((248, 244, 88), cf.DRY),      # arid zone
    ((228, 184, 120), cf.DRY),     # evaporite belts
    ((216, 156, 76), cf.DRY),      # evaporite belts, dark
    ((164, 200, 92), cf.SEMI),     # mid-latitude warm humid belt
    ((140, 184, 56), cf.SEMI),     # mid-latitude warm humid belt, variant
    ((44, 148, 64), cf.SEMI),      # high-latitude temperate belt
]


def legend_prototypes():
    """Floegel legend prototypes plus the tints the vector figures use."""
    return cf.extract_legend_prototypes(os.path.join(cf.INPUT_DIR, 'legenda.png')) + FIGURE_TINTS


def land_mask(rgb, prototypes, frame=None):
    """Climate-zone land of a Chumakov figure, with symbols and labels closed."""
    land = (cf.classify_floegel(rgb, prototypes) > 0).astype(np.uint8)
    land = cv2.morphologyEx(land, cv2.MORPH_CLOSE, np.ones((LAND_CLOSE, LAND_CLOSE), np.uint8))
    land = cv2.morphologyEx(land, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    if frame is not None:
        x0, x1, y0, y1 = frame
        land[:, :x0 + 3] = False
        land[:, x1 - 2:] = False
        land[:y0 + 3, :] = False
        land[y1 - 2:, :] = False
    return land


def load_georef():
    if not os.path.isfile(GEOREF_PATH):
        return {}
    with open(GEOREF_PATH, encoding='utf-8') as handle:
        return json.load(handle)


def extract(pdf_path, ages):
    """Render, georeference and crop each figure; write the crops and the georef table."""
    prototypes = legend_prototypes()
    table = load_georef()
    for age in ages:
        figure, stage, _page, _rect = figure_info(age)
        rgb = render_figure(pdf_path, age)
        frame = cb.find_frame(rgb)
        spacing = graticule_spacing(rgb, frame)
        deg = GRATICULE_DEG / spacing
        plate = cb.PlateCarree(frame[0], frame[2], deg, deg)
        _classes, meta = cf.geotiff_classes(age)
        land = land_mask(rgb, prototypes, frame)
        params, before, after = cb.fit_shoreline(age, land, plate, meta, coarse=COARSE_SCAN)

        # Crop: the raster window plus a margin, taken back to figure pixels.
        b = meta['bounds']
        lons = np.array([b.left - CROP_MARGIN, b.right + CROP_MARGIN])
        lats = np.array([b.bottom - CROP_MARGIN, b.top + CROP_MARGIN])
        grid_lon, grid_lat = np.meshgrid(np.linspace(*lons, 30), np.linspace(*lats, 30))
        src_lon, src_lat = cb._unrotate_shift(params, cb.window_center(meta), grid_lon, grid_lat)
        px, py = plate.to_pixel(src_lon, src_lat)
        h, w = land.shape
        cx0, cx1 = int(max(0, np.floor(px.min()))), int(min(w, np.ceil(px.max())))
        cy0, cy1 = int(max(0, np.floor(py.min()))), int(min(h, np.ceil(py.max())))
        crop = rgb[cy0:cy1, cx0:cx1]
        cv2.imwrite(image_path(age), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        crop_plate = cb.PlateCarree(plate.x0 - cx0, plate.y0 - cy0, deg, deg, plate.west, plate.north)
        table[age] = dict(
            figure=figure, stage=stage, dpi=DPI, graticule_px=round(spacing, 2),
            plate=crop_plate.as_dict(), fit=[round(p, 3) for p in params],
            shore_distance_before_deg=round(before, 3), shore_distance_after_deg=round(after, 3),
        )
        print(f'[chumakov] {age} Ma Fig. {figure} {stage}: graticule {spacing:.1f} px, '
              f'fit rot {params[0]:+.2f} dlon {params[1]:+.2f} dlat {params[2]:+.2f}, '
              f'shore {before:.2f} -> {after:.2f} deg, crop {crop.shape[1]}x{crop.shape[0]}')
    with open(GEOREF_PATH, 'w', encoding='utf-8') as handle:
        json.dump(table, handle, indent=1, sort_keys=True)
        handle.write('\n')


def load_crop(age):
    bgr = cv2.imread(image_path(age))
    if bgr is None:
        raise FileNotFoundError(image_path(age))
    entry = load_georef()[age]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cb.PlateCarree(**entry['plate']), entry


def _method_note(age, entry, params, before, after):
    figure, stage = entry['figure'], entry['stage']
    return (
        f"<b>Fonte:</b> Hay &amp; Floegel (2012), Earth-Science Reviews 115, Fig. {figure} ({stage}), "
        "apos Chumakov et al. (1995). <b>Metodo:</b> sem pontos de controle. A figura e "
        f"equiretangular com graticula de 30 graus ({entry['graticule_px']} px a {entry['dpi']} dpi), "
        "que da a escala; a posicao vem do ajuste entre a costa do mapa e a nossa linha de costa: "
        f"rotacao {params[0]:+.2f} graus, deslocamento {params[1]:+.2f} graus em longitude e "
        f"{params[2]:+.2f} graus em latitude (distancia media de costa {before:.2f} -> {after:.2f} graus). "
        "Uma unica transformacao vale para a janela inteira. Mesmo de-para do Floegel "
        "(verdes = Semi-arido, amarelo e laranja = Seco, rosa = Umido). Compara-se so onde o mapa "
        "tem zona climatica."
    )


def score_age(age, prototypes, write_reports=True):
    rgb, plate, entry = load_crop(age)
    ref_classes, meta = cf.geotiff_classes(age)
    land = land_mask(rgb, prototypes)
    params, before, after = cb.fit_shoreline(age, land, plate, meta, around=tuple(entry['fit']))
    warped = cb.warp_to_raster(rgb, plate, meta, params)
    valid = warped.sum(axis=2) < 765
    atlas_classes = cf.classify_floegel(warped, prototypes, valid_mask=valid)
    mask = atlas_classes > 0
    cm = cf.confusion_matrix(ref_classes, atlas_classes, mask)
    metrics = cf.compute_metrics(cm)
    ref_pct = cf.class_area_pct(ref_classes, mask)
    atlas_pct = cf.class_area_pct(atlas_classes, mask)
    if write_reports:
        shown = warped.copy()
        cf.draw_coastline(shown, meta, age)
        cv2.imwrite(cf.warped_path(age), cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
        cf.save_metrics_csv(age, cm, metrics, ref_pct, atlas_pct)
        cf.save_class_composite(age, ref_classes, atlas_classes, mask, cm)
        cf.save_report_html(age, cm, metrics, ref_pct, atlas_pct, 'shoreline', 0,
                            method_note=_method_note(age, entry, params, before, after))
    accuracy = metrics.get('overall_accuracy', 0) * 100
    print(f"[chumakov] {age} Ma Fig. {entry['figure']} {entry['stage']} -> acuracia {accuracy:.2f}% "
          f"kappa {metrics.get('kappa', 0):.3f} ajuste rot {params[0]:+.2f} dlon {params[1]:+.2f} "
          f"dlat {params[2]:+.2f}" + ('' if write_reports else ' (checagem; relatorio nao gravado)'))
    return {
        'age_ma': age, 'figure': entry['figure'], 'stage': entry['stage'],
        'georeference': 'shoreline fit' if write_reports else 'shoreline fit (check)',
        'accuracy_pct': round(accuracy, 2), 'kappa': round(metrics.get('kappa', 0), 3),
        'rotation_deg': round(params[0], 2), 'shift_lon_deg': round(params[1], 2),
        'shift_lat_deg': round(params[2], 2), 'shore_distance_after_deg': round(after, 3),
        'pixels': metrics.get('total_pixels', 0),
    }


def cmd_extract(args):
    ages = [args.age] if args.age else list(FIGURES) + list(CHECK_FIGURES)
    extract(args.pdf, ages)


def cmd_compare(args):
    prototypes = legend_prototypes()
    georef = load_georef()
    ages = [args.age] if args.age else [a for a in AGES if a in georef and os.path.isfile(image_path(a))]
    if not ages:
        print('[chumakov] No crops in COMPARISON/inputs; run extract first.')
        return
    rows, failures = [], 0
    for age in ages:
        try:
            rows.append(score_age(age, prototypes))
        except Exception as exc:
            failures += 1
            print(f'[chumakov] {age} Ma failed: {exc}')
    if args.check:
        for age in CHECK_FIGURES:
            if age in georef and os.path.isfile(image_path(age)):
                rows.append(score_age(age, prototypes, write_reports=False))
    if rows and not args.age:
        with open(os.path.join(cf.COMP_DIR, SUMMARY_NAME), 'w', newline='', encoding='utf-8') as h:
            writer = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if rows:
        cf.generate_comparison_index()
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Hay & Floegel (2012) / Chumakov et al. (1995): idades sem pontos de controle.')
    parser.add_argument('--suffix', default=cf.GEOTIFF_SUFFIX)
    sub = parser.add_subparsers(dest='command', required=True)
    p_ext = sub.add_parser('extract', help='Renderiza, georreferencia e recorta as figuras do PDF')
    p_ext.add_argument('--pdf', required=True, help='PDF de Hay & Floegel (2012)')
    p_ext.add_argument('--age', choices=list(FIGURES) + list(CHECK_FIGURES))
    p_ext.set_defaults(func=cmd_extract)
    p_cmp = sub.add_parser('compare', help='Ajuste de costa, metricas e relatorios')
    p_cmp.add_argument('--age', choices=AGES)
    p_cmp.add_argument('--check', action='store_true',
                       help='Tambem pontua Albiano (105) e Aptiano (115) sem gravar relatorio')
    p_cmp.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    cf.GEOTIFF_SUFFIX = args.suffix
    args.func(args)


if __name__ == '__main__':
    main()
