# Paleogeographic Map Renderer - 110 Million Years Ago

This project renders paleogeographic maps using Folium, displaying GeoJSON data
points, reconstructed coastlines, optional basin outlines, and GeoTIFF raster data.

## Features

- **GeoJSON Point Data**: Displays geological formation points with climate classification
- **Coastline Data**: Shows reconstructed coastlines from 110 Ma
- **GeoTIFF Raster**: Displays the indicator-IDW climate-class surface
- **Interactive Map**: Full-featured Folium map with layer controls, basin filter, legend,
  fullscreen, and measurement tools
- **PDF Export**: One vector PDF of the raster extent. Coastlines, basin outlines,
  data points and graticule labels stay editable paths. The climate surface stays
  the embedded image.
- **Access**: Published HTML is encrypted with StatiCrypt. A direct URL shows the
  password page, not the map. The point files stay in local `GEOJSON/`.
- **Add points**: `add-point.html` queues points in the browser and downloads
  one GeoJSON FeatureCollection in the October column pattern, without `Weight`.
  `Show_on_Ma` is copied from `TIME`. Nothing is written onto the published maps.

## Installation

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script to generate the interactive maps. The published candidate
condenses data points at 0.5°, adds the expert conceptual maps as
weight-0.5 anchors on a 5° grid, applies the per-age reference-frame rotation,
then paints an indicator IDW per class (weight / d⁴, every anchor, no snap
disk). The KNN reclassification of the anchors survives only as the
discarded `--knn` option (ADR 0044). Each class's share of the IDW weight
is blurred with a 2° Gaussian before the winner is picked, so zone borders
come out round instead of following the anchor grid; around an anchor whose
zone that would erase, the blur steps down until the anchor keeps its class.
`--edge-smooth` sets the sigma in degrees (0 turns it off). Class colors are
equal thirds of the [1, 3] span. Markers stay on the original citation class.

```bash
python render_map.py --power 4.0 --condensation-radius 0.5 --gradient-sharp 18.0 --kdtree --pdf
```

This writes GeoTIFFs, one interactive HTML map per age, a condensation audit
in `CONDENSED/`, and regenerates `index.html`. It then encrypts the published
HTML with StatiCrypt (`npm install` once, so the `staticrypt` CLI is present).
`--gradient-sharp` is accepted and ignored: the raster is class codes, not a
sharpened ramp.

## Verify after changes

After changing the renderer, the viewer, or the comparison pages, regenerate
maps if needed and run the UI/UX suite. The suite starts a local server, opens
Chromium, and exercises the viewer, every generated map, and the comparison tools.

```bash
# Maps already on disk: just run the tests
python verify.py

# After a renderer change: generate the published maps, then test
python verify.py --generate --power 4.0 --condensation-radius 0.5 --gradient-sharp 18.0 --kdtree --pdf

# Generate only one age, then test
python verify.py --generate --map 110 --power 4.0 --condensation-radius 0.5 --gradient-sharp 18.0 --kdtree --pdf
```

The first run may need Playwright's browser:

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

`python verify.py --headed` shows the browser. `python verify.py --slow` also
runs live PDF export. Extra pytest flags go after `--`, for example
`python verify.py -- tests/test_viewer.py -k combo`.

## Data Layers

- **Data points**: Geological formation points colored by climate:
  - Blue: Humid (H)
  - Yellow: Dry (D)
  - Green: Semi-arid (S)
  
- **Paleozones**: Not drawn. Conceptual anchors in
  `conceptual/{age}_ma_conceptual.geojson` feed the interpolator. The
  polygons are not painted into the raster and there is no Paleozones checkbox.
  An age with no conceptual file falls back to sampling
  `{age}_ma_paleozones.geojson` onto a 5° lattice.

- **Basins**: Optional sedimentary-basin outlines for that reconstruction age.
  Stroke only, on when the map opens. A click on the outline or inside
  the basin shows its name; a click on a data point still shows the point.
  Not every age has them.
  
- **Coastlines**: Reconstructed shoreline, drawn as a faint gray reference

- **Raster**: Indicator-IDW class surface with rounded zone borders. Dry, semi-arid, and humid are solid colors. Semi-arid is the class that won the cell, not the average of dry and humid.

- **Color stats**: Share of the raster area falling in each climate class, and the
  number of data points behind the map (conceptual points are not counted)

## Files

- `render_map.py`: Main script to generate the map
- `verify.py`: Generate maps (optional) and run the UI/UX regression suite
- `tests/`: Playwright + pytest coverage for the viewer, maps, and comparison pages
- `requirements.txt`: Python package dependencies
- `GEOJSON/`: Point files (`{age}_ma.geojson`), coastlines (`{age}_ma_coastline.geojson`),
  optional paleozones (`{age}_ma_paleozones.geojson`), and optional basin
  outlines (`{age}_ma_basins.geojson`)
- `conceptual/`: Conceptual lattices, one file per age
  (`{age}_ma_conceptual.geojson`, 65 Ma through 145 Ma)
- `GEOTIFF/`: Contains GeoTIFF raster files
- `RASTER/`: Contains ArcGIS raster data (not directly used in current implementation)

## Output

For every dataset in `GEOJSON/`, the script generates:

- `CONDENSED/`: `{age}_condensed.geojson` (anchors that enter the interpolator)
  and `{age}_overruled.geojson` (data points whose class lost the vote)
- `GENERATED_GEOTIFFS/`: one indicator-IDW GeoTIFF per age
- `GENERATED_IDW_MAPS/`: one interactive `map_<age>_*.html` per age, plus its
  raster overlay PNG and, with `--pdf`, the raster-area vector PDF
- `index.html`: the viewer that switches between those IDW maps

Data points are condensed at 0.5° in source coordinates before the paleo-frame
rotation. Conceptual points skip that cluster and still anchor the raster.
Anchors keep their own class: the weighted class IDW replaces the old KNN
reclassification, which remains only as the discarded `--knn` option
(ADR 0044). Markers on the map stay on the original citations.

## PDF export

The download is the region covered by the interpolated raster, sized so the map
fills the page edge to edge. It is a vector PDF: coastlines, basin outlines, data
points and graticule labels are paths and text. The climate surface stays the
embedded raster image.

In the viewer, **PDF** exports that raster area from the map as it currently
stands, so whatever is checked under **Layers** (Raster, Coastlines, Basins when
present, Data points, Color stats) and whichever basins are filtered in is what
the PDF shows. Interactive controls (zoom, layer switcher, basin filter, measure)
are left out.

Running with `--pdf` pre-renders that page next to each map HTML
(`map_<age>_*_raster.pdf`). That file is what the viewer falls back to when it
cannot render in the browser, for instance when `index.html` is opened straight
from disk instead of being served over HTTP.

## Comparison with Floegel reference maps

For **105 Ma** and **115 Ma**, the project can compare generated maps against published Floegel reference maps (spatial similarity: IoU, accuracy, Cohen's kappa).

Workflow (detailed instructions in [`COMPARISON/README.md`](COMPARISON/README.md)):

1. **Generate reference render and GCP picker**
   ```bash
   python compare_floegel.py render-reference
   ```

2. **Mark control points** — open `COMPARISON/gcp_picker.html`, click matching points on Floegel (left) and the GeoTIFF render (right), export `gcp_105.json` and `gcp_115.json`, and save them in `COMPARISON/`.

3. **Run comparison**
   ```bash
   python compare_floegel.py compare
   ```

Results: open `COMPARISON/index_comparison.html` for HTML reports and CSV metrics.

The Floegel maps are the Chumakov et al. (1995) climate-zone maps reprinted in Hay & Floegel (2012, Earth-Science Reviews 115). The other four maps of that paper (Berriasian, Cenomanian, Santonian, Maastrichtian) are scored at **140, 95, 85 and 70 Ma** without control points: the 30° graticule gives the scale and a shoreline fit gives the position (`docs/adr/0048-hay-floegel-stages-scored-by-shoreline-fit.md`).

```bash
python compare_chumakov.py compare            # uses the committed crops
```

## Comparison with Burgener et al. (2023) palaeo-Köppen maps

For **70, 80, 85, 90, 95, 105, 120, 130 and 145 Ma**, the generated maps are compared with the Burgener et al. (2023) palaeo-Köppen maps reprinted in Scotese et al. (2024, Figs. 89–97, CC BY 4.0). The figures are global equirectangular plates, so their frame georeferences them and no control points are needed; a rotation and shift fitted on the shoreline takes up the difference between PALEOMAP and our reconstruction (`docs/adr/0046-burgener-comparison-georeferenced-by-frame.md`).

```bash
python compare_burgener.py compare
```

Each age gets `COMPARISON/comparison_report_{age}_burgener.html` and `metrics_{age}_burgener.csv`; `COMPARISON/burgener_summary.csv` has one row per age. The viewer shows a **Comparison Burgener (2023)** button for those ages, next to **Comparison Hay & Flögel (2012)** at 70, 85, 95, 105, 115 and 140 Ma.

## Notes

- The map opens framed on the interpolated raster area
- You can toggle layers on/off using the layer control
- The basin filter narrows the data points down to selected basins. Basin outlines are a separate layer and do not follow that filter
- 145 Ma is generated with the other ages and is listed in the viewer age menu
- Use the fullscreen button for better viewing
- The measurement tool allows you to measure distances on the map
- In the viewer, the age menu changes the reconstruction age and `P` exports a PDF
