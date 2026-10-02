# Paleogeographic Map Renderer - 110 Million Years Ago

This project renders paleogeographic maps using Folium, displaying GeoJSON data
points, reconstructed coastlines, optional paleozone polygons, and GeoTIFF raster data.

## Features

- **GeoJSON Point Data**: Displays geological formation points with climate classification
- **Coastline Data**: Shows reconstructed coastlines from 110 Ma
- **GeoTIFF Raster**: Displays the indicator-IDW climate-class surface
- **Interactive Map**: Full-featured Folium map with layer controls, basin filter, legend,
  fullscreen, and measurement tools
- **PDF Export**: Downloads the map exactly as it is on screen, framed either on the whole
  map or on the interpolated raster area

## Installation

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script to generate the interactive maps. The published candidate
condenses data points at 0.5°, reclassifies each interpolator anchor from
the 8 nearest others (KNN, same power), then paints an indicator IDW
(power 4, every anchor, no snap disk). Class colors are equal thirds of
the [1, 3] span. Markers stay on the original citation class.

```bash
python render_map.py --power 4.0 --condensation-radius 0.5 --gradient-sharp 18.0 --kdtree --pdf
```

This writes GeoTIFFs, one interactive HTML map per age, a condensation audit
in `CONDENSED/`, and regenerates `index.html`. `--gradient-sharp` is accepted
and ignored: the raster is class codes, not a sharpened ramp.

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
  
- **Paleozones**: Optional climate-belt polygons (Humid / Semi-arid / Dry). Not every
  reconstruction age has them; when the file is missing the layer is omitted.
  
- **Coastlines**: Reconstructed coastline polylines

- **Raster**: Indicator-IDW class surface. Dry, semi-arid, and humid are solid colors. Semi-arid is the class that won the cell, not the average of dry and humid.

- **Color stats**: Share of the raster area falling in each climate class

## Files

- `render_map.py`: Main script to generate the map
- `verify.py`: Generate maps (optional) and run the UI/UX regression suite
- `tests/`: Playwright + pytest coverage for the viewer, maps, and comparison pages
- `requirements.txt`: Python package dependencies
- `GEOJSON/`: Point files (`{age}_ma.geojson`), coastlines (`{age}_ma_coastline.geojson`),
  and optional paleozones (`{age}_ma_paleozones.geojson`)
- `GEOTIFF/`: Contains GeoTIFF raster files
- `RASTER/`: Contains ArcGIS raster data (not directly used in current implementation)

## Output

For every dataset in `GEOJSON/`, the script generates:

- `CONDENSED/`: `{age}_condensed.geojson` (anchors that enter the interpolator)
  and `{age}_overruled.geojson` (data points whose class lost the vote)
- `GENERATED_GEOTIFFS/`: one indicator-IDW GeoTIFF per age
- `GENERATED_IDW_MAPS/`: one interactive `map_<age>_*.html` per age, plus its
  raster overlay PNG and, with `--pdf`, the full and raster-area PDFs
- `index.html`: the viewer that switches between those IDW maps

Data points are condensed at 0.5° in source coordinates before the paleo-frame
rotation. Conceptual points skip that cluster and still anchor the raster.
Each anchor is then reclassified from its 8 nearest neighbors. Markers on
the map stay on the original citations.

## PDF export

The map can be exported at two scopes:

- **entire map**: the full extent of the data, coastlines included
- **raster area**: only the region covered by the interpolated (coloured) raster

Both are framed so the map fills the page edge to edge; the page itself is sized to the
aspect ratio of the exported region, so there are no white margins to trim.

In the viewer, tick **Raster area only** next to the **PDF** button to choose the scope. The
export is rendered in the browser from the map as it currently stands, so whatever is checked
under **Layers** (Raster, Paleozones when present, Coastlines, Data points, Color stats) and whichever basins are
filtered in is exactly what the PDF shows. Interactive controls (zoom, layer switcher, basin
filter, measure) are left out.

Running with `--pdf` pre-renders both scopes next to each map HTML
(`map_<age>_*_full.pdf` and `map_<age>_*_raster.pdf`). Those are vector PDFs, and they are
what the viewer falls back to when it cannot render in the browser, for instance when
`index.html` is opened straight from disk instead of being served over HTTP.

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

## Notes

- The map opens framed on the interpolated raster area
- You can toggle layers on/off using the layer control
- The basin filter narrows the data points down to selected basins
- Use the fullscreen button for better viewing
- The measurement tool allows you to measure distances on the map
- In the viewer, the arrow keys step through ages and `P` exports a PDF
