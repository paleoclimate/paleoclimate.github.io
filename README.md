# Paleogeographic Map Renderer - 110 Million Years Ago

This project renders a paleogeographic map from 110 million years ago using Folium, displaying GeoJSON data points, coastlines, and GeoTIFF raster data.

## Features

- **GeoJSON Point Data**: Displays geological formation points with climate classification
- **Coastline Data**: Shows reconstructed coastlines from 110 Ma
- **GeoTIFF Raster**: Displays interpolated surface data (KNN + IDW)
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

Run the script to generate the interactive maps (KNN + IDW). The published
parameter set is:

```bash
python render_map.py --power 4.0 --gradient-sharp 18.0 --kdtree
```

This writes GeoTIFFs, one interactive HTML map per age, and regenerates `index.html`.

## Verify after changes

After changing the renderer, the viewer, or the comparison pages, regenerate
maps if needed and run the UI/UX suite. The suite starts a local server, opens
Chromium, and exercises the viewer, every generated map, and the comparison tools.

```bash
# Maps already on disk: just run the tests
python verify.py

# After a renderer change: generate the published maps, then test
python verify.py --generate --power 4.0 --gradient-sharp 18.0 --kdtree

# Generate only one age, then test
python verify.py --generate --map 110 --power 4.0 --gradient-sharp 18.0 --kdtree
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
  
- **Coastlines**: Reconstructed coastline polylines

- **Raster**: Interpolated surface data from the GeoTIFF file

- **Color stats**: Share of the raster area falling in each climate class

## Files

- `render_map.py`: Main script to generate the map
- `verify.py`: Generate maps (optional) and run the UI/UX regression suite
- `tests/`: Playwright + pytest coverage for the viewer, maps, and comparison pages
- `requirements.txt`: Python package dependencies
- `GEOJSON/`: Contains GeoJSON files with point and coastline data
- `GEOTIFF/`: Contains GeoTIFF raster files
- `RASTER/`: Contains ArcGIS raster data (not directly used in current implementation)

## Output

For every dataset in `GEOJSON/`, the script generates:

- `GENERATED_GEOTIFFS/`: interpolated GeoTIFFs (IDW-only and KNN + IDW)
- `GENERATED_IDW_MAPS/`, `GENERATED_KNN_IDW_MAPS/`: one interactive `map_<age>_*.html`
  per map, plus its raster overlay PNG
- `index.html`: the viewer that switches between ages

## PDF export

The map can be exported at two scopes:

- **entire map**: the full extent of the data, coastlines included
- **raster area**: only the region covered by the interpolated (coloured) raster

Both are framed so the map fills the page edge to edge; the page itself is sized to the
aspect ratio of the exported region, so there are no white margins to trim.

In the viewer, tick **Raster area only** next to the **PDF** button to choose the scope. The
export is rendered in the browser from the map as it currently stands, so whatever is checked
under **Layers** (Raster, Coastlines, Data points, Color stats) and whichever basins are
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
