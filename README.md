# Paleogeographic Map Renderer - 110 Million Years Ago

This project renders a paleogeographic map from 110 million years ago using Folium, displaying GeoJSON data points, coastlines, and GeoTIFF raster data.

## Features

- **GeoJSON Point Data**: Displays geological formation points with climate classification
- **Coastline Data**: Shows reconstructed coastlines from 110 Ma
- **GeoTIFF Raster**: Displays interpolated surface data (KNN + IDW)
- **Interactive Map**: Full-featured Folium map with layer controls, fullscreen, and measurement tools

## Installation

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script to generate the interactive map (KNN + IDW):

```bash
python render_map.py
```

This will create an HTML file named `map_110_ma.html` that you can open in any web browser.

## Data Layers

- **Data Points (110 Ma)**: Geological formation points colored by climate:
  - Blue: Humid (H)
  - Yellow: Dry (D)
  - Green: Semi-arid (S)
  
- **Coastlines (110 Ma)**: Reconstructed coastline polylines

- **Raster (KNN + IDW)**: Interpolated surface data from the GeoTIFF file

## Files

- `render_map.py`: Main script to generate the map
- `requirements.txt`: Python package dependencies
- `GEOJSON/`: Contains GeoJSON files with point and coastline data
- `GEOTIFF/`: Contains GeoTIFF raster files
- `RASTER/`: Contains ArcGIS raster data (not directly used in current implementation)

## Output

The script generates:
- `map_110_ma.html`: Interactive map file (KNN + IDW)
- `map_110_ma_original.html`: Original raster map
- `map_110_ma_knn_idw.html`: KNN + IDW map
- `raster_overlay.png`: Raster visualization (KNN + IDW)

## Notes

- The map automatically fits to show all data
- You can toggle layers on/off using the layer control
- Use the fullscreen button for better viewing
- The measurement tool allows you to measure distances on the map

