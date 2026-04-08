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
from PIL import Image
import cv2

def load_geojson(filepath):
    """Load a GeoJSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

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
                climate = props.get('Climate_cl', 'S')
                values.append(climate_to_numeric(climate))
    return np.array(points), np.array(values)

def knn_smooth_values(points, values, k=8, power=2.0, exclude_self=True):
    """
    Apply KNN regression to smooth point values using spatial neighbors.
    """
    if len(points) == 0:
        raise ValueError("No points available for KNN")
    if len(points) == 1:
        return values.copy()

    distances = cdist(points, points)
    if exclude_self:
        np.fill_diagonal(distances, np.inf)

    k = min(k, len(points) - 1 if exclude_self else len(points))
    if k <= 0:
        return values.copy()

    smoothed = np.zeros_like(values, dtype=np.float64)
    for i in range(len(points)):
        nearest_idx = np.argsort(distances[i])[:k]
        nearest_dist = distances[i][nearest_idx].astype(np.float64)
        nearest_dist[nearest_dist == 0] = 1e-10
        weights = 1.0 / (nearest_dist ** power)
        weights = weights / np.sum(weights)
        smoothed[i] = np.sum(weights * values[nearest_idx])

    return smoothed

def idw_interpolation(points, values, grid_lons, grid_lats, power=2, n_neighbors=12, 
                      preserve_points=True, point_radius=0.15):
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
    
    # Calculate distances from all grid points to all data points
    distances = cdist(grid_points, points)
    
    # Limit to n_neighbors nearest points
    n_neighbors = min(n_neighbors, len(points))
    
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
            # Find indices of N nearest neighbors
            nearest_idx = np.argsort(dist_i)[:n_neighbors]
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

def create_idw_raster(points_data, output_path, resolution=0.1, power=2, n_neighbors=12,
                      preserve_points=True, point_radius=0.15, points=None, values=None):
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
    
    # Perform IDW interpolation using N nearest neighbors
    grid_values = idw_interpolation(points, values, grid_lons, grid_lats, 
                                    power=power, n_neighbors=n_neighbors,
                                    preserve_points=preserve_points, 
                                    point_radius=point_radius)
    
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
                                climate = props.get('Climate_cl', 'S')
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
            
            # Create custom colormap matching climate classification:
            # D (Dry) = Yellow (#eaf51d) = RGB(234, 245, 29)
            # S (Semi-arid) = Dark Green (#0a7a18) = RGB(10, 122, 24)
            # H (Humid) = Blue (#0798db) = RGB(7, 152, 219)
            
            # Map values directly: 1.0 (Dry) -> 2.0 (Semi-arid) -> 3.0 (Humid)
            # Use expected range for better color mapping
            expected_min = 1.0  # Dry
            expected_max = 3.0  # Humid
            
            # Initialize RGB arrays with transparent/black for invalid pixels
            r = np.zeros_like(data, dtype=np.float32)
            g = np.zeros_like(data, dtype=np.float32)
            b = np.zeros_like(data, dtype=np.float32)
            
            # Normalize valid data to 0-1 range based on expected values
            # Clamp values to expected range for consistent color mapping
            data_clamped = np.clip(data, expected_min, expected_max)
            normalized = (data_clamped - expected_min) / (expected_max - expected_min)
            
            # Diminuir gradiente: transições mais abruptas (stretch toward extremes)
            normalized = np.clip((normalized - 0.5) * gradient_sharp + 0.5, 0, 1)
            
            # Define color stops with sharper transitions
            # Yellow (Dry) at t=0: RGB(234, 245, 29)
            # Dark Green (Semi-arid) at t=0.5: RGB(10, 122, 24)
            # Blue (Humid) at t=1.0: RGB(7, 152, 219)
            
            # For t in [0, 0.5]: Yellow -> Dark Green
            mask1 = (normalized <= 0.5) & valid_mask
            t1 = normalized[mask1] * 2.0  # Scale to [0, 1] for this segment
            r[mask1] = 234 - t1 * 224  # 234 -> 10
            g[mask1] = 245 - t1 * 123  # 245 -> 122
            b[mask1] = 29 - t1 * 5     # 29 -> 24
            
            # For t in [0.5, 1.0]: Dark Green -> Blue
            mask2 = (normalized > 0.5) & valid_mask
            t2 = (normalized[mask2] - 0.5) * 2.0  # Scale to [0, 1] for this segment
            r[mask2] = 10 - t2 * 3     # 10 -> 7
            g[mask2] = 122 + t2 * 30   # 122 -> 152
            b[mask2] = 24 + t2 * 195   # 24 -> 219
            
            # Convert to uint8 and ensure valid range
            r = np.clip(r, 0, 255).astype(np.uint8)
            g = np.clip(g, 0, 255).astype(np.uint8)
            b = np.clip(b, 0, 255).astype(np.uint8)
            
            print(f"Color range - R: [{r[valid_mask].min()}, {r[valid_mask].max()}], "
                  f"G: [{g[valid_mask].min()}, {g[valid_mask].max()}], "
                  f"B: [{b[valid_mask].min()}, {b[valid_mask].max()}]")
            
            # Combine into RGB image
            colored_uint8 = np.stack([r, g, b], axis=-1).astype(np.uint8)
            
            # Flip image vertically because raster origin is top-left but geographic is bottom-left
            colored_uint8 = np.flipud(colored_uint8)
            
            # Create RGB image
            img_colored = Image.fromarray(colored_uint8)
            
            # Save to a permanent location
            img_colored.save(raster_img_path)
            print(f"Raster image saved to: {raster_img_path}")
            
            # Add ImageOverlay to map
            image_overlay = ImageOverlay(
                image=raster_img_path,
                bounds=image_bounds,
                opacity=0.7,
                name=layer_name,
                interactive=True,
                cross_origin=False,
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


def build_color_stats_html(stats, title_label):
    if stats is None:
        return "<div style='font-family: Arial; font-size: 11px;'>Sem estatísticas disponíveis.</div>"

    return f"""
    <div style="
        background: rgba(255, 255, 255, 0.9);
        padding: 4px 6px;
        border-radius: 3px;
        box-shadow: 0 0 3px rgba(0,0,0,0.25);
        font-family: Arial;
        font-size: 10px;
        max-width: 180px;
    ">
        <b>{title_label}</b><br>
        <table style="width:100%; border-collapse: collapse; font-size: 10px;">
            <tr><th align="left">Classe</th><th align="right">Pixels</th><th align="right">%</th></tr>
            <tr><td>Amarelo</td><td align="right">{stats['count_yellow']}</td><td align="right">{stats['pct_yellow']:.2f}%</td></tr>
            <tr><td>Verde</td><td align="right">{stats['count_green']}</td><td align="right">{stats['pct_green']:.2f}%</td></tr>
            <tr><td>Azul</td><td align="right">{stats['count_blue']}</td><td align="right">{stats['pct_blue']:.2f}%</td></tr>
            <tr><td>Outros</td><td align="right">{stats['nao_classificado']}</td><td align="right">{stats['pct_nao_classificado']:.2f}%</td></tr>
        </table>
    </div>
    """


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
                font-family: Arial, sans-serif;
                font-size: 9px;
                color: #333;
                font-weight: bold;
                text-shadow:
                    1px 0 0 rgba(255,255,255,0.8), -1px 0 0 rgba(255,255,255,0.8),
                    0 1px 0 rgba(255,255,255,0.8), 0 -1px 0 rgba(255,255,255,0.8);
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
                color: '#555',
                weight: 0.7,
                opacity: 0.5,
                dashArray: '4 4',
                interactive: false,
                pane: 'graticule'
            }};
            var majorLineStyle = {{
                color: '#555',
                weight: 1.0,
                opacity: 0.6,
                dashArray: '6 3',
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
        }})();
        {{% endmacro %}}
    """
    macro = MacroElement()
    macro._template = Template(graticule_js)
    map_obj.add_child(macro)


def create_map(points_data, coastline_data, geotiff_path=None, output_file='map.html', 
               map_title='Paleogeographic Map - 110 Ma', raster_img_path='raster_overlay.png',
               point_values_override=None, raster_layer_name='Raster (IDW Interpolation)',
               gradient_sharp=2.5,
               color_stats_img_path=None, color_stats_name=None):
    """Create a Folium map with points, coastlines, and optional raster.
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
        center_lat = (min(all_lats) + max(all_lats)) / 2
        center_lon = (min(all_lons) + max(all_lons)) / 2
    else:
        # Default center (South America region)
        center_lat = -20
        center_lon = -20
    
    # Create base map
    print("Creating Folium map...")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=3,
        tiles=None
    )
    
    # Add OpenStreetMap hidden and without checkbox in LayerControl
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap', overlay=True, 
                     control=False, show=False).add_to(m)
    
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
        name='Coastlines (110 Ma)',
        style_function=lambda feature: {
            'color': 'black',
            'weight': 2,
            'opacity': 0.8
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME', 'TIME'],
            aliases=['Location:', 'Age (Ma):'],
            sticky=True
        )
    ).add_to(m)
    
    # Add points layer with styling based on climate (40% smaller, thinner border)
    point_radius_px = 3   # 60% of original 5
    point_weight = 1      # thinner black border
    def style_points(feature):
        climate = feature.get('properties', {}).get('Climate_cl', '')
        color_map = {
            'H': '#0798db',    # Blue (Humid)
            'D': '#eaf51d',    # Yellow (Dry)
            'S': '#0a7a18'     # Dark Green (Semi-arid)
        }
        color = color_map.get(climate, 'gray')
        return {
            'fillColor': color,
            'color': 'black',
            'radius': point_radius_px,
            'fillOpacity': 0.7,
            'weight': point_weight
        }
    
    # Create a FeatureGroup for points
    points_group = folium.FeatureGroup(name='Data Points (110 Ma)')
    
    for feature in points_data.get('features', []):
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        
        if geom.get('type') == 'Point':
            coords = geom.get('coordinates', [])
            if coords:
                climate = props.get('Climate_cl', '')
                formation = props.get('Formation', 'N/A')
                basin = props.get('Basin_Sub_', 'N/A')
                country = props.get('Country', 'N/A')
                
                # Create popup with information
                popup_html = f"""
                <div style="font-family: Arial; font-size: 12px;">
                    <b>Formation:</b> {formation}<br>
                    <b>Basin:</b> {basin}<br>
                    <b>Country:</b> {country}<br>
                    <b>Climate:</b> {climate}<br>
                    <b>Age:</b> {props.get('TIME', 'N/A')} Ma
                </div>
                """
                
                point_style = style_points(feature)
                
                folium.CircleMarker(
                    location=[coords[1], coords[0]],
                    radius=point_radius_px,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{formation} ({basin})",
                    color=point_style['color'],
                    fillColor=point_style['fillColor'],
                    fillOpacity=0.7,
                    weight=point_weight
                ).add_to(points_group)
    
    points_group.add_to(m)

    # Basin filter control (topleft, next to zoom)
    basins = sorted(set(
        feature.get('properties', {}).get('Basin_Sub_') or 'N/A'
        for feature in points_data.get('features', [])
        if feature.get('geometry', {}).get('type') == 'Point'
           and feature.get('geometry', {}).get('coordinates')
    ))
    marker_basins = [
        feature.get('properties', {}).get('Basin_Sub_') or 'N/A'
        for feature in points_data.get('features', [])
        if feature.get('geometry', {}).get('type') == 'Point'
           and feature.get('geometry', {}).get('coordinates')
    ]

    if basins:
        pg_name = points_group.get_name()
        basins_json = json.dumps(basins, ensure_ascii=False)
        marker_basins_json = json.dumps(marker_basins, ensure_ascii=False)

        basin_css = MacroElement()
        basin_css._template = Template("""
            {% macro header(this, kwargs) %}
            <style>
                .basin-filter-control {
                    background: white;
                    border-radius: 4px;
                    box-shadow: 0 1px 5px rgba(0,0,0,0.4);
                    font-family: Arial, sans-serif;
                    font-size: 11px;
                    max-height: 60vh;
                    display: flex;
                    flex-direction: column;
                }
                .basin-filter-header {
                    padding: 4px 8px;
                    background: #2c3e50;
                    color: white;
                    font-weight: bold;
                    font-size: 11px;
                    cursor: pointer;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    user-select: none;
                    border-radius: 4px;
                    gap: 6px;
                }
                .basin-filter-control.open .basin-filter-header {
                    border-radius: 4px 4px 0 0;
                }
                .basin-filter-header .arrow {
                    font-size: 8px;
                    transition: transform 0.2s;
                }
                .basin-filter-control.open .basin-filter-header .arrow {
                    transform: rotate(180deg);
                }
                .basin-filter-body {
                    display: none;
                    padding: 4px 6px;
                    overflow-y: auto;
                    max-height: calc(60vh - 28px);
                }
                .basin-filter-control.open .basin-filter-body {
                    display: block;
                }
                .basin-filter-actions {
                    display: flex;
                    gap: 4px;
                    margin-bottom: 4px;
                    padding-bottom: 4px;
                    border-bottom: 1px solid #ddd;
                }
                .basin-filter-actions button {
                    flex: 1;
                    padding: 2px 4px;
                    font-size: 9px;
                    border: 1px solid #bdc3c7;
                    border-radius: 2px;
                    background: #ecf0f1;
                    cursor: pointer;
                    font-family: Arial, sans-serif;
                }
                .basin-filter-actions button:hover {
                    background: #bdc3c7;
                }
                .basin-filter-list {
                    list-style: none;
                    padding: 0;
                    margin: 0;
                }
                .basin-filter-list li {
                    padding: 1px 2px;
                    display: flex;
                    align-items: center;
                }
                .basin-filter-list li:hover {
                    background: #ecf0f1;
                    border-radius: 2px;
                }
                .basin-filter-list label {
                    cursor: pointer;
                    white-space: nowrap;
                    font-size: 10px;
                    display: flex;
                    align-items: center;
                    gap: 3px;
                }
                .basin-filter-list input[type="checkbox"] {
                    margin: 0;
                }
            </style>
            {% endmacro %}
        """)
        m.get_root().add_child(basin_css)

        basin_js = f"""
            {{% macro script(this, kwargs) %}}
            (function() {{
                var map = {{{{ this._parent.get_name() }}}};
                var pointsGroup = {pg_name};
                var allBasins = {basins_json};
                var markerBasins = {marker_basins_json};

                var allMarkers = [];
                pointsGroup.eachLayer(function(layer) {{
                    allMarkers.push(layer);
                }});

                var selectedBasins = new Set(allBasins);

                var filterControl = L.control({{position: 'topleft'}});
                filterControl.onAdd = function() {{
                    var container = L.DomUtil.create('div', 'basin-filter-control');
                    L.DomEvent.disableClickPropagation(container);
                    L.DomEvent.disableScrollPropagation(container);

                    var header = L.DomUtil.create('div', 'basin-filter-header', container);
                    header.innerHTML = 'Basins <span class="arrow">&#9660;</span>';

                    var body = L.DomUtil.create('div', 'basin-filter-body', container);

                    var actions = L.DomUtil.create('div', 'basin-filter-actions', body);
                    var selectAllBtn = L.DomUtil.create('button', '', actions);
                    selectAllBtn.textContent = 'Select All';
                    var deselectAllBtn = L.DomUtil.create('button', '', actions);
                    deselectAllBtn.textContent = 'Deselect All';

                    var list = L.DomUtil.create('ul', 'basin-filter-list', body);
                    var checkboxes = [];
                    allBasins.forEach(function(basin) {{
                        var li = L.DomUtil.create('li', '', list);
                        var label = L.DomUtil.create('label', '', li);
                        var cb = document.createElement('input');
                        cb.type = 'checkbox';
                        cb.checked = true;
                        cb.value = basin;
                        label.appendChild(cb);
                        label.appendChild(document.createTextNode(' ' + basin));
                        checkboxes.push(cb);
                        cb.addEventListener('change', function() {{
                            if (this.checked) {{
                                selectedBasins.add(this.value);
                            }} else {{
                                selectedBasins.delete(this.value);
                            }}
                            applyFilter();
                        }});
                    }});

                    header.addEventListener('click', function() {{
                        container.classList.toggle('open');
                    }});

                    selectAllBtn.addEventListener('click', function() {{
                        checkboxes.forEach(function(cb) {{
                            cb.checked = true;
                            selectedBasins.add(cb.value);
                        }});
                        applyFilter();
                    }});

                    deselectAllBtn.addEventListener('click', function() {{
                        checkboxes.forEach(function(cb) {{
                            cb.checked = false;
                        }});
                        selectedBasins.clear();
                        applyFilter();
                    }});

                    return container;
                }};
                filterControl.addTo(map);

                function applyFilter() {{
                    for (var i = 0; i < allMarkers.length; i++) {{
                        if (selectedBasins.has(markerBasins[i])) {{
                            if (!pointsGroup.hasLayer(allMarkers[i])) {{
                                pointsGroup.addLayer(allMarkers[i]);
                            }}
                        }} else {{
                            if (pointsGroup.hasLayer(allMarkers[i])) {{
                                pointsGroup.removeLayer(allMarkers[i]);
                            }}
                        }}
                    }}
                }}
            }})();
            {{% endmacro %}}
        """
        basin_macro = MacroElement()
        basin_macro._template = Template(basin_js)
        m.add_child(basin_macro)

    # Add color stats as a fixed Leaflet control (bottom-left) toggled via LayerControl checkbox
    if color_stats_img_path and os.path.exists(color_stats_img_path):
        stats = classificar_por_intervalos(color_stats_img_path)
        if stats is not None:
            stats_checkbox_name = color_stats_name or 'Color Stats'
            y = stats['pct_yellow']
            g = stats['pct_green']
            b = stats['pct_blue']
            o = stats['pct_nao_classificado']
            cy = stats['count_yellow']
            cg = stats['count_green']
            cb = stats['count_blue']
            co = stats['nao_classificado']
            # Empty FeatureGroup just to get a checkbox in the LayerControl
            stats_group = folium.FeatureGroup(name=stats_checkbox_name, show=True)
            stats_group.add_to(m)
            control_js = f"""
            {{% macro script(this, kwargs) %}}
                var colorStatsControl = L.control({{position: 'bottomleft'}});
                colorStatsControl.onAdd = function(map) {{
                    var div = L.DomUtil.create('div', 'color-stats-control');
                    div.style.background = 'rgba(255,255,255,0.92)';
                    div.style.padding = '2px 4px';
                    div.style.borderRadius = '2px';
                    div.style.boxShadow = '0 0 2px rgba(0,0,0,0.2)';
                    div.style.fontFamily = 'Arial, sans-serif';
                    div.style.fontSize = '8px';
                    div.style.lineHeight = '1.2';
                    div.innerHTML = '<table style="border-collapse:collapse;font-size:8px">'
                        + '<tr><th align="left">Class</th><th align="right" style="padding-left:5px">Pixels</th><th align="right" style="padding-left:3px">%</th></tr>'
                        + '<tr><td>Dry (Yellow)</td><td align="right" style="padding-left:5px">{cy}</td><td align="right" style="padding-left:3px">{y:.1f}%</td></tr>'
                        + '<tr><td>Semi-Arid (Green)</td><td align="right" style="padding-left:5px">{cg}</td><td align="right" style="padding-left:3px">{g:.1f}%</td></tr>'
                        + '<tr><td>Humid (Blue)</td><td align="right" style="padding-left:5px">{cb}</td><td align="right" style="padding-left:3px">{b:.1f}%</td></tr>'
                        + '<tr><td>Others</td><td align="right" style="padding-left:5px">{co}</td><td align="right" style="padding-left:3px">{o:.1f}%</td></tr>'
                        + '</table>';
                    return div;
                }};
                var _map = {{{{ this._parent.get_name() }}}};
                colorStatsControl.addTo(_map);
                _map.on('overlayremove', function(e) {{
                    if (e.name === '{stats_checkbox_name}') {{
                        _map.removeControl(colorStatsControl);
                    }}
                }});
                _map.on('overlayadd', function(e) {{
                    if (e.name === '{stats_checkbox_name}') {{
                        colorStatsControl.addTo(_map);
                    }}
                }});
            {{% endmacro %}}
            """
            macro = MacroElement()
            macro._template = Template(control_js)
            m.add_child(macro)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # CSS: shrink zoom/fullscreen 30%, compact LayerControl, larger Color Stats font
    ui_css = MacroElement()
    ui_css._template = Template("""
        {% macro header(this, kwargs) %}
        <style>
            .leaflet-control-zoom a {
                width: 26px !important;
                height: 26px !important;
                line-height: 26px !important;
                font-size: 16px !important;
            }
            .leaflet-control-zoom a.fullscreen-icon,
            .leaflet-control-zoom a.leaflet-control-zoom-fullscreen {
                width: 26px !important;
                height: 26px !important;
                line-height: 26px !important;
                background-size: 26px 52px !important;
            }
            .leaflet-control-layers { font-size: 8px !important; padding: 2px 4px !important; }
            .leaflet-control-layers label { margin-bottom: 0 !important; }
            .leaflet-control-layers-overlays label,
            .leaflet-control-layers-base label { padding: 0 !important; line-height: 1.3 !important; }
            .leaflet-control-layers-separator { margin: 2px 0 !important; }
            .color-stats-control { font-size: 10px !important; }
            .color-stats-control table { font-size: 10px !important; }
        </style>
        {% endmacro %}
    """)
    m.get_root().add_child(ui_css)

    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add measure tool
    plugins.MeasureControl().add_to(m)

    # Add coordinate graticule (always visible, 30° intervals)
    _add_graticule(m, interval=30)
    
    # Fit bounds to raster extent (tightest framing around interpolated area)
    raster_bounds = None
    if geotiff_path and os.path.exists(geotiff_path):
        try:
            raster_bounds = get_geotiff_bounds(geotiff_path)
        except:
            pass
    if raster_bounds:
        m.fit_bounds(raster_bounds)
    elif bounds_list:
        all_lats = [b[0][0] for b in bounds_list] + [b[1][0] for b in bounds_list]
        all_lons = [b[0][1] for b in bounds_list] + [b[1][1] for b in bounds_list]
        m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])
    
    # Save map
    print(f"Saving map to {output_file}...")
    m.save(output_file)
    print(f"Map saved successfully! Open {output_file} in your browser.")
    
    return output_file

def html_to_pdf(html_path, pdf_path, wait_seconds=2, pdf_width_in=14, pdf_height_in=10):
    """Export an HTML map to PDF: hide all UI controls and fit PDF to map content (no white borders)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  PDF skipped (install playwright: pip install playwright && playwright install chromium)")
        return False
    if not os.path.exists(html_path):
        return False
    html_path_abs = os.path.abspath(html_path)
    pdf_path_abs = os.path.abspath(pdf_path)
    url = 'file://' + html_path_abs
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            # Viewport in pixels (landscape) so map can fill; will match PDF size
            viewport_w = int(pdf_width_in * 96)
            viewport_h = int(pdf_height_in * 96)
            page = browser.new_page(viewport={'width': viewport_w, 'height': viewport_h})
            page.goto(url, wait_until='networkidle', timeout=30000)
            page.wait_for_timeout(wait_seconds * 1000)

            # Hide all on-screen controls: Zoom, Full Screen, Measure, Layer control (OSM, Raster, Coastlines, Data points)
            page.evaluate("""
                () => {
                    const sel = [
                        '.leaflet-control-zoom',
                        '.leaflet-control-layers',
                        '.leaflet-control-attribution',
                        '.leaflet-control-fullscreen',
                        '.leaflet-control-measure',
                        '.color-stats-control',
                        '[class*="leaflet-control"]',
                        '[class*="fullscreen"]',
                        '[class*="measure"]'
                    ].join(', ');
                    document.querySelectorAll(sel).forEach(el => { el.style.setProperty('display', 'none', 'important'); });
                }
            """)
            page.wait_for_timeout(300)

            # Shrink point markers and coastline strokes only for PDF
            page.evaluate("""
                () => {
                    if (typeof L === 'undefined') return;
                    for (const key of Object.keys(window)) {
                        try {
                            const v = window[key];
                            if (v && v instanceof L.Map) {
                                v.eachLayer(l => {
                                    if (l instanceof L.CircleMarker && !(l instanceof L.Marker)) {
                                        l.setRadius(0.5);
                                        l.setStyle({ weight: 0.3 });
                                    } else if (l instanceof L.Polyline || l instanceof L.GeoJSON || l instanceof L.FeatureGroup) {
                                        const shrinkWeight = (layer) => {
                                            if (layer.setStyle && layer.options && typeof layer.options.weight === 'number') {
                                                layer.setStyle({ weight: layer.options.weight * 0.8 });
                                            }
                                            if (layer.eachLayer) {
                                                layer.eachLayer(shrinkWeight);
                                            }
                                        };
                                        shrinkWeight(l);
                                    }
                                });
                            }
                        } catch(e) {}
                    }
                }
            """)
            page.wait_for_timeout(300)

            # Make map fill the entire viewport (remove white around map)
            page.evaluate("""
                () => {
                    const style = document.createElement('style');
                    style.textContent = `
                        html, body { margin: 0 !important; padding: 0 !important; width: 100% !important; height: 100% !important; overflow: hidden !important; }
                        .folium-map, #map, [id^="map_"] { position: fixed !important; top: 0 !important; left: 0 !important; width: 100% !important; height: 100% !important; }
                    `;
                    document.head.appendChild(style);
                    const mapEl = document.querySelector('.folium-map') || document.getElementById('map') || document.querySelector('[id^="map_"]');
                    if (mapEl) {
                        mapEl.style.width = '100%';
                        mapEl.style.height = '100%';
                        mapEl.style.position = 'fixed';
                        mapEl.style.top = '0';
                        mapEl.style.left = '0';
                        if (typeof L !== 'undefined') {
                            for (const key of Object.keys(window)) {
                                try {
                                    if (window[key] instanceof L.Map) { window[key].invalidateSize(); break; }
                                } catch(e) {}
                            }
                        }
                    }
                }
            """)
            page.wait_for_timeout(500)

            # PDF: same size as viewport, no margins, so content fills the page
            page.pdf(
                path=pdf_path_abs,
                width='%sin' % pdf_width_in,
                height='%sin' % pdf_height_in,
                margin={'top': '0', 'right': '0', 'bottom': '0', 'left': '0'},
                print_background=True
            )
            browser.close()
        print(f"  PDF saved: {pdf_path}")
        return True
    except Exception as e:
        print(f"  PDF failed for {html_path}: {e}")
        return False

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

def _extract_age_sort_key(filename):
    """Extract numeric age from filename for sorting, e.g. 'map_65_ma_...' -> 65."""
    import re
    m = re.search(r'_(\d+)_ma', filename)
    return int(m.group(1)) if m else 0

def generate_index_html(dir_knn_idw, dir_idw, output='index.html'):
    """Generate an index.html with a dropdown to switch between KNN+IDW map HTMLs."""
    maps_list = []
    folder = dir_knn_idw
    if os.path.isdir(folder):
        htmls = sorted(
            [f for f in os.listdir(folder) if f.endswith('.html')],
            key=_extract_age_sort_key
        )
        for h in htmls:
            age = _extract_age_sort_key(h)
            label = f"{age} Ma"
            pdf_name = h.replace('.html', '.pdf')
            pdf_path = f"{folder}/{pdf_name}"
            maps_list.append({
                'path': f"{folder}/{h}",
                'label': label,
                'pdf': pdf_path if os.path.isfile(pdf_path) else ''
            })

    if not maps_list:
        return

    import json as _json
    maps_json = _json.dumps(maps_list)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Paleogeographic Maps Viewer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: Arial, sans-serif; height: 100vh; display: flex; flex-direction: column; }}
  .toolbar {{
    display: flex; align-items: center; gap: 10px;
    padding: 6px 12px; background: #2c3e50; color: #fff;
    font-size: 13px; flex-shrink: 0;
  }}
  .toolbar label {{ font-weight: bold; }}
  .toolbar select {{
    padding: 3px 6px; font-size: 12px; border-radius: 3px;
    border: 1px solid #7f8c8d; background: #ecf0f1; min-width: 160px;
  }}
  .toolbar .btn-pdf {{
    padding: 4px 10px; font-size: 11px; border-radius: 3px;
    border: 1px solid #7f8c8d; background: #e74c3c; color: #fff;
    cursor: pointer; font-weight: bold; font-family: Arial, sans-serif;
  }}
  .toolbar .btn-pdf:hover {{ background: #c0392b; }}
  .toolbar .btn-pdf.disabled {{
    background: #95a5a6; cursor: default; pointer-events: none; opacity: 0.7;
  }}
  iframe {{ flex: 1; border: none; width: 100%; }}
</style>
</head>
<body>
<div class="toolbar">
  <label for="mapSelect">Map:</label>
  <select id="mapSelect" onchange="onSelectMap(this.value)"></select>
  <button id="pdfBtn" class="btn-pdf disabled" onclick="downloadPdf()">&#8681; PDF</button>
</div>
<iframe id="mapFrame"></iframe>
<script>
var _maps = {maps_json};
var _currentPdf = '';
(function() {{
  var select = document.getElementById('mapSelect');
  _maps.forEach(function(m, i) {{
    var opt = document.createElement('option');
    opt.value = i;
    opt.textContent = m.label;
    select.appendChild(opt);
  }});
  if (_maps.length > 0) onSelectMap(0);
}})();

function onSelectMap(idx) {{
  var m = _maps[idx];
  document.getElementById('mapFrame').src = m.path;
  var btn = document.getElementById('pdfBtn');
  _currentPdf = m.pdf || '';
  if (_currentPdf) {{
    btn.classList.remove('disabled');
  }} else {{
    btn.classList.add('disabled');
  }}
}}

function downloadPdf() {{
  if (!_currentPdf) return;
  var a = document.createElement('a');
  a.href = _currentPdf;
  a.download = _currentPdf.split('/').pop();
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}}
</script>
</body>
</html>"""

    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Index page generated: {output}")

def main():
    """Generate maps for every point+costa dataset found in GEOJSON/."""
    parser = argparse.ArgumentParser(description='Generate paleogeographic maps (IDW and KNN+IDW) for all datasets in GEOJSON/.')
    parser.add_argument('--power', type=float, required=True,
                        help='Power parameter for IDW and KNN (e.g. 4.0)')
    parser.add_argument('--gradient-sharp', type=float, default=2.5,
                        help='Gradient sharpening factor for color transitions (default: 2.5, higher = more abrupt)')
    parser.add_argument('--geojson-dir', default='GEOJSON',
                        help='Directory containing point and coastline GeoJSON files (default: GEOJSON)')
    parser.add_argument('--pdf', action='store_true',
                        help='Export each map to PDF (requires playwright; PDFs are generated with OSM off)')
    args = parser.parse_args()
    power = args.power
    gradient_sharp = args.gradient_sharp
    params_suffix = f'_power{power}_gradient_sharp{gradient_sharp}'

    datasets = discover_geojson_datasets(args.geojson_dir)
    if not datasets:
        print(f"No datasets found in {args.geojson_dir}/ (need both X.geojson and X_costa.geojson for each X).")
        return
    print(f"Found {len(datasets)} dataset(s): {[b for b, _, _ in datasets]}")

    dir_geotiffs = 'GENERATED_GEOTIFFS'
    dir_idw_maps = 'GENERATED_IDW_MAPS'
    dir_knn_idw_maps = 'GENERATED_KNN_IDW_MAPS'
    for d in (dir_geotiffs, dir_idw_maps, dir_knn_idw_maps):
        os.makedirs(d, exist_ok=True)

    generated = []
    for base, points_path, coast_path in datasets:
        title_label = base.replace('_', ' ')  # e.g. 110 ma
        print("\n" + "=" * 60)
        print(f"DATASET: {base}")
        print("=" * 60)

        points_data = load_geojson(points_path)
        coastline_data = load_geojson(coast_path)
        n_pts = len(points_data.get('features', []))
        n_coast = len(coastline_data.get('features', []))
        print(f"Loaded {n_pts} points from {points_path}")
        print(f"Loaded {n_coast} coastline features from {coast_path}")
        if n_pts == 0:
            print(f"Skipping {base}: no point features.")
            continue

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
            create_map(
                points_data=points_data,
                coastline_data=coastline_data,
                geotiff_path=original_raster_path,
                output_file=map1_file,
                map_title=f'Paleogeographic Map - {title_label} (Original Data)',
                raster_img_path=os.path.join(dir_idw_maps, f'raster_overlay_{base}_original.png'),
                raster_layer_name='Raster (Original)',
                gradient_sharp=gradient_sharp
            )
            generated.append(map1_file)
            if args.pdf:
                html_to_pdf(map1_file, os.path.join(dir_idw_maps, os.path.basename(map1_file).replace('.html', '.pdf')))
        else:
            print(f"Original raster not found ({original_raster_path}), skipping Original map.")

        print("\nExecuting IDW Interpolation (no KNN)")
        points, values = extract_points_and_values(points_data)
        create_idw_raster(
            points_data=points_data,
            points=points,
            values=values,
            output_path=idw_only_raster_path,
            resolution=0.1,
            power=power
        )
        print(f"Generating Map: IDW only ({map_idw_file})")
        create_map(
            points_data=points_data,
            coastline_data=coastline_data,
            geotiff_path=idw_only_raster_path,
            output_file=map_idw_file,
            map_title=f'Paleogeographic Map - {title_label} (IDW only)',
            raster_img_path=raster_overlay_idw_png,
            raster_layer_name='Raster (IDW only)',
            gradient_sharp=gradient_sharp,
            color_stats_img_path=raster_overlay_idw_png,
            color_stats_name=f'Color Stats (IDW)'
        )
        generated.append(map_idw_file)
        if args.pdf:
            html_to_pdf(map_idw_file, os.path.join(dir_idw_maps, os.path.basename(map_idw_file).replace('.html', '.pdf')))

        print("Executing KNN Smoothing + IDW Interpolation")
        knn_values = knn_smooth_values(points, values, k=8, power=power, exclude_self=True)
        create_idw_raster(
            points_data=points_data,
            points=points,
            values=knn_values,
            output_path=idw_raster_path,
            resolution=0.1,
            power=power
        )
        print(f"Generating Map: KNN + IDW ({map_knn_idw_file})")
        create_map(
            points_data=points_data,
            coastline_data=coastline_data,
            geotiff_path=idw_raster_path,
            output_file=map_knn_idw_file,
            map_title=f'Paleogeographic Map - {title_label} (KNN + IDW Interpolated)',
            raster_img_path=raster_overlay_knn_idw_png,
            point_values_override=knn_values,
            raster_layer_name='Raster (KNN + IDW)',
            gradient_sharp=gradient_sharp,
            color_stats_img_path=raster_overlay_knn_idw_png,
            color_stats_name=f'Color Stats (KNN + IDW)'
        )
        generated.append(map_knn_idw_file)
        if args.pdf:
            html_to_pdf(map_knn_idw_file, os.path.join(dir_knn_idw_maps, os.path.basename(map_knn_idw_file).replace('.html', '.pdf')))

    print("\n" + "=" * 60)
    print("All maps generated successfully!")
    print("=" * 60)
    print(f"Power: {power}, Gradient sharp: {gradient_sharp}")
    print(f"Datasets processed: {len(datasets)}")
    print(f"GeoTIFFs: {dir_geotiffs}/")
    print(f"IDW-only maps (HTML/PDF): {dir_idw_maps}/")
    print(f"KNN+IDW maps (HTML/PDF): {dir_knn_idw_maps}/")
    for f in generated:
        print(f"  {f}")

    generate_index_html(dir_knn_idw_maps, dir_idw_maps)

if __name__ == '__main__':
    main()
