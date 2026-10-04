# A basin click shows the name, and a data point wins

Turning on Basins draws a stroke. A click on that stroke or anywhere inside the polygon opens a popup with `BASIN_NAME` and nothing else. The fill is invisible (`fillOpacity` 0) and only exists so the interior is a hit target; `pointer-events` is `all`, because a transparent fill is not clickable under the SVG default. Data points are raised above the basins whenever that overlay is added, so a click on a plotted point opens the point popup instead of the basin name. The basin filter still only hides markers.

**Considered options:** (1) invisible fill plus the point kept on top; (2) only the stroke is clickable; (3) a click on a point inside a basin shows the basin name. (2) misses the interior. (3) hides the point the reader clicked.
