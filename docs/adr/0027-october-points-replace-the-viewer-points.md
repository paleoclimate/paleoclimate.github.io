# The October point files replace the viewer points

The point files the viewer interpolates were the rectified export, including the conceptual grid. Dique's October set (`OUTUBRO_final/OUTUBRO`, names like `100.00Ma.geojson`) is the hand-adjusted final: points that sat outside their basin were moved, and the review was done with Dique. Those files replace `{age}_ma.geojson`. Coastlines and basin outlines stay. The October files contain data points only. At 100 Ma, 14 of 261 shared points moved and one old data id is absent. Conceptual points are no longer in the point file, so they no longer anchor the interpolator.

**Considered options:** (1) replace the point files and leave coastlines and basins; (2) keep the conceptual grid from the previous file and only copy moved coordinates. (2) would mix two generations of the table. The file that was handed over is the whole point set.
