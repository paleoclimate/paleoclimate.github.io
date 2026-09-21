# Paleozones do not constrain interpolation

Paleozones share the Humid / Semi-arid / Dry taxonomy with data points, so it is tempting to paint or seed the IDW/KNN raster from the polygons. We do not: the published raster stays an interpolation of data points only. Paleozones are an optional Folium overlay.

**Considered options:** (1) overlay only; (2) hard-fill the raster from polygons; (3) seed or mask the interpolator with polygon class. (2) and (3) would replace the point interpolator over most of the map, hide point–belt disagreement, and make 115 Ma and 145 Ma (no Paleozone file) a different method from the other ages. Overlay keeps Color stats and the Floegel comparison measuring interpolation, and lets both evidence types be toggled independently.
