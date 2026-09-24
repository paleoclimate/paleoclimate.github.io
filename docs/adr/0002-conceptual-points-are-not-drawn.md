# Conceptual points are not drawn

Point files mix data points with conceptual points: climate-class locations whose ID is null or the placeholder N/A. Conceptual points are omitted from the drawing (marker, popup, basin filter, and PDF). They still feed KNN/IDW, so the raster can show climate where no marker is visible.

**Considered options:** (1) leave them out of the drawing only; (2) also drop them from interpolation. (2) would republish the climate surface at every reconstruction age except 145 Ma and change the Floegel comparison. (1) keeps the published raster and Color stats as an interpolation of every climate-class location in the point file, including the conceptual grid.

**Consequences:** Null and N/A are the same absence of identity. 70 Ma and 75 Ma store null; the other ages store N/A. A colored patch with no marker is expected where only a conceptual point anchors the interpolator.
