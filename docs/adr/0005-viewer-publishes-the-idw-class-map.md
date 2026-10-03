# The viewer publishes the IDW class map

The maps published after this note smooth the anchors with KNN first. See `0011-published-maps-smooth-anchors-with-knn.md`.

The candidate the group will review is the indicator-IDW surface. The previous viewer opened the KNN + IDW map, and that pre-smooth is what pulled a class that had just won its vote back toward the neighbors.

`index.html` lists the IDW maps only. This round does not write a KNN + IDW GeoTIFF or HTML. The Floegel comparison reports already on disk stay as they are; they are not regenerated against the new raster.

**Considered options:** (1) publish IDW and leave KNN ungenerated; (2) generate KNN beside IDW and keep the viewer on KNN; (3) regenerate the Floegel scores now. (2) puts the discussion on the map the group set aside. (3) waits until power and the class rule are locked.
