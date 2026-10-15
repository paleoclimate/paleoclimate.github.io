# Hay and Floegel are scored on the raster the map is showing

The comparison reports were still the warp against an older reference image (105 Ma was 652×747, the map overlay is 638×753; 115 Ma was 702×736 against 623×746). The viewer paints `knn_idw_power4.0_smooth2.0_class-thirds_kdtree`. Class codes now come from that GeoTIFF, equal thirds of [1, 3], which is how the overlay is colored. Control points picked on the old image are moved by a similarity fit of the coastline on that image to the coastline of the current grid, then the reference render is rebuilt at the overlay's size. Hay uses the same grid when `COMPARISON/inputs/{age}_hay.png` exists. Those images are not in the repo, so Hay is not scored.

**Considered options:** (1) rescore Floegel on the current class raster and shift the old control points with the coastline; (2) leave the old reports in place; (3) drop the scores until someone re-picks points by hand. (2) is the mismatch on screen. (3) withholds the comparison the atlas already links.

This supersedes the choice in `docs/adr/0030-hay-comparison-waits-for-the-images.md` to leave the Floegel scores on the old raster.
