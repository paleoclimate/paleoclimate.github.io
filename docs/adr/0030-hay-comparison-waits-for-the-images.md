# Hay uses the Floegel comparison, and the Hay images are absent

Hay and Floegel are the same kind of comparison: warp a published map onto the project raster with the control points, then score Dry / Semi-arid / Humid. Floegel already has that pipeline for 105 Ma and 115 Ma. Hay uses the same entry, reading `COMPARISON/inputs/{age}_hay.png`. Those files are not in the repository or in the local Downloads folder, so this round does not write Hay scores. The Floegel reports already on disk stay as they are: their control points were picked against an older GeoTIFF name, and they are not recomputed against `_smooth2.0_class-thirds`.

**Considered options:** (1) share the pipeline and skip Hay when the image is missing; (2) re-score Floegel now with the old control points on the new raster; (3) invent Hay maps. (2) would score a different picture than the one the points were picked on. (3) would not be Hay.
