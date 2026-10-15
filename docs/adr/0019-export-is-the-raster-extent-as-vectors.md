# Export is the raster extent, as a vector PDF

The viewer offered two downloads: the whole map as vectors, and the interpolated raster area as one bitmap. The published download is now only the region the raster covers, drawn as a vector PDF. Coastlines, basin outlines, data points and graticule labels stay paths and text, so they can still be edited. The climate surface stays the embedded overlay image, because that layer is a raster. The "Raster area only" toggle is gone; the PDF button always frames that extent. Pre-rendered files are `*_raster.pdf` only.

**Considered options:** (1) this single vector PDF of the raster extent; (2) keep both scopes and only switch the raster download from a bitmap to vectors; (3) also emit an SVG or a shapefile. (2) still ships the empty margin around the raster, which is the page the team asked to drop. (3) would add a second file beside the PDF the viewer and the vector check already use.
