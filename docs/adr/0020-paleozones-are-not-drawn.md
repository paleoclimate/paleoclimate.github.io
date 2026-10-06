# Paleozones are not drawn

Paleozone polygons were an optional overlay on top of the raster. They covered most of the reconstruction and the team asked for that layer to go away. Published maps no longer load or draw `{base}_paleozones.geojson`. The file is still discovered so it is not treated as a point dataset. Paleozones still do not seed or mask the interpolator.

**Considered options:** (1) stop drawing the layer; (2) keep it and leave the checkbox off. (2) leaves the belts one click away, which is the layer the team said can disappear.

**Consequences:** Supersedes the overlay in `docs/adr/0001-paleozones-do-not-constrain-interpolation.md` and the hover rule in `docs/adr/0017-paleozone-hover-shows-nothing.md`. Interpolation is unchanged.
