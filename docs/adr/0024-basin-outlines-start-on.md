# Basin outlines start on

The Basins checkbox used to open off, so the map loaded without the sedimentary limits and the pre-rendered PDF left them out. The outlines now start drawn on every age that has a basin file. The stroke stays the faint gray contour. The filter on data points is unchanged. The published PDF is the default state, so it includes the outlines until someone turns the layer off and exports again. An age with no file still has no checkbox.

**Considered options:** (1) start the layer on; (2) keep it off until the reader checks it. (2) is what the map did, and the request was for the limits to be there when the map opens.

**Consequences:** Supersedes the "checkbox starts off" line in `docs/adr/0013-basin-outlines-are-a-stroke-overlay.md`. The stroke style is unchanged.
