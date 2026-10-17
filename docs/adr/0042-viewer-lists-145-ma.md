# The viewer lists 145 Ma again

145 Ma stayed in the generation and was left out of the age menu. That menu is the only control that changes the reconstruction age. The request is to show 145 Ma in the select again, so the age goes back into `index.html`'s `MAPS` array with the other sixteen. `VIEWER_HIDDEN_AGES` is empty. The map, the GeoTIFF and the PDFs were already produced; listing the age does not change the interpolator.

**Considered options:** (1) list 145 Ma with the other ages; (2) keep generating it and leave it out of the menu. (2) is `docs/adr/0014-viewer-hides-145-ma.md`. (1) is the request.

**Consequences:** Supersedes the omission in `docs/adr/0014-viewer-hides-145-ma.md`. Generation of the age is unchanged.
