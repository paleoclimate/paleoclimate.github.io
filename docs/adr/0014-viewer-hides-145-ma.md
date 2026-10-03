# The viewer hides 145 Ma

145 Ma is still generated with the same condensation and indicator IDW as the other ages, including its HTML, GeoTIFF, and PDFs. The viewer no longer lists it: the age combo, the slider, and the arrow keys only walk the ages in `index.html`'s `MAPS` array, and 145 Ma is left out of that array. This supersedes the viewer-facing part of `docs/adr/0006-keep-145-ma-in-this-candidate.md`. Generation of the age is unchanged.

**Considered options:** (1) generate the files and omit the age from the viewer; (2) stop generating 145 Ma; (3) keep it in the age list. (1) is the request. (2) would drop a dataset the previous round kept. (3) would leave it one arrow-key away.
