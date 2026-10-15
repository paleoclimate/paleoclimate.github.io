# Adding a point downloads a GeoJSON feature

The October point file is the column pattern. Submitting the add-points screen downloads one GeoJSON feature for the local `GEOJSON/` folder. It does not post the row anywhere and it does not write the point file on the server. Which fields are shown, and which are filled in on export, is `docs/adr/0033-add-points-form-matches-the-geojson.md`.

**Considered options:** (1) a download the author merges locally; (2) a form that appends to the published point file. (2) would be the database the viewer is not supposed to expose, and this site has no server to store it.
