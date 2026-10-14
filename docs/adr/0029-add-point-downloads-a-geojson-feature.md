# Adding a point downloads a GeoJSON feature

The October point file is the column pattern. The add-point screen collects every property in that file except `Weight`, plus the longitude and latitude that place the marker. Submitting it downloads one GeoJSON feature for the local `GEOJSON/` folder. It does not post the row anywhere and it does not write the point file on the server.

**Considered options:** (1) a download the author merges locally; (2) a form that appends to the published point file. (2) would be the database the viewer is not supposed to expose, and this site has no server to store it.
