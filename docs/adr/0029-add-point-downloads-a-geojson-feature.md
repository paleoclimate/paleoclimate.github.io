# Adding a point downloads a GeoJSON feature

Superseded by `docs/adr/0035-add-points-queue-downloads-one-collection.md`. The screen still does not post the row anywhere and it does not write the point file on the server. It now queues features and downloads one collection.

The October point file is the column pattern. Which fields are shown, and which are filled in on export, is `docs/adr/0037-first-section-fields-are-required.md` and `docs/adr/0036-show-on-ma-copies-time.md`. `docs/adr/0033-add-points-form-matches-the-geojson.md` is the earlier column layout.

**Considered options:** (1) a download the author merges locally; (2) a form that appends to the published point file. (2) would be the database the viewer is not supposed to expose, and this site has no server to store it.
