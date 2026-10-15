# Added points wait in the browser, then download as one collection

The add-points screen used to download a single feature on submit. It now keeps a queue. Add point validates the form, appends one feature, clears the form back to Choose (a native reset would select 65 Ma and Dry), and shows how many points are queued. Each queued row can be removed. Download GeoJSON writes one FeatureCollection and then drops the queue. The queue is stored under `pcvs:added-points` in localStorage, so a reload, a closed browser, or back and forward still has it. The file is `points-YYYYMMDD-HHMMSS.geojson`, and every feature's FILE1 is that filename. The lead sentence tells the author, in English, to add as many points as they wish and then export the standardized GeoJSON; nothing is written onto the published maps.

**Considered options:** (1) one collection after an explicit download, queue in localStorage, cleared only by that download; (2) one file per point, as before; (3) sessionStorage, which a closed browser would forget. (2) makes the author download after every row. (3) loses the queue on the cases the screen was asked to survive.

This supersedes the one-feature download in `docs/adr/0029-add-point-downloads-a-geojson-feature.md`.
