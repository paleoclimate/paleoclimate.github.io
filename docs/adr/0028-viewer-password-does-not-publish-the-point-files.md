# The viewer asks for a password and does not publish the point files

Superseded by `docs/adr/0031-published-html-is-staticrypt-ciphertext.md`. The hash gate below does not hide the HTML file.

The point table is local (`GEOJSON/`, gitignored). The fear is a leaked viewer link handing someone that table. The site is static files: a login cannot hide bytes that are already in the HTML, and GitHub Pages has no server session. The viewer and each generated map ask for a shared password before they draw. The check stores a SHA-256 digest, not the password; the password is defined in the environment variable `PALEOCLIMATE_SITE_PASSWORD`. The source GeoJSON is still not committed and there is no download link for it. The map HTML still contains the fields the popup shows, so someone with the file can read those fields without the password.

**Considered options:** (1) this gate, with the limit stated; (2) a server account system; (3) no gate. (2) is a different product than this static atlas. (3) leaves the maps one click from the link.
