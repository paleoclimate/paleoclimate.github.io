# Published HTML is StatiCrypt ciphertext

The hash gate hid the atlas with CSS and left the map HTML in the file. Anyone with the URL could read it in view-source or in DevTools before typing the password. StatiCrypt replaces each published page — the viewer, the add-points page, every generated map, and the comparison pages — with a password shell and an AES payload. A direct URL shows that shell. The team password is the one the suite already uses (`Paleoclima-2026-entrega`). After a correct password the browser has to hold the decrypted page, so DevTools can see it then; until then the response is ciphertext. The hashed passphrase is kept in sessionStorage so the map iframe in the same tab decrypts without a second prompt. The add-points form that gets encrypted lives in `templates/add-point.html`.

The password field is a real password input, narrower than the old full-width box, with an eye that shows what was typed.

**Considered options:** (1) StatiCrypt on the published HTML, session unlock for the iframe; (2) keep the SHA-256 overlay; (3) a server login. (2) does not hide the file. (3) needs a backend this atlas does not have.

This supersedes the gate in `docs/adr/0028-viewer-password-does-not-publish-the-point-files.md`.
