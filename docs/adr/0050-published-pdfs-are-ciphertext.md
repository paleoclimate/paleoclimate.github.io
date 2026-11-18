# Published PDFs are ciphertext

A pre-rendered `*_raster.pdf` in the repository was a normal PDF. Anyone who cloned the public repo could open it and pull the data points out of the vectors. The published file is now `PCVSENC1` plus an AES-GCM payload keyed with the same StatiCrypt secret the atlas already uses. The bytes are not a PDF. After the viewer unlocks, the PDF button decrypts that file when live export is unavailable, and the download the browser saves is a real PDF. Opening the file from disk, or fetching it without the session key, does not yield the map.

**Considered options:** (1) seal the PDF with the atlas key and decrypt in the viewer; (2) stop publishing a PDF and keep only live export; (3) leave the plaintext PDF. (2) drops the fallback the button already has when the frame cannot export. (3) is the leak.

This sits on `docs/adr/0031-published-html-is-staticrypt-ciphertext.md` and `docs/adr/0019-export-is-the-raster-extent-as-vectors.md`. The live export is still a vector PDF of the raster extent. Only the file stored next to the map is sealed.
