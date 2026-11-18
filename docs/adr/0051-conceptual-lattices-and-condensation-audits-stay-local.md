# Conceptual lattices and condensation audits stay local

The citation points already stay in gitignored `GEOJSON/`. The conceptual lattices (`conceptual/{age}_ma_conceptual.geojson`) and the condensation audits (`CONDENSED/`) are built from those points. Committing them put the same private table on the public repository in another shape. Both folders are gitignored. A teammate copies them in before generating maps. A check that reads one of those folders is skipped when the folder is absent.

**Considered options:** (1) stop tracking both folders; (2) keep them in git so a clone can regenerate with no extra files; (3) track only the audits. (2) republishes the citations. (3) still publishes the lattices, which are the expert reading of those points.

This sits on the existing choice that `GEOJSON/` is not versioned. The published maps stay in the repository. The password that seals them is unchanged: `STATICRYPT_PASSWORD` or `.staticrypt-password` (`docs/adr/0031-published-html-is-staticrypt-ciphertext.md`, `docs/adr/0050-published-pdfs-are-ciphertext.md`).
