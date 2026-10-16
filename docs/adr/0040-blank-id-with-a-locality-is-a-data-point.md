# A blank ID with a locality is a data point

`is_conceptual_point` treated a null or `N/A` ID as the conceptual grid. That was true of the old export, where those rows also had no country, basin or formation. The October table leaves ID null on real localities from 125 Ma through 145 Ma (120 rows: Picún Leufú, Vaca Muerta, and the rest), so those citations were not drawn and skipped condensation. A row is conceptual only when the ID is blank and country, basin and formation are all blank. A paleozone lattice anchor is built that way. A named locality with a blank ID is a data point: it is drawn, it votes in condensation, and Color stats count it.

**Considered options:** (1) keep the ID-only test; (2) require a blank locality as well. (1) keeps hiding the October rows the reviewers moved. (2) is the run. Supersedes the ID encoding in `docs/adr/0002-conceptual-points-are-not-drawn.md`. Conceptual anchors are still not drawn.
