# Indicator IDW assigns a class, it does not average one

Dry beside humid is a real pattern (altitude, for example). A numeric IDW of Dry=1 and Humid=3 paints 2, and 2 is semi-arid, along every transition. Both reviewers rejected that for this candidate.

Each grid cell sums `1 / distance^power` per climate class across every interpolator anchor (condensed data points and conceptual points). The cell takes the class with the greatest weight. Power is 1. There is no cap of 12 neighbors and no disk that snaps a cell to the nearest anchor. Semi-arid appears only where semi-arid anchors outweigh the others. When weights tie, the cell follows the nearest tied anchor, and a remaining tie prefers Dry, then Humid, then Semi-arid, so the balance of dry and humid does not become semi-arid.

The raster stores class codes 1, 2 and 3. The map paints each code with that class's solid color. Equal thirds of the [1, 3] span are the classifier: below 5/3 dry, below 7/3 semi-arid, otherwise humid. `gradient_sharp` is no longer applied.

**Considered options:** (1) numeric IDW plus a wider color ramp; (2) indicator IDW as above; (3) slice the color scale as 0–1 / 1–2 / 2–3; (4) keep the 0.15° snap disk and the 12-neighbor cap; (5) drop conceptual points from the interpolator. (1) still invents semi-arid between dry and humid. (3) would put a dry anchor (value 1) on the semi-arid boundary. Conceptual points stay, because they are how the surface approaches the paleozone belts. KNN smoothing stays in the code and is not part of this candidate.
