# Power 4 and the 0.5° radius hold under method B

Power 4 and the 0.5° condensation radius were chosen in `docs/adr/0010-published-maps-use-power-four-and-radius-half.md` by macro IoU against the paleozone polygons on 100, 120 and 130 Ma, under the pipeline before method B. That score is circular: the polygons were drawn over the same citations. `validation/power_radius_sensitivity.py` checks both with method B (`docs/adr/0044-reliability-weighted-class-idw-replaces-knn.md`) and the held-out citation, one factor at a time, on the 17 ages with the per-age frame correction.

| Power (radius 0.5°) | Leave one out | Leave one basin out | Dry recall (LOO / basin) |
|---|---|---|---|
| 1 | 0.379 [0.332; 0.424] | 0.249 [0.202; 0.293] | 0.31 / 0.05 |
| 2 | 0.489 [0.445; 0.536] | 0.393 [0.350; 0.436] | 0.56 / 0.34 |
| 3 | 0.507 [0.466; 0.551] | 0.457 [0.417; 0.499] | 0.61 / 0.49 |
| **4** | **0.515 [0.472; 0.560]** | **0.460 [0.417; 0.501]** | 0.63 / 0.52 |
| 5 | 0.516 [0.472; 0.559] | 0.461 [0.418; 0.504] | 0.64 / 0.53 |
| 6 | 0.510 [0.467; 0.554] | 0.458 [0.416; 0.501] | 0.64 / 0.53 |
| 8 | 0.514 [0.469; 0.559] | 0.457 [0.415; 0.499] | 0.65 / 0.54 |

Kappa rises steeply from power 1 to 3 and is flat from 3 to 8, with overlapping intervals. Power 4 sits on that plateau; 5 differs by 0.001. A higher power keeps more of the dry class, which is the rarest, but the gain from 4 to 8 is two to three points of recall and does not move kappa.

| Radius (power 4) | Targets | Leave one out | Leave one basin out |
|---|---|---|---|
| 0.25° | 1 400 | 0.520 [0.480; 0.559] | 0.452 [0.412; 0.489] |
| **0.5°** | **1 241** | **0.515 [0.472; 0.560]** | **0.460 [0.417; 0.501]** |
| 1.0° | 1 073 | 0.505 [0.457; 0.551] | 0.499 [0.454; 0.546] |

The radius changes the condensed groups, so each row scores a different set of targets and the rows are not strictly comparable. Leave one out barely moves. Leave one basin out is higher at 1.0°, but 1.0° is the radius rejected in `docs/adr/0007-condensation-radius-is-half-a-degree.md` because it merges too many citations into one vote; with fewer, larger groups the held-out basin also loses fewer anchors.

**Decision:** keep power 4 and the 0.5° radius. Both are inside the stable range of the held-out score; nothing here justifies a change that would republish every map.

**Considered options:** (1) keep 4 and 0.5°; (2) move to power 5, the numerical maximum; (3) move to 1.0°. (2) gains 0.001, inside the interval. (3) changes the targets, so its score is not a like-for-like gain, and it reintroduces the merging problem of ADR 0007.

**Consequences:** the article and the report can state that the power and radius were checked against held-out citations under the final method, not only by IoU against the conceptual maps. The edge smooth (σ 2.0°) acts after the class decision and is not part of this check.
