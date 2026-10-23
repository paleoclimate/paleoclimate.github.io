# Reliability-weighted class IDW replaces the KNN reclassification

Before the class IDW, every anchor was reclassified by KNN: the inverse-distance mean of the class codes (1 Dry, 2 Semi-arid, 3 Humid) of its 8 nearest anchors, cut into equal thirds of [1, 3]. That treats categories as numbers. The mean of a Dry and a Humid neighbour is 2, so Semi-arid appeared where no citation says Semi-arid. It also weighted the neighbours by distance a second time, since the class IDW already does that. Every citation counted the same, whatever its reliability, and a conceptual anchor counted as much as a citation.

The interpolator now skips the KNN step. Each anchor adds `weight / d^power` to its own class and the cell takes the class with the largest sum; the class code never enters the arithmetic. A condensed anchor carries the mean `Weight` (citation reliability, 1 to 10, from the research group's list of paleo-indicators) of the rows that won its vote, so stacked citations at one site do not add up. A missing or invalid `Weight` counts as 1. A conceptual anchor carries `CONCEPTUAL_ANCHOR_WEIGHT = 0.5`, half the weight of the least reliable citation, because it is an interpretation drawn over the same citations, not an observation. Power 4, the 0.5° condensation radius, the 0.1° grid and the 2.0° edge smooth are unchanged. A second GeoTIFF in `GENERATED_CONFIDENCE/` stores the winning class's share of the weight before smoothing (1/3 to 1). `--knn` reproduces the previous pipeline and `--no-weights` gives every anchor weight 1. The file names keep their `knn_idw` tag so the viewer and the comparison scripts still find them; the caption reads "Reliability-weighted class IDW".

**Validation** (17 ages, 1 241 condensed citation groups with a single winning class, prediction at the withheld group without the edge smooth, Cohen's kappa with a 95% bootstrap interval):

| | Leave one out | Leave one basin out |
|---|---|---|
| Weighted class IDW (this decision) | 0.512 [0.469; 0.557] | 0.457 [0.415; 0.500] |
| Previous pipeline (KNN, weights 1) | 0.436 | 0.384 |
| Weighted class IDW without the conceptual anchors | 0.403 [0.353; 0.453] | 0.214 [0.167; 0.259] |

Macro IoU with the conceptual maps is 0.749 against 0.602 before.

**Sensitivity of the conceptual weight.** One factor at a time, with everything else fixed: from 0.1 to 0.75 the kappa runs from 0.491 to 0.517 (leave one out) and from 0.415 to 0.468 (leave one basin out), with overlapping intervals. The jump is between 0 and 0.1. The result depends on having the conceptual anchors, not on the exact weight. The domain is [0, 1) on principle: an interpretation must not weigh as much as the weakest citation (`Weight` 1), so weights of 1 and above were not tried and are not an option. The citation weights move kappa by at most 0.02; they stay because a more reliable citation should pull harder.

**Circularity.** The conceptual maps were drawn by the group while looking at all the citations, including the one withheld in each round. Part of the withheld citation's information is still in the lattice, so the gain from the conceptual anchors is partly circular, and the IoU against those same maps is circular by construction. The run without the conceptual anchors is the floor and is reported next to the main result.

**Considered options:** (1) weighted class IDW without KNN; (2) keep the KNN on the class codes; (3) KNN as a vote per class instead of a mean; (4) a KNN classifier trained on the conceptual maps of the other ages, used as the lattice; (5) per-class calibration factors on the IDW sums; (6) ordinary indicator kriging. (2) is the problem above and scores lower. (3) and (4) do not beat (1) and keep two interpolators in series. (5) raises the scores by tuning to the validation set with no physical reason for the factors, so it was rejected. (6) ties with (1) on citations alone (kappa 0.409 against 0.403) and wipes out most of the Dry class (recall 0.37 against 0.52); it stays as a benchmark only.

**Consequences:** Every published raster changes. Calling the interpolation functions without `weights` gives exactly the previous result, so older callers and tests keep working. The confidence rasters reveal where the data sit, so `GENERATED_CONFIDENCE/` is ignored by git and stays local. The validation scripts and their aggregated results live outside the repository, with the private data.
