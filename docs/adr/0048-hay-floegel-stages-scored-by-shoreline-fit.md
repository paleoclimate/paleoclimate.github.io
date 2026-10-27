# The other Hay & Floegel stages are scored by shoreline fit

"Hay" and "Floegel" in this repository are one source: Hay & Floegel (2012), *New thoughts about the Cretaceous climate and oceans*, Earth-Science Reviews 115, 262–272. Its Figs. 2–7 reprint the climate-zone maps of Chumakov et al. (1995) for six stages. The Floegel comparison used two of them (Albian, Fig. 4, at 105 Ma; Aptian, Fig. 3, at 115 Ma) with hand-picked control points. The `hay` atlas slot of `docs/adr/0030-hay-comparison-waits-for-the-images.md` waited for images that are this same paper; it scores nothing.

`compare_chumakov.py` adds the other four stages at the PCVS age nearest the middle of each stage: Berriasian (Fig. 2) at 140 Ma, Cenomanian (Fig. 5) at 95 Ma, Santonian (Fig. 6) at 85 Ma and Maastrichtian (Fig. 7) at 70 Ma. The figures are vector drawings in an equirectangular frame with a 30° graticule and no coordinate labels. The graticule spacing gives the scale. The corner is taken as 180° W, 90° N, which is exact for the full-globe Berriasian map; the position comes from the shoreline fit of `docs/adr/0046-burgener-comparison-georeferenced-by-frame.md`, preceded by a coarse scan of ±45° in longitude and ±30° in latitude whose three best separated minima are refined. Classification uses the Floegel legend prototypes and de-para (greens Semi-arid, yellow and tan Dry, pink Humid), so the six stages are scored the same way. The reports carry the Floegel names (`comparison_report_{age}.html`), so the viewer's "Comparison Hay & Flögel (2012)" button covers the new ages.

`extract` runs once, from the PDF, and keeps only a crop around the raster window (`COMPARISON/inputs/{age}_chumakov.png`) plus the georeference (`chumakov_georef.json`): only the georeference table is committed; the crops stay local because the paper is © Elsevier and not open access (`docs/adr/0049-elsevier-figures-stay-local.md`). `compare` refits on the crop.

The vector figures fill some zones with paler tints than the scanned legend (pale pink humid zone, pale yellow arid zone, light greens); those tints are added as prototypes of the same classes, otherwise the pale pink is as close to paper white as to the legend pink and the humid zone drops out.

**Results** (rasters with the per-age frame correction, 26/10/2026):

| Age | Stage (figure) | Georeference | Accuracy | Kappa |
|---|---|---|---|---|
| 140 Ma | Berriasian (2) | shoreline fit | 37.6% | 0.021 |
| 115 Ma | Aptian (3) | control points | 9.4% | 0.011 |
| 105 Ma | Albian (4) | control points | 21.2% | 0.084 |
| 95 Ma | Cenomanian (5) | shoreline fit | 17.1% | −0.003 |
| 85 Ma | Santonian (6) | shoreline fit | 25.6% | 0.074 |
| 70 Ma | Maastrichtian (7) | shoreline fit | 35.9% | 0.171 |

Agreement is low at every stage, as at 105 and 115 Ma. The Chumakov maps call most of the compared land arid (52% to 91% Dry), where the PCVS is mostly Semi-arid and Humid.

**Check against the control points.** `compare --check` scores the Albian and Aptian maps by the automatic georeference, without writing reports: 13.0% / kappa 0.014 at 105 Ma (control points 21.2% / 0.084) and 10.5% / 0.014 at 115 Ma (control points 9.4% / 0.011). A rigid rotation and shift cannot bend Chumakov's South America onto ours, while the thin-plate spline of the control points can. The two methods differ by a few points of accuracy and both stay near chance, so the conclusion does not depend on the georeference.

**Considered options:** (1) automatic georeference by graticule and shoreline; (2) hand-picked control points for four more ages in the GCP picker; (3) leave the four stages out. (2) is the method of 105 and 115 Ma but is manual and not reproducible by the scorer. (3) leaves most of the paper unused.

**Limitations:** one rotation and shift per age for the whole window; Chumakov's palaeogeography differs from ours, and the coastline of the figure is compared with present-day coastlines rotated to the age. The stage-to-age assignment uses the stage middle (Cohen et al., 2013); the Berriasian could also be read at 145 Ma, the base of the Cretaceous.

**Consequences:** publishing the site runs `compare_chumakov.py compare`. The comparisons now run before `index.html` is written, so the viewer links every report produced in the same run, including Burgener's.
