# Color stats counts real data points

The raster-coverage panel reports how the interpolated surface splits into Dry, Semi-arid and Humid. It now also shows how many data points went into that map. The count is the source records with a formation identity and a climate class: conceptual points (ID null or N/A) are left out, and condensation is not applied. A cluster of nearby citations still counts as every citation in it.

**Considered options:** (1) this census of data points; (2) the condensed anchors that enter the interpolator; (3) every interpolator anchor, conceptual points included. (2) hides how many citations were merged. (3) counts the conceptual grid the panel was asked to leave out. The class breakdown in `pontos_dado_por_idade_raio_aglutinacao_0.5_power_4.0.csv` is the same census split by climate.
