# Published maps use power 4 and a 1.0° radius

The maps published after this note use power 2 and a 0.5° radius. See `0009-published-maps-use-power-two-and-radius-half.md`.

Power 1 left each conceptual anchor as a small disk. The semi-arid grid outvoted a dry or humid neighbor everywhere except on top of the point, so aligned conceptual points stayed separate circles. A higher power hands each anchor its local territory, which is what lets same-class neighbors meet.

This generation uses indicator IDW power 4 and condenses data points at 1.0°. The snap disk stays off. Conceptual points still skip the condensation vote. The class tables for this run are `idw_classes_por_idade_raio_aglutinacao_1.0_power_4.0.csv` and `pontos_dado_por_idade_raio_aglutinacao_1.0_power_4.0.csv`. The point census counts data points only.

**Considered options:** (1) power 2, the step discussed after the 0.5° maps; (2) power 4 with the 1.0° radius, as asked for this regeneration; (3) change the code default radius from 0.5° to 1.0°. (2) is the run. (3) stays aside so the half-degree tests still describe the function default; the published command passes `--condensation-radius 1.0`.
