# Validation

Held-out scoring of the class IDW against the citations, and the sensitivity of
the score to the weight of the conceptual maps. The scripts reuse the functions
of `render_map.py` (condensation, anchor weights, rotation, class IDW), so they
score the method the maps use.

## Run

From the repository root, with the private data in the local folders:

```bash
python -m validation.leave_out                  # held-out citation, by basin, 10 degree block
python -m validation.mesh_weight_sensitivity    # mesh weight 0, 0.1, 0.25, 0.5, 0.75
```

Options (both scripts): `--geojson-dir` (default `GEOJSON/`), `--conceptual-dir`
(default `conceptual/`), `--out` (default `VALIDATION_OUT/`), `--ages`.

## What they do

- `leave_out`: each target (a condensed group of citations with one winning
  class) is withdrawn with the whole group, then predicted from the remaining
  anchors. Schemes: one citation group (`loo`), one basin (`bacia`), one 10
  degree block (`bloco10`). Configurations, with and without the mesh and with
  and without the KNN reclassification, are in `common.CONFIGS`; `M_metodo_B`
  is the method of the maps (citation `Weight`, mesh at 0.5, no KNN). Cohen's
  kappa and accuracy come with a 95% bootstrap interval.
- `mesh_weight_sensitivity`: method B with the mesh weight varied one factor at
  a time; kappa for the held-out citation and the held-out basin, plus the dry
  class recall.

## Results and privacy

Results go to `VALIDATION_OUT/` (`metricas.json`, `sensibilidade_peso_malha.json`),
which `.gitignore` excludes. They hold aggregate metrics only, no coordinates,
authors or references. The data folders (`GEOJSON/`, `conceptual/`) are
read-only inputs and never go in a commit of this folder.

## Tests

`pytest tests/test_validation.py` uses fictitious points generated in the test.
