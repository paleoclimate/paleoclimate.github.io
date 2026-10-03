# Paleoclimate maps

A reconstruction-age atlas of climate from geological points, reconstructed coastlines, optional paleozone polygons, and optional basin outlines.

## Language

**Reconstruction age**:
A Ma snapshot (for example 65 Ma) that owns one map dataset.
_Avoid_: map, time slice, timestep

**Data point**:
A geological formation location at a reconstruction age, carrying a climate class.
_Avoid_: sample, marker, record, conceptual point

**Conceptual point**:
A climate-class location at a reconstruction age with no formation identity.
_Avoid_: data point, paleozone

**Climate class**:
The Humid (H), Semi-arid (S), or Dry (D) classification shared by a data point, a conceptual point, and a Paleozone.
_Avoid_: paleozone, climate zone, climate value

**Coastline**:
The reconstructed shoreline for a reconstruction age.
_Avoid_: costa, linha de costa, coast

**Paleozone**:
A polygonal climate region at a reconstruction age, classified with the same climate class as data points (Humid, Semi-arid, or Dry). An age may have no Paleozones.
_Avoid_: Paleozona, climate zone, climate class, polygon, basin outline

**Basin**:
The sedimentary basin named on a data point. The basin filter shows or hides markers by this name.
_Avoid_: basin outline, paleozone

**Basin outline**:
The polygonal limit of a sedimentary basin at a reconstruction age. An age may have no basin outlines.
_Avoid_: basin, bacia, paleozone
