# Published maps smooth anchors with KNN before the indicator IDW

The maps published after this note also round the class-zone borders. See `0012-published-maps-round-zone-borders.md`.

Gabriel accepted the power 4, 0.5° indicator maps and asked to turn KNN on and see how it behaves. The pre-smooth uses the eight nearest other anchors, inverse-distance weights at the same power, and replaces each anchor's class with the class of that weighted mean. The cell rule stays indicator IDW: the class with the greatest `1 / distance^power`, every anchor, no snap disk. A mean that lands in the middle third of [1, 3] becomes semi-arid, which is the behavior this pre-smooth is there to show.

Power stays 4 and the condensation radius stays 0.5°. Markers stay on the original citation class. The viewer opens these KNN + IDW maps. `--gradient-sharp` is still ignored.

**Considered options:** (1) leave KNN off; (2) restore numeric IDW of the smoothed floats, with the old 12-neighbor cap and 0.15° snap disk; (3) KNN-reclass the anchors, then the same indicator IDW. (3) is the run. (2) would also turn the cell rule back into an average, which is a second change on top of the one he asked to see.
