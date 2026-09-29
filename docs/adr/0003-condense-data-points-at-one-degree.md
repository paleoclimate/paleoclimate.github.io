# Condense data points at 1° before interpolation

Piles of citations at one locality were voting against each other inside the IDW and washing the semi-arid class out. The candidate surface condenses those citations first.

Data points of one reconstruction age cluster at 1.0° Euclidean distance in the source file, before the −13° paleo-frame rotation. The link is transitive: neighbors of neighbors stay in the group until nothing else is within 1.0°. Each row is one vote. The `Weight` column is ignored. The winning class sits at the mean longitude and latitude. An exact tie emits one anchor per tied class at that same coordinate, with no jitter. A data point whose class lost is listed in the audit and still drawn on the map in its own class.

Conceptual points do not enter the cluster or the vote. They are copied onto the interpolator as their own anchors.

**Considered options:** (1) 0.1° as in the first condensation note; (2) 1.0°, about 110 km, which is what the meeting meant; (3) break ties with `Weight` or with a spatial jitter; (4) cluster after the paleo rotation; (5) let conceptual points vote inside the radius; (6) stop the chain at basin boundaries. (2) is the radius Gabriel corrected to. Weight and basin limits were left for a later round. Jitter was declined because the markers do not move and coincident anchors are enough for the class vote. Clustering before the rotation is the frame he asked to try. Conceptual points stay out of the vote and in the interpolator.
