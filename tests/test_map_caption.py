"""Viewer caption of the class IDW maps, with and without --knn."""

from __future__ import annotations

import render_map as rm

NAME = 'map_100_ma_knn_idw_power4.0_smooth2.0_class-thirds_kdtree.html'


def test_default_caption_names_the_weighted_class_idw():
    caption = rm._map_caption(NAME, rm.METHOD_KDTREE)
    assert caption.startswith(rm.WEIGHTED_IDW_CAPTION)
    assert 'KNN' not in caption


def test_knn_run_is_not_labelled_as_the_weighted_idw():
    caption = rm._map_caption(NAME, rm.METHOD_KDTREE, knn=True)
    assert caption.startswith(rm.KNN_IDW_CAPTION)
    assert rm.WEIGHTED_IDW_CAPTION not in caption
