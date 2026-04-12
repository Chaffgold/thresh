import thresh_py

def test_perfect_match():
    gt = [[(1, [0.0, 0.0, 0.0])]]
    tracks = [[(1, [0.0, 0.0, 0.0])]]
    mota, motp, idsw = thresh_py.compute_mot_metrics_py(gt, tracks, 1.0)
    assert mota == 1.0
    assert idsw == 0

def test_no_tracks_zero_mota():
    gt = [[(1, [0.0, 0.0, 0.0])]]
    tracks = [[]]
    mota, motp, idsw = thresh_py.compute_mot_metrics_py(gt, tracks, 1.0)
    assert mota == 0.0
