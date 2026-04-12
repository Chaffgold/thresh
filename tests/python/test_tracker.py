import thresh_py

def test_create_tracker():
    tracker = thresh_py.PyMultiObjectTracker(10.0, 100.0)
    assert tracker.alive_count() == 0
    assert tracker.confirmed_count() == 0

def test_step_creates_tracks():
    tracker = thresh_py.PyMultiObjectTracker(10.0, 100.0)
    det = [[100.0, 200.0, 50.0]]
    for _ in range(6):
        tracker.step(det, 1.0)
    assert tracker.alive_count() >= 1

def test_get_tracks_returns_states():
    tracker = thresh_py.PyMultiObjectTracker(10.0, 100.0)
    for _ in range(6):
        tracker.step([[100.0, 200.0, 50.0]], 1.0)
    tracks = tracker.get_tracks()
    assert len(tracks) >= 1
    t = tracks[0]
    assert hasattr(t, 'id')
    assert hasattr(t, 'position')
    assert hasattr(t, 'velocity')
    assert hasattr(t, 'is_confirmed')
    assert len(t.position) == 3
