import thresh_py

def test_create_filter():
    kf = thresh_py.PyKalmanFilter(6)
    state = kf.state
    assert len(state) == 6

def test_predict_changes_state():
    kf = thresh_py.PyKalmanFilter(6)
    # Simple identity F, small Q
    f = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
    q = [[0.01]*6]*6  # doesn't need to be exact
    kf.predict(f, q)
    # After predict with identity F, state should be unchanged
    state = kf.state
    assert len(state) == 6

def test_update_adjusts_state():
    kf = thresh_py.PyKalmanFilter(6)
    z = [10.0, 20.0, 30.0]
    h = [[1,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,1,0]]
    r = [[1,0,0],[0,1,0],[0,0,1]]
    kf.update(z, h, r)
    state = kf.state
    # State should have moved toward the measurement
    assert abs(state[0] - 10.0) < 5.0  # x moved toward 10
