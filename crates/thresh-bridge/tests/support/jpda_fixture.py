"""Invented scalar observations of a two-state CV model; no oracle probabilities."""

from datetime import datetime, timedelta, timezone

import numpy as np

from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.measures import SquaredMahalanobis
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.kalman import KalmanUpdater


def components(diffusion=0.1, measurement_variance=1.0):
    model = LinearGaussian(ndim_state=2, mapping=(0,), noise_covar=[[measurement_variance]])
    return KalmanPredictor(ConstantVelocity(diffusion)), KalmanUpdater(model)


def fixture(states, observations, observation_seconds, call_seconds=1.0,
            diffusion=0.1, measurement_variance=1.0):
    epoch = datetime(2025, 1, 1, tzinfo=timezone.utc)
    predictor, updater = components(diffusion, measurement_variance)
    track_order = [
        Track([GaussianState(np.asarray(state)[:, None], np.eye(2), timestamp=epoch)])
        for state in states
    ]
    detection_order = [
        Detection([[value]], timestamp=epoch + timedelta(seconds=seconds),
                  measurement_model=updater.measurement_model)
        for value, seconds in zip(observations, observation_seconds, strict=True)
    ]
    return dict(predictor=predictor, updater=updater, track_order=track_order,
                detection_order=detection_order, tracks=set(track_order),
                detections=set(detection_order),
                timestamp=epoch + timedelta(seconds=call_seconds))


def marginals(associations, data):
    assert len(associations) == len(data['track_order'])
    detection_columns = {id(detection): i + 1
                         for i, detection in enumerate(data['detection_order'])}
    rows = []
    for track in data['track_order']:
        assert any(key is track for key in associations)
        row = [0.0] * (1 + len(detection_columns))
        seen = set()
        for hypothesis in associations[track]:
            column = detection_columns[id(hypothesis.measurement)] if hypothesis else 0
            assert column not in seen
            seen.add(column)
            row[column] = float(hypothesis.probability)
        assert 0 in seen
        rows.append(row)
    return rows


def prediction_context(associations, data):
    track = data['track_order'][0]
    detection = data['detection_order'][0]
    detected = next(hypothesis for hypothesis in associations[track] if hypothesis)
    missed = next(hypothesis for hypothesis in associations[track] if not hypothesis)
    assert detected.measurement is detection
    assert detected.prediction.timestamp == detection.timestamp
    assert missed.prediction.timestamp == data['timestamp']
    assert missed.measurement.timestamp == data['timestamp']
    return (float(detected.prediction.state_vector[0, 0]),
            float(missed.prediction.state_vector[0, 0]),
            detected.prediction.timestamp != missed.prediction.timestamp)


def gate_geometry(data):
    track = data['track_order'][0]
    detection = data['detection_order'][0]
    prediction = data['predictor'].predict(track, timestamp=detection.timestamp)
    measurement = data['updater'].predict_measurement(prediction, detection.measurement_model)
    return (float(SquaredMahalanobis()(measurement, detection)),
            float(PDAHypothesiser._gate_threshold(0.99, 1)))
