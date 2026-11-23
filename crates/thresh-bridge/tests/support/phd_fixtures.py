"""Invented PHD inputs and inspection only; numerical expectations live in Rust."""

from datetime import datetime, timedelta, timezone

import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import ConstantVelocity, RandomWalk
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import Prediction
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.update import GaussianMixtureUpdate, Update
from stonesoup.updater.kalman import KalmanUpdater


def make_scan(kind, prior_weights, birth_weights, measurements):
    previous = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamp = previous + timedelta(seconds=1)
    mean = [[0.0], [1.0]] if kind == "cv" else [[0.0]]
    transition = ConstantVelocity(0.0) if kind == "cv" else RandomWalk(0.0)
    model = LinearGaussian(len(mean), [0], np.array([[1.0]]))

    def component(weight, tag, time):
        return TaggedWeightedGaussianState(
            np.array(mean), np.eye(len(mean)), timestamp=time, weight=weight, tag=tag
        )

    return {
        "predictor": KalmanPredictor(transition),
        "updater": KalmanUpdater(model),
        "prior": [component(weight, f"prior-{index}", previous)
                  for index, weight in enumerate(prior_weights)],
        "births": [component(weight, "birth", timestamp) for weight in birth_weights],
        "detections": {Detection(np.array([[value]]), timestamp=timestamp,
                                 measurement_model=model) for value in measurements},
        "timestamp": timestamp,
    }


def numeric_components(result, timestamp):
    return [(float(component.weight), component.state_vector[:, 0].tolist(),
             component.covar.flatten().tolist(), component.tag,
             component.timestamp == timestamp) for component in result.components]


def add_ancestry(scan):
    """Exercise Prediction's mutation of a prior update's reachable ancestry."""
    ancestor = scan["prior"][0]
    old_prediction = Prediction.from_state(ancestor, prior=ancestor)
    hypothesis = SingleHypothesis(old_prediction, next(iter(scan["detections"])))
    scan["prior"][0] = Update.from_state(ancestor, hypothesis=hypothesis)


def share_prior_arrays_and_ancestry(scan):
    """Keep two equal components distinct while sharing their reachable graph."""
    first, second = scan["prior"]
    ancestor = TaggedWeightedGaussianState(
        np.array([[-1.0]]), np.array([[2.0]]), timestamp=first.timestamp,
        weight=0.25, tag="ancestor",
    )
    first = Prediction.from_state(first, prior=ancestor)
    second = Prediction.from_state(second, prior=ancestor, tag=first.tag)
    second.state_vector = first.state_vector
    second.covar = first.covar
    scan["prior"][:] = [first, second]
    assert first is not second
    assert first.state_vector is second.state_vector
    assert first.covar is second.covar
    assert first.prior is second.prior is ancestor
    assert first.tag == second.tag


def state_snapshot(state):
    fields = (id(state), state.tag, float(state.weight), state.timestamp,
              id(state.state_vector), state.state_vector.dtype.str,
              state.state_vector.shape, state.state_vector.tobytes(),
              id(state.covar), state.covar.dtype.str, state.covar.shape,
              state.covar.tobytes())
    if hasattr(state, "hypothesis"):
        fields += (id(state.hypothesis), state_snapshot(state.hypothesis.prediction))
    if getattr(state, "prior", None) is not None:
        fields += (type(state._property_prior).__name__, state_snapshot(state.prior))
    return fields


def snapshot(scan):
    detections = tuple(sorted((id(item), item.timestamp, id(item.state_vector),
                               item.state_vector.tobytes(), id(item.measurement_model))
                              for item in scan["detections"]))
    return (id(scan["prior"]), tuple(map(state_snapshot, scan["prior"])),
            id(scan["births"]), tuple(map(state_snapshot, scan["births"])),
            id(scan["detections"]), detections, id(scan["timestamp"]))


def provenance_uses_detached_components(result, scan):
    if not isinstance(result, GaussianMixtureUpdate):
        return False
    originals = {id(component) for component in scan["prior"] + scan["births"]}
    arrays = [component.state_vector for component in scan["prior"] + scan["births"]]
    arrays += [component.covar for component in scan["prior"] + scan["births"]]
    for group in result.hypothesis:
        for hypothesis in group:
            prediction = hypothesis.prediction
            if hypothesis and not any(hypothesis.measurement is detection
                                      for detection in scan["detections"]):
                return False
            if id(prediction) in originals:
                return False
            if any(np.shares_memory(prediction.state_vector, array) or
                   np.shares_memory(prediction.covar, array) for array in arrays):
                return False
    for component in result.components:
        if any(np.shares_memory(component.state_vector, array) or
               np.shares_memory(component.covar, array) for array in arrays):
            return False
    return True
