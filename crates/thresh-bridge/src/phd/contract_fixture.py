"""Synthetic contract doubles only; numerical reference tests use real graphs."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import RandomWalk
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.detection import Detection
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.state import TaggedWeightedGaussianState, WeightedGaussianState
from stonesoup.updater.kalman import KalmanUpdater


def component(weight=0.5, mean=0.0, variance=1.0, tag="existing", time=None):
    return TaggedWeightedGaussianState(
        state_vector=np.array([[mean]], dtype=np.float64),
        covar=np.array([[variance]], dtype=np.float64),
        weight=weight,
        tag=tag,
        timestamp=time or datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def models():
    return KalmanPredictor(RandomWalk(0.0)), KalmanUpdater(
        LinearGaussian(ndim_state=1, mapping=[0], noise_covar=np.array([[1.0]]))
    )


def fixture():
    time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    parent = component(weight=2.5, mean=-1.0, tag="ancestor", time=time)
    prior = component(time=time)
    birth = component(tag="birth", time=time)
    # Shared arrays AND shared ancestry test the one-graph deep-copy guarantee.
    birth.state_vector = prior.state_vector
    birth.covar = prior.covar
    prior.ancestry = parent
    birth.ancestry = parent
    predictor, updater = models()
    detection = Detection(np.array([[0.0]]), timestamp=time, measurement_model=updater.measurement_model)
    f = SimpleNamespace(
        prior=[prior], births=[birth], detections={detection}, time=time,
        predictor=predictor, updater=updater, calls=[], failure_stage=None,
        failure_kind=ValueError, mutate=False, working=None, hypotheses=None,
        result=None, reduced=None, parent=parent, copy_count=0,
    )

    class CopyWitness:
        def __deepcopy__(self, memo):
            f.copy_count += 1
            copied = CopyWitness()
            memo[id(self)] = copied
            return copied

    prior.copy_witness = birth.copy_witness = CopyWitness()

    def hypothesise(components, detections, timestamp):
        f.calls.append("hypothesise")
        assert detections is f.detections and timestamp is f.time
        assert components is not f.prior and components is not f.births
        assert components[0] is not prior and components[1] is not birth
        assert components[0].ancestry is components[1].ancestry
        assert components[0].ancestry is not parent
        assert components[0].state_vector is components[1].state_vector
        assert components[0].covar is components[1].covar
        assert components[0].copy_witness is components[1].copy_witness
        assert f.copy_count == 1
        assert not np.shares_memory(components[0].state_vector, prior.state_vector)
        assert not np.shares_memory(components[0].covar, prior.covar)
        f.working = components
        if f.mutate:
            components[0].state_vector[0, 0] = 999.0
            components[1].covar[0, 0] = 888.0
            components[0].tag = "mutated"
            components[1].weight = 9.0
            components[1].timestamp += timedelta(days=1)
            components[0].ancestry.state_vector[0, 0] = 777.0
            components[1].ancestry.covar[0, 0] = 666.0
            components[1].ancestry.tag = "mutated ancestor"
            components[1].ancestry.weight = 99.0
            components[1].ancestry.timestamp += timedelta(days=1)
        fail("hypothesis")
        f.hypotheses = SimpleNamespace(components=components)
        return f.hypotheses

    def update(hypotheses):
        f.calls.append("update")
        assert hypotheses is f.hypotheses
        fail("update")
        f.result = SimpleNamespace(components=hypotheses.components, hypothesis=hypotheses)
        return f.result

    def reduce(components):
        f.calls.append("reduce")
        assert components is f.result.components
        fail("reduction")
        f.reduced = list(reversed(components))
        return f.reduced

    def fail(stage):
        if stage == f.failure_stage:
            raise f.failure_kind("synthetic stonesoup numpy stage failure")

    f.hypothesiser = SimpleNamespace(hypothesise=hypothesise)
    f.phd_updater = SimpleNamespace(update=update)
    f.reducer = SimpleNamespace(reduce=reduce)
    return f


def assert_preserved(f):
    assert len(f.prior) == 1 and len(f.births) == 1 and len(f.detections) == 1
    a, b, p = f.prior[0], f.births[0], f.parent
    assert a.tag == "existing" and b.tag == "birth" and p.tag == "ancestor"
    assert a.weight == 0.5 and b.weight == 0.5 and p.weight == 2.5
    assert a.timestamp == b.timestamp == p.timestamp == f.time
    assert a.state_vector is b.state_vector and a.covar is b.covar
    assert a.ancestry is b.ancestry is p
    np.testing.assert_array_equal(a.state_vector, [[0.0]])
    np.testing.assert_array_equal(a.covar, [[1.0]])
    np.testing.assert_array_equal(p.state_vector, [[-1.0]])
    np.testing.assert_array_equal(p.covar, [[1.0]])


class CopyFailure(TaggedWeightedGaussianState):
    def __deepcopy__(self, memo):
        raise RuntimeError("synthetic component copy failure")


class UnexpectedWeight:
    def __float__(self):
        raise ValueError("synthetic weight inspection failure")


class CountedCollection:
    def __init__(self, values):
        self.values = values
        self.advances = 0

    def __iter__(self):
        for value in self.values:
            self.advances += 1
            yield value


class UnwritableUpdate:
    def __init__(self, components, hypothesis):
        self._components = components
        self.hypothesis = hypothesis

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, value):
        raise ValueError("synthetic result assignment failure")


def unwritable_result(f):
    def update(hypotheses):
        f.calls.append("update")
        f.result = UnwritableUpdate(hypotheses.components, hypotheses)
        return f.result
    f.phd_updater.update = update


def with_copy_failure(f):
    f.prior = [CopyFailure(np.array([[0.0]]), np.array([[1.0]]), timestamp=f.time,
                           weight=0.5, tag="existing")]


def malformed_cases():
    return [
        "f.prior = None",
        "f.births = 42",
        "f.detections = None",
        "f.prior = iter(f.prior)",
        "f.detections = iter(f.detections)",
        "f.prior = [object()]",
        "f.births = [WeightedGaussianState(np.array([[0.0]]), np.array([[1.0]]), timestamp=f.time, weight=.5)]",
        "f.prior[0].tag = 'birth'",
        "f.births[0].tag = 'other'",
        "f.prior[0].tag = None",
        "del f.prior[0].tag",
        "f.prior[0].weight = -0.1",
        "f.births[0].weight = float('nan')",
        "f.prior[0].weight = float('inf')",
        "f.prior[0].weight = object()",
        "f.prior[0].state_vector = np.array([[float('nan')]])",
        "f.prior[0].covar = np.array([[float('inf')]])",
        "f.prior[0].state_vector = np.array([[0.0, 1.0]])",
        "f.prior[0].covar = np.eye(2)",
        "f.births[0].state_vector = np.array([[0.0], [1.0]]); f.births[0].covar = np.eye(2)",
        "f.time = None",
        "f.time = f.time.replace(tzinfo=None)",
        "f.time = f.time.astimezone(timezone(timedelta(hours=1)))",
        "f.prior[0].timestamp = None",
        "f.prior[0].timestamp = f.time.replace(tzinfo=None)",
        "f.prior[0].timestamp = f.time + timedelta(seconds=1)",
        "f.prior.append(component(tag='second', time=f.time - timedelta(seconds=1)))",
        "f.births[0].timestamp = f.time - timedelta(seconds=1)",
        "next(iter(f.detections)).timestamp = None",
        "next(iter(f.detections)).timestamp = f.time + timedelta(seconds=1)",
        "f.detections = {object()}",
        "next(iter(f.detections)).state_vector = np.array([[float('nan')]])",
        "next(iter(f.detections)).state_vector = np.array([[0.0], [1.0]])",
        "f.detections.add(Detection(np.array([[0.0], [1.0]]), timestamp=f.time))",
    ]


def malformed_fixture(source):
    f = fixture()
    exec(source, globals(), {"f": f})
    return f


def reducer_inputs(values):
    return [component(weight=w, mean=m, variance=v, tag=str(i)) for i, (w, m, v) in enumerate(values)]


def numeric_components(components):
    return sorted((float(c.weight), float(c.mean[0, 0]), float(c.covar[0, 0])) for c in components)
