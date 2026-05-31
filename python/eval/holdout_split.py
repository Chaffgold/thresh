"""Region-based train / held-out split for evaluation (task 9.1).

Splits trajectories **by geographic region** (not by individual trajectory) so a
region never leaks across the split — e.g. train on a KSEA-region set, evaluate
on a disjoint KPHX-region set (design Risk: hold out by region, not by aircraft,
so the simulator quirks stay roughly constant between train and eval).

Torch-free (pyarrow + stdlib); runs in the default CI lane. Each trajectory is
keyed by its `icao24`; its region is a coarse lat/lon grid cell of its mean
position. Regions are partitioned deterministically into train / held-out.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random

import pyarrow.parquet as pq


@dataclass
class RegionSplit:
    """A deterministic region-based split.

    Attributes:
        train_icao24: set of trajectory ids in the training regions.
        test_icao24: set of trajectory ids in the held-out regions.
        train_regions / test_regions: the region keys on each side.
    """

    train_icao24: set[str]
    test_icao24: set[str]
    train_regions: set[str]
    test_regions: set[str]


def region_key(lat: float, lon: float, cell_deg: float = 1.0) -> str:
    """Grid-cell key for a lat/lon at `cell_deg` resolution.

    Keyed by integer cell *indices* (`floor(coord / cell_deg)`), so sub-degree
    resolutions never collide — unlike formatting the scaled coordinate, where
    e.g. `47.5` and `48.0` would both round to ``"48"`` and silently merge
    distinct cells, defeating the no-leakage guarantee.
    """
    lat_idx = math.floor(lat / cell_deg)
    lon_idx = math.floor(lon / cell_deg)
    return f"{lat_idx},{lon_idx}"


def _mean_region_per_trajectory(
    icao24: list[str], lat: list[float], lon: list[float], cell_deg: float
) -> dict[str, str]:
    """Map each `icao24` to the region key of its mean lat/lon."""
    # lat/lon are non-nullable in the canonical trajectory schema.
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])  # lat, lon, n
    for ic, la, lo in zip(icao24, lat, lon, strict=True):
        s = sums[ic]
        s[0] += float(la)
        s[1] += float(lo)
        s[2] += 1.0
    out: dict[str, str] = {}
    for ic, acc in sums.items():
        lat_sum, lon_sum, n = acc[0], acc[1], acc[2]
        if n > 0:
            out[ic] = region_key(lat_sum / n, lon_sum / n, cell_deg)
    return out


def split_regions(
    traj_to_region: dict[str, str], *, train_fraction: float = 0.8, seed: int = 42
) -> RegionSplit:
    """Partition the regions (and the trajectories within them) into train /
    held-out. Deterministic given `seed`. Split is by region, so all
    trajectories in a region land on the same side."""
    regions = sorted(set(traj_to_region.values()))
    Random(seed).shuffle(regions)
    n_train = max(1, round(len(regions) * train_fraction)) if regions else 0
    # Keep at least one region for the held-out side when there are ≥2 regions.
    if len(regions) >= 2:
        n_train = min(n_train, len(regions) - 1)
    train_regions = set(regions[:n_train])
    test_regions = set(regions[n_train:])

    train_icao24 = {ic for ic, r in traj_to_region.items() if r in train_regions}
    test_icao24 = {ic for ic, r in traj_to_region.items() if r in test_regions}
    return RegionSplit(train_icao24, test_icao24, train_regions, test_regions)


def split_parquet(
    path: Path | str, *, train_fraction: float = 0.8, seed: int = 42, cell_deg: float = 1.0
) -> RegionSplit:
    """Read a canonical-trajectory Parquet and produce a region-based split."""
    cols = pq.read_table(path).to_pydict()
    traj_to_region = _mean_region_per_trajectory(
        cols["icao24"], cols["lat"], cols["lon"], cell_deg
    )
    return split_regions(traj_to_region, train_fraction=train_fraction, seed=seed)
