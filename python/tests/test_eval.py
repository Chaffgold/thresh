"""Tests for the Phase 9 eval harness Python side (holdout split + the
eval-tracker output parser). Torch-free; runs in the default CI lane."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from eval.holdout_split import region_key, split_parquet, split_regions
from eval.run_tracker import parse_eval_output


def test_region_key_grids() -> None:
    assert region_key(47.6, -122.3, cell_deg=1.0) == region_key(47.1, -122.9, cell_deg=1.0)
    assert region_key(47.6, -122.3, cell_deg=1.0) != region_key(33.4, -112.0, cell_deg=1.0)


def test_split_regions_is_disjoint_and_by_region() -> None:
    # Three regions: KSEA-ish (A,B share a region), KPHX-ish (C), KJFK-ish (D).
    traj_to_region = {"A": "47,-122", "B": "47,-122", "C": "33,-112", "D": "40,-73"}
    split = split_regions(traj_to_region, train_fraction=0.66, seed=42)

    assert split.train_regions.isdisjoint(split.test_regions)
    assert split.train_regions | split.test_regions == {"47,-122", "33,-112", "40,-73"}
    assert split.train_icao24.isdisjoint(split.test_icao24)
    assert split.train_icao24 | split.test_icao24 == {"A", "B", "C", "D"}
    # A and B share a region → never split apart.
    assert ("A" in split.train_icao24) == ("B" in split.train_icao24)
    # At least one region held out.
    assert len(split.test_regions) >= 1
    # Deterministic.
    assert split_regions(traj_to_region, train_fraction=0.66, seed=42) == split


def test_split_parquet(tmp_path: Path) -> None:
    path = tmp_path / "traj.parquet"
    table = pa.table(
        {
            "icao24": pa.array(["aaa", "aaa", "bbb", "ccc"], pa.string()),
            "lat": pa.array([47.6, 47.5, 33.4, 40.6], pa.float64()),
            "lon": pa.array([-122.3, -122.2, -112.0, -73.8], pa.float64()),
        }
    )
    pq.write_table(table, path)
    split = split_parquet(path, train_fraction=0.66, seed=1)
    assert split.train_icao24 | split.test_icao24 == {"aaa", "bbb", "ccc"}
    assert split.train_icao24.isdisjoint(split.test_icao24)


def test_parse_eval_output() -> None:
    text = (
        "note: some stderr-like line\n"
        '{"scenario":"single-cruise","mota":0.904,"motp":20.59,"idf1":0.949,'
        '"id_switches":0,"frames":301,"distance_threshold_m":50.0}\n'
        '{"scenario":"maneuvering","mota":0.890,"motp":36.67,"idf1":0.942,'
        '"id_switches":0,"frames":301,"distance_threshold_m":50.0}\n'
    )
    results = parse_eval_output(text)
    assert len(results) == 2
    assert results[0]["scenario"] == "single-cruise"
    assert abs(float(results[0]["mota"]) - 0.904) < 1e-9
    assert results[1]["scenario"] == "maneuvering"
