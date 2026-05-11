"""Tests for partitioned Parquet storage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from acquisition.schema import TrajectoryRecord
from acquisition.storage import read_partition, write_partition


def _ts_us(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    return int(
        datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1_000_000
    )


def _rec(icao: str, ts_us: int, **overrides: object) -> TrajectoryRecord:
    base: dict[str, object] = {
        "icao24": icao,
        "timestamp_us": ts_us,
        "lat": 47.45,
        "lon": -122.31,
        "source": "opensky",
    }
    base.update(overrides)
    return TrajectoryRecord(**base)  # type: ignore[arg-type]


class TestRoundTrip:
    def test_synthetic_stream_round_trips(self, tmp_path: Path) -> None:
        """Spec scenario: round-trip a tiny synthetic stream through schema and writer."""
        t0 = _ts_us(2026, 5, 11, 12, 0)
        records = [
            _rec(
                "a3c8b7",
                t0 + i * 1_000_000,
                alt_geom_m=3000.0,
                vel_ground_mps=120.0,
                track_deg=45.0,
                callsign="UAL123",
                category="A4",
                quality_nic=8,
                quality_nac_p=10,
            )
            for i in range(5)
        ]

        written = write_partition(records, root=tmp_path, source="opensky")
        assert len(written) == 1
        assert written[0].parent.name == "date=2026-05-11"
        assert written[0].parent.parent.name == "source=opensky"

        read_back = read_partition(written[0])
        assert len(read_back) == len(records)
        # Trajectories non-empty and timestamps monotonic per icao (spec).
        assert read_back
        timestamps = [r.timestamp_us for r in read_back if r.icao24 == "a3c8b7"]
        assert all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1))
        assert all(r.source == "opensky" for r in read_back)

    def test_provenance_round_trips(self, tmp_path: Path) -> None:
        rec = _rec(
            "a3c8b7",
            _ts_us(2026, 5, 11),
            provenance={"lat": True, "lon": False},
            source="adsbx",
        )
        written = write_partition([rec], root=tmp_path, source="adsbx")
        read_back = read_partition(written[0])
        assert read_back[0].provenance == {"lat": True, "lon": False}


class TestPartitioning:
    def test_records_spanning_two_days_get_two_files(self, tmp_path: Path) -> None:
        day1 = _ts_us(2026, 5, 11, 23, 30)
        day2 = _ts_us(2026, 5, 12, 0, 30)
        records = [_rec("a3c8b7", day1), _rec("a3c8b7", day2)]
        written = write_partition(records, root=tmp_path, source="opensky")
        assert len(written) == 2
        partition_names = {p.parent.name for p in written}
        assert partition_names == {"date=2026-05-11", "date=2026-05-12"}


class TestMonotonicity:
    def test_unsorted_input_is_sorted_within_partition(self, tmp_path: Path) -> None:
        """The writer sorts within each (icao, day) bucket before writing."""
        t0 = _ts_us(2026, 5, 11, 12, 0)  # midday — sub-second math stays in-day
        records = [
            _rec("a3c8b7", t0),
            _rec("a3c8b7", t0 - 1_000_000),  # earlier in the same day
            _rec("a3c8b7", t0 + 1_000_000),
        ]
        written = write_partition(records, root=tmp_path, source="opensky")
        assert len(written) == 1
        # Verify the on-disk order is sorted by timestamp.
        read_back = read_partition(written[0])
        timestamps = [r.timestamp_us for r in read_back]
        assert timestamps == sorted(timestamps)


class TestSourceMismatch:
    def test_writer_rejects_records_with_wrong_source(self, tmp_path: Path) -> None:
        rec = _rec("a3c8b7", _ts_us(2026, 5, 11), source="adsbx")
        with pytest.raises(ValueError, match="source mismatch"):
            write_partition([rec], root=tmp_path, source="opensky")
