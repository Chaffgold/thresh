"""Tests for per-ICAO track stitching."""

from __future__ import annotations

from acquisition.schema import TrajectoryRecord
from acquisition.stitching import stitch_tracks


def _rec(icao: str, t_s: int) -> TrajectoryRecord:
    return TrajectoryRecord(
        icao24=icao,
        timestamp_us=t_s * 1_000_000,
        lat=0.0,
        lon=0.0,
        source="opensky",
    )


def test_spec_scenario_splits_on_60s_gap() -> None:
    """Spec scenario: [t0, t0+10, t0+20, t0+200, t0+210] → two tracks."""
    t0 = 1_000_000
    records = [_rec("a3c8b7", t0 + dt) for dt in (0, 10, 20, 200, 210)]

    tracks = list(stitch_tracks(records, gap_threshold_s=60.0))

    assert len(tracks) == 2
    assert [r.timestamp_us for r in tracks[0].records] == [
        (t0 + dt) * 1_000_000 for dt in (0, 10, 20)
    ]
    assert [r.timestamp_us for r in tracks[1].records] == [
        (t0 + dt) * 1_000_000 for dt in (200, 210)
    ]


def test_track_id_uses_first_timestamp() -> None:
    t0 = 1_700_000_000  # 2023-11-14T22:13:20Z
    records = [_rec("a3c8b7", t0), _rec("a3c8b7", t0 + 5)]
    tracks = list(stitch_tracks(records))
    assert tracks[0].track_id.startswith("a3c8b7-")
    assert "2023-11-14T22:13:20" in tracks[0].track_id


def test_multiple_icaos_split_independently() -> None:
    from acquisition.stitching import Track

    t0 = 1_000_000
    records = [
        _rec("a3c8b7", t0),
        _rec("a3c8b7", t0 + 10),
        _rec("b1d2e3", t0 + 5),
        _rec("b1d2e3", t0 + 100),  # gap > 60s → split for this ICAO
    ]
    tracks: list[Track] = list(stitch_tracks(records, gap_threshold_s=60.0))
    a_tracks = [t for t in tracks if t.icao24 == "a3c8b7"]
    b_tracks = [t for t in tracks if t.icao24 == "b1d2e3"]

    assert len(a_tracks) == 1
    assert len(b_tracks) == 2


def test_unsorted_input_is_sorted_internally() -> None:
    t0 = 1_000_000
    records = [_rec("a3c8b7", t0 + 20), _rec("a3c8b7", t0), _rec("a3c8b7", t0 + 10)]
    tracks = list(stitch_tracks(records))
    assert len(tracks) == 1
    assert [r.timestamp_us for r in tracks[0].records] == [
        (t0 + dt) * 1_000_000 for dt in (0, 10, 20)
    ]


def test_empty_input_yields_nothing() -> None:
    assert list(stitch_tracks([])) == []


def test_single_record_yields_single_track() -> None:
    tracks = list(stitch_tracks([_rec("a3c8b7", 100)]))
    assert len(tracks) == 1
    assert len(tracks[0].records) == 1
