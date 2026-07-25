"""Tests for the OpenSky sample CLI and the checked-in CI dry-run sample.

The CLI's fetch path is exercised against ``httpx.MockTransport`` (no
live OpenSky calls, matching ``test_opensky.py``). The checked-in
``test-data/trajectories/opensky-sample.parquet`` is validated
end-to-end through :func:`acquisition.opensky.load_zenodo_dump` — this
is the CI dry-run for the acquisition schema against real OpenSky-derived
data (flight-data-training-pipeline task 2.7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from acquisition.opensky import OPENSKY_BASE_URL, BoundingBox, load_zenodo_dump
from acquisition.opensky_cli import capture, dedup_sort, main, parse_args, resolve_window
from acquisition.schema import TrajectoryRecord

SAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "test-data" / "trajectories" / "opensky-sample.parquet"
)
SAMPLE_BUDGET_BYTES = 5 * 1024 * 1024

FIXTURE_TIME = 1_700_000_000
FIXTURE_STATE_ROW: list[object] = [
    "a3c8b7",
    "UAL123  ",
    "United States",
    FIXTURE_TIME,
    FIXTURE_TIME,
    -122.31,
    47.45,
    3000.0,
    False,
    120.0,
    45.0,
    1.5,
    None,
    3050.0,
    "2345",
    False,
    0,
    4,
]


def _record(icao24: str, timestamp_us: int) -> TrajectoryRecord:
    return TrajectoryRecord(
        icao24=icao24,
        timestamp_us=timestamp_us,
        lat=47.45,
        lon=-122.31,
        source="opensky",
    )


class TestWindowResolution:
    def test_explicit_time_window(self) -> None:
        args = parse_args(["--bbox", "0", "1", "0", "1", "--time", "100", "200"])
        window = resolve_window(args, now_s=999)
        assert (window.start_s, window.end_s) == (100, 200)

    def test_duration_window_anchors_at_now(self) -> None:
        args = parse_args(["--bbox", "0", "1", "0", "1", "--duration-s", "60"])
        window = resolve_window(args, now_s=500)
        assert (window.start_s, window.end_s) == (500, 560)


class TestDedupSort:
    def test_drops_duplicate_key_keeps_first(self) -> None:
        a1 = _record("a3c8b7", 1_000_000)
        a1_dup = _record("a3c8b7", 1_000_000)
        b = _record("0000aa", 2_000_000)
        a2 = _record("a3c8b7", 3_000_000)
        result = dedup_sort([a2, a1, a1_dup, b])
        assert [(r.icao24, r.timestamp_us) for r in result] == [
            ("0000aa", 2_000_000),
            ("a3c8b7", 1_000_000),
            ("a3c8b7", 3_000_000),
        ]


class TestCaptureAgainstMockTransport:
    BBOX = BoundingBox(lat_min=47.0, lat_max=48.0, lon_min=-123.0, lon_max=-122.0)

    def _mock_client(self, params_seen: list[dict[str, str]]) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            params_seen.append(dict(request.url.params))
            payload: dict[str, Any] = {"time": FIXTURE_TIME, "states": [FIXTURE_STATE_ROW]}
            return httpx.Response(200, content=json.dumps(payload))

        return httpx.Client(base_url=OPENSKY_BASE_URL, transport=httpx.MockTransport(handler))

    def test_polls_current_snapshots_and_dedups(self) -> None:
        params_seen: list[dict[str, str]] = []
        raw = capture(
            self.BBOX,
            3,
            poll_interval_s=1.0,
            auth=None,
            client=self._mock_client(params_seen),
        )
        assert len(params_seen) == 3
        for params in params_seen:
            # Live anonymous polls must never send `time` (OpenSky 403s on
            # stale values: "Authenticate to get historical data")...
            assert "time" not in params
            # ...and must keep extended=1 so aircraft category is included.
            assert params["extended"] == "1"
        assert len(raw) == 3  # same aircraft repeated per snapshot...
        assert len(dedup_sort(raw)) == 1  # ...collapses on (icao24, timestamp_us)

    def test_mid_run_error_keeps_partial_records(self) -> None:
        polls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal polls
            polls += 1
            if polls > 2:
                return httpx.Response(403, content="Authenticate to get historical data")
            payload: dict[str, Any] = {"time": FIXTURE_TIME, "states": [FIXTURE_STATE_ROW]}
            return httpx.Response(200, content=json.dumps(payload))

        client = httpx.Client(base_url=OPENSKY_BASE_URL, transport=httpx.MockTransport(handler))
        raw = capture(
            self.BBOX,
            5,
            poll_interval_s=1.0,
            auth=None,
            client=client,
            retries=0,
            backoff_s=0.0,
        )
        assert len(raw) == 2, "records before the failure must survive"
        assert polls == 3, "capture stops polling after the fatal error"


class TestCredentialResolution:
    """OAuth2 credential resolution happens before any network call."""

    def test_missing_credentials_file_exits_before_any_fetch(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="not found"):
            main(
                [
                    "--bbox", "0", "1", "0", "1",
                    "--credentials-file", str(tmp_path / "absent.json"),
                ]
            )

    def test_historical_window_requires_authentication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--time is authenticated-only; anonymous callers must be told, not silently
        handed the current snapshot."""
        monkeypatch.delenv("OPENSKY_CLIENT_ID", raising=False)
        monkeypatch.delenv("OPENSKY_CLIENT_SECRET", raising=False)
        monkeypatch.setattr(
            "acquisition.opensky_cli.load_credentials", lambda **_kwargs: None
        )
        with pytest.raises(SystemExit, match="requires authentication"):
            main(["--bbox", "0", "1", "0", "1", "--time", "100", "200"])

    def test_secrets_are_not_accepted_as_flags(self) -> None:
        """argv is world-readable via `ps`, so there must be no secret-bearing flag."""
        with pytest.raises(SystemExit):
            parse_args(["--bbox", "0", "1", "0", "1", "--credentials", "user:pass"])


class TestCheckedInSample:
    """CI dry-run over the real OpenSky-derived sample (task 2.7)."""

    def test_sample_exists_within_budget(self) -> None:
        assert SAMPLE_PATH.is_file(), f"missing checked-in sample: {SAMPLE_PATH}"
        assert SAMPLE_PATH.stat().st_size < SAMPLE_BUDGET_BYTES

    def test_sample_loads_as_canonical_records(self) -> None:
        records = list(load_zenodo_dump(SAMPLE_PATH))
        assert len(records) >= 50, "sample should contain a meaningful capture"
        assert all(r.source == "opensky" for r in records)

    def test_sample_is_deduplicated_and_sorted(self) -> None:
        records = list(load_zenodo_dump(SAMPLE_PATH))
        keys = [(r.icao24, r.timestamp_us) for r in records]
        assert len(set(keys)) == len(keys), "duplicate (icao24, timestamp_us) rows"
        assert keys == sorted(keys), "rows must be sorted by (icao24, timestamp_us)"

    def test_sample_contains_trajectories_not_just_snapshots(self) -> None:
        records = list(load_zenodo_dump(SAMPLE_PATH))
        per_aircraft: dict[str, int] = {}
        for r in records:
            per_aircraft[r.icao24] = per_aircraft.get(r.icao24, 0) + 1
        assert len(per_aircraft) >= 2, "expected multiple aircraft"
        assert max(per_aircraft.values()) >= 2, "expected a time series for at least one aircraft"
