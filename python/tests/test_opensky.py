"""Tests for the OpenSky REST client and Zenodo loader.

No live OpenSky calls — the REST client is exercised against a mock
transport built on ``httpx.MockTransport``. Zenodo loading is tested
against a temp Parquet file written from the canonical schema.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from acquisition.opensky import (
    BoundingBox,
    OpenSkyError,
    TimeRange,
    fetch_state_vectors,
    load_zenodo_dump,
    state_row_to_record,
)
from acquisition.schema import TrajectoryRecord
from acquisition.storage import write_partition

FIXTURE_TIME = 1_700_000_000
FIXTURE_STATE_ROW: list[object] = [
    "a3c8b7",          # icao24
    "UAL123  ",        # callsign (often padded)
    "United States",   # origin_country
    FIXTURE_TIME,      # time_position
    FIXTURE_TIME,      # last_contact
    -122.31,           # longitude
    47.45,             # latitude
    3000.0,            # baro_altitude
    False,             # on_ground
    120.0,             # velocity
    45.0,              # true_track
    1.5,               # vertical_rate
    None,              # sensors
    3050.0,            # geo_altitude
    "2345",            # squawk
    False,             # spi
    0,                 # position_source
    4,                 # category (numeric → "A4")
]


class TestStateRowTranslation:
    def test_translates_full_row(self) -> None:
        rec = state_row_to_record(FIXTURE_STATE_ROW, FIXTURE_TIME)
        assert rec is not None
        assert rec.icao24 == "a3c8b7"
        assert rec.callsign == "UAL123"
        assert rec.source == "opensky"
        assert rec.lat == 47.45
        assert rec.lon == -122.31
        assert rec.alt_baro_m == 3000.0
        assert rec.alt_geom_m == 3050.0
        assert rec.vel_ground_mps == 120.0
        assert rec.track_deg == 45.0
        assert rec.vrate_mps == 1.5
        assert rec.category == "A4"
        assert rec.timestamp_us == FIXTURE_TIME * 1_000_000

    def test_skips_rows_without_position(self) -> None:
        row = list(FIXTURE_STATE_ROW)
        row[5] = None  # longitude
        row[6] = None  # latitude
        assert state_row_to_record(row, FIXTURE_TIME) is None

    def test_skips_rows_without_icao24(self) -> None:
        row = list(FIXTURE_STATE_ROW)
        row[0] = None
        assert state_row_to_record(row, FIXTURE_TIME) is None

    def test_uppercase_icao_normalised(self) -> None:
        row = list(FIXTURE_STATE_ROW)
        row[0] = "A3C8B7"
        rec = state_row_to_record(row, FIXTURE_TIME)
        assert rec is not None and rec.icao24 == "a3c8b7"

    def test_string_category_passes_through(self) -> None:
        row = list(FIXTURE_STATE_ROW)
        row[17] = "B6"
        rec = state_row_to_record(row, FIXTURE_TIME)
        assert rec is not None and rec.category == "B6"

    def test_zero_category_treated_as_none(self) -> None:
        row = list(FIXTURE_STATE_ROW)
        row[17] = 0
        rec = state_row_to_record(row, FIXTURE_TIME)
        assert rec is not None and rec.category is None

    def test_missing_time_position_uses_fetch_time(self) -> None:
        row = list(FIXTURE_STATE_ROW)
        row[3] = None
        rec = state_row_to_record(row, FIXTURE_TIME)
        assert rec is not None and rec.timestamp_us == FIXTURE_TIME * 1_000_000


def _mock_transport(payloads: Sequence[Mapping[str, Any]]) -> httpx.MockTransport:
    """Return a transport that serves ``payloads`` in order, then 404s."""
    iterator: Iterator[Mapping[str, Any]] = iter(payloads)

    def handler(_request: httpx.Request) -> httpx.Response:
        try:
            payload = next(iterator)
        except StopIteration:
            return httpx.Response(404, text="no more payloads")
        return httpx.Response(200, text=json.dumps(payload))

    return httpx.MockTransport(handler)


class TestFetchStateVectors:
    def test_polls_each_step_in_time_range(self) -> None:
        payloads = [
            {"time": FIXTURE_TIME, "states": [FIXTURE_STATE_ROW]},
            {"time": FIXTURE_TIME + 10, "states": [FIXTURE_STATE_ROW]},
        ]
        bbox = BoundingBox(lat_min=40.0, lat_max=50.0, lon_min=-130.0, lon_max=-110.0)
        rng = TimeRange(start_s=FIXTURE_TIME, end_s=FIXTURE_TIME + 20)
        with httpx.Client(
            base_url="https://opensky-network.org/api",
            transport=_mock_transport(payloads),
        ) as client:
            records = list(
                fetch_state_vectors(bbox, rng, poll_interval_s=10.0, client=client)
            )
        assert len(records) == 2

    def test_empty_states_array_yields_no_records(self) -> None:
        payloads = [{"time": FIXTURE_TIME, "states": []}]
        bbox = BoundingBox(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
        rng = TimeRange(start_s=FIXTURE_TIME, end_s=FIXTURE_TIME + 1)
        with httpx.Client(
            base_url="https://opensky-network.org/api",
            transport=_mock_transport(payloads),
        ) as client:
            assert list(fetch_state_vectors(bbox, rng, client=client)) == []

    def test_400_response_raises_opensky_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="bad request")

        bbox = BoundingBox(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
        rng = TimeRange(start_s=FIXTURE_TIME, end_s=FIXTURE_TIME + 1)
        with httpx.Client(
            base_url="https://opensky-network.org/api",
            transport=httpx.MockTransport(handler),
        ) as client, pytest.raises(OpenSkyError, match="HTTP 400"):
            list(fetch_state_vectors(bbox, rng, client=client, retries=0))


class TestBoundingBoxValidation:
    def test_rejects_inverted_lat_range(self) -> None:
        with pytest.raises(ValueError, match="latitude"):
            BoundingBox(lat_min=50.0, lat_max=40.0, lon_min=-130.0, lon_max=-110.0)

    def test_rejects_out_of_range_lon(self) -> None:
        with pytest.raises(ValueError, match="longitude"):
            BoundingBox(lat_min=40.0, lat_max=50.0, lon_min=-200.0, lon_max=0.0)


class TestTimeRangeValidation:
    def test_rejects_empty_range(self) -> None:
        with pytest.raises(ValueError, match="empty time range"):
            TimeRange(start_s=100, end_s=100)


class TestZenodoLoader:
    def _write_fixture(self, tmp_path: Path) -> Path:
        """Use the canonical storage writer to produce a Zenodo-like parquet."""
        records = [
            TrajectoryRecord(
                icao24="a3c8b7",
                timestamp_us=(FIXTURE_TIME + i) * 1_000_000,
                lat=47.45 + i * 0.01,
                lon=-122.31 + i * 0.01,
                alt_geom_m=3050.0,
                alt_baro_m=3000.0,
                vel_ground_mps=120.0,
                track_deg=45.0,
                vrate_mps=1.5,
                category="A4",
                callsign="UAL123",
                quality_nic=8,
                quality_nac_p=10,
                source="opensky",
            )
            for i in range(2)
        ]
        written = write_partition(records, root=tmp_path, source="opensky")
        return written[0]

    def test_loads_records_from_parquet(self, tmp_path: Path) -> None:
        path = self._write_fixture(tmp_path)
        records = list(load_zenodo_dump(path))
        assert len(records) == 2
        assert records[0].icao24 == "a3c8b7"
        assert records[0].callsign == "UAL123"

    def test_correct_sha256_passes(self, tmp_path: Path) -> None:
        path = self._write_fixture(tmp_path)
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        records = list(load_zenodo_dump(path, sha256=expected))
        assert len(records) == 2

    def test_wrong_sha256_raises(self, tmp_path: Path) -> None:
        path = self._write_fixture(tmp_path)
        with pytest.raises(OpenSkyError, match="checksum mismatch"):
            list(load_zenodo_dump(path, sha256="0" * 64))
