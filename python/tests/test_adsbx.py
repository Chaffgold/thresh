"""Tests for the ADS-B Exchange v2 client and poller."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from acquisition.adsbx import (
    AdsbxError,
    AdsbxRateLimited,
    aircraft_record_to_canonical,
    fetch_airport,
)
from acquisition.adsbx_poller import coalesce_records
from acquisition.schema import TrajectoryRecord

SNAPSHOT_TIME_S = 1_700_000_000
SNAPSHOT_TIME_US = SNAPSHOT_TIME_S * 1_000_000


def _aircraft_record(**overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "hex": "a3c8b7",
        "type": "adsb_icao",
        "flight": "UAL123  ",
        "r": "N12345",
        "t": "B738",
        "lat": 47.45,
        "lon": -122.31,
        "alt_baro": 9842,  # ~3000 m in feet
        "alt_geom": 10000,  # ~3048 m
        "gs": 233,  # ~120 m/s in knots
        "track": 45.0,
        "baro_rate": 295,  # ~1.5 m/s in feet/min
        "geom_rate": 295,
        "category": "A4",
        "nic": 8,
        "nac_p": 10,
        "seen_pos": 0.5,
    }
    base.update(overrides)
    return base


class TestRecordTranslation:
    def test_translates_full_record(self) -> None:
        rec = aircraft_record_to_canonical(_aircraft_record(), SNAPSHOT_TIME_US)
        assert rec is not None
        assert rec.icao24 == "a3c8b7"
        assert rec.callsign == "UAL123"
        assert rec.source == "adsbx"
        assert rec.lat == pytest.approx(47.45)
        assert rec.lon == pytest.approx(-122.31)
        assert rec.alt_baro_m == pytest.approx(2999.84, abs=0.5)
        assert rec.alt_geom_m == pytest.approx(3048.0, abs=0.5)
        assert rec.vel_ground_mps == pytest.approx(120.0, abs=0.5)
        assert rec.track_deg == 45.0
        assert rec.vrate_mps == pytest.approx(1.5, abs=0.05)
        assert rec.category == "A4"
        assert rec.quality_nic == 8
        assert rec.quality_nac_p == 10

    def test_skips_record_without_position(self) -> None:
        for missing in ("lat", "lon"):
            record = _aircraft_record()
            record[missing] = None
            assert aircraft_record_to_canonical(record, SNAPSHOT_TIME_US) is None

    def test_skips_record_without_hex(self) -> None:
        record = _aircraft_record()
        record["hex"] = None
        assert aircraft_record_to_canonical(record, SNAPSHOT_TIME_US) is None

    def test_uppercase_hex_normalised(self) -> None:
        rec = aircraft_record_to_canonical(
            _aircraft_record(hex="A3C8B7"), SNAPSHOT_TIME_US
        )
        assert rec is not None and rec.icao24 == "a3c8b7"

    def test_seen_pos_offsets_timestamp(self) -> None:
        rec = aircraft_record_to_canonical(
            _aircraft_record(seen_pos=2.5), SNAPSHOT_TIME_US
        )
        assert rec is not None
        assert rec.timestamp_us == SNAPSHOT_TIME_US - 2_500_000

    def test_missing_seen_pos_uses_snapshot_time(self) -> None:
        record = _aircraft_record()
        del record["seen_pos"]
        rec = aircraft_record_to_canonical(record, SNAPSHOT_TIME_US)
        assert rec is not None and rec.timestamp_us == SNAPSHOT_TIME_US

    def test_invalid_track_dropped(self) -> None:
        rec = aircraft_record_to_canonical(
            _aircraft_record(track=500.0), SNAPSHOT_TIME_US
        )
        assert rec is not None and rec.track_deg is None

    def test_geom_rate_preferred_over_baro_rate(self) -> None:
        rec = aircraft_record_to_canonical(
            _aircraft_record(geom_rate=590, baro_rate=10), SNAPSHOT_TIME_US
        )
        # geom_rate=590 fpm → ~3 m/s
        assert rec is not None
        assert rec.vrate_mps == pytest.approx(3.0, abs=0.1)


class TestProvenance:
    def test_no_provenance_when_no_mlat_or_tisb(self) -> None:
        rec = aircraft_record_to_canonical(_aircraft_record(), SNAPSHOT_TIME_US)
        assert rec is not None and rec.provenance is None

    def test_mlat_marks_canonical_field_names(self) -> None:
        rec = aircraft_record_to_canonical(
            _aircraft_record(mlat=["lat", "lon", "alt_baro"]),
            SNAPSHOT_TIME_US,
        )
        assert rec is not None and rec.provenance == {
            "lat": True,
            "lon": True,
            "alt_baro_m": True,
        }

    def test_tisb_also_recorded(self) -> None:
        rec = aircraft_record_to_canonical(
            _aircraft_record(tisb=["gs", "track"]),
            SNAPSHOT_TIME_US,
        )
        assert rec is not None and rec.provenance == {
            "vel_ground_mps": True,
            "track_deg": True,
        }


def _mock_transport(payloads: Sequence[Mapping[str, Any]]) -> httpx.MockTransport:
    iterator = iter(payloads)

    def handler(_request: httpx.Request) -> httpx.Response:
        try:
            payload = next(iterator)
        except StopIteration:
            return httpx.Response(404, text="exhausted")
        return httpx.Response(200, text=json.dumps(payload))

    return httpx.MockTransport(handler)


class TestFetchAirport:
    def test_translates_payload(self) -> None:
        payload = {
            "now": SNAPSHOT_TIME_S * 1000,
            "ac": [_aircraft_record(), _aircraft_record(hex="b1d2e3")],
        }
        with httpx.Client(
            base_url="https://gateway.adsbexchange.com/api/aircraft/v2",
            transport=_mock_transport([payload]),
            headers={"x-rapidapi-key": "test"},
        ) as client:
            response = fetch_airport("KSEA", "test", client=client)
        assert response.snapshot_time_us == SNAPSHOT_TIME_US
        assert len(response.records) == 2
        assert {r.icao24 for r in response.records} == {"a3c8b7", "b1d2e3"}

    def test_empty_aircraft_list_yields_empty_response(self) -> None:
        payload = {"now": SNAPSHOT_TIME_S * 1000, "ac": []}
        with httpx.Client(
            base_url="https://gateway.adsbexchange.com/api/aircraft/v2",
            transport=_mock_transport([payload]),
            headers={"x-rapidapi-key": "test"},
        ) as client:
            response = fetch_airport("KSEA", "test", client=client)
        assert response.records == []

    def test_400_raises_adsbx_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="bad airport")

        with httpx.Client(
            base_url="https://gateway.adsbexchange.com/api/aircraft/v2",
            transport=httpx.MockTransport(handler),
            headers={"x-rapidapi-key": "test"},
        ) as client, pytest.raises(AdsbxError, match="HTTP 400"):
            fetch_airport("KSEA", "test", client=client, retries=0)


class TestRateLimit:
    def test_three_consecutive_429s_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="slow down", headers={"Retry-After": "0"})

        with httpx.Client(
            base_url="https://gateway.adsbexchange.com/api/aircraft/v2",
            transport=httpx.MockTransport(handler),
            headers={"x-rapidapi-key": "test"},
        ) as client, pytest.raises(AdsbxRateLimited, match="three consecutive 429"):
            fetch_airport("KSEA", "test", client=client, retries=10, backoff_s=0.0)

    def test_single_429_followed_by_success_recovers(self) -> None:
        responses = iter(
            [
                httpx.Response(429, text="slow", headers={"Retry-After": "0"}),
                httpx.Response(
                    200,
                    text=json.dumps(
                        {"now": SNAPSHOT_TIME_S * 1000, "ac": [_aircraft_record()]}
                    ),
                ),
            ]
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return next(responses)

        with httpx.Client(
            base_url="https://gateway.adsbexchange.com/api/aircraft/v2",
            transport=httpx.MockTransport(handler),
            headers={"x-rapidapi-key": "test"},
        ) as client:
            response = fetch_airport("KSEA", "test", client=client, retries=2, backoff_s=0.0)
        assert len(response.records) == 1


class TestCoalesce:
    def _rec(self, icao: str, ts_us: int) -> TrajectoryRecord:
        return TrajectoryRecord(
            icao24=icao,
            timestamp_us=ts_us,
            lat=0.0,
            lon=0.0,
            source="adsbx",
        )

    def test_dedup_by_icao_and_timestamp(self) -> None:
        snapshot1 = [self._rec("a3c8b7", 100), self._rec("b1d2e3", 100)]
        snapshot2 = [self._rec("a3c8b7", 100), self._rec("a3c8b7", 200)]
        unique, dups = coalesce_records([snapshot1, snapshot2])
        assert dups == 1
        assert len(unique) == 3
        keys = {(r.icao24, r.timestamp_us) for r in unique}
        assert keys == {("a3c8b7", 100), ("b1d2e3", 100), ("a3c8b7", 200)}

    def test_empty_input(self, tmp_path: Path) -> None:
        del tmp_path  # unused
        unique, dups = coalesce_records([])
        assert unique == [] and dups == 0
