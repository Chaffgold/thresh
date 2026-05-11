"""Tests for the canonical trajectory schema and category mapping."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from acquisition.schema import (
    THRESH_CLASSES,
    TRAJECTORY_PARQUET_SCHEMA,
    TrajectoryRecord,
    map_category,
)


def _minimal_record(**overrides: object) -> TrajectoryRecord:
    base: dict[str, object] = {
        "icao24": "abcdef",
        "timestamp_us": 1_700_000_000_000_000,
        "lat": 47.45,
        "lon": -122.31,
        "source": "opensky",
    }
    base.update(overrides)
    return TrajectoryRecord(**base)  # type: ignore[arg-type]


class TestTrajectoryRecord:
    def test_minimum_fields_validate(self) -> None:
        rec = _minimal_record()
        assert rec.icao24 == "abcdef"
        assert rec.source == "opensky"

    def test_icao24_must_be_lowercase_hex_6(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_record(icao24="ABCDEF")  # uppercase
        with pytest.raises(ValidationError):
            _minimal_record(icao24="abcde")  # too short
        with pytest.raises(ValidationError):
            _minimal_record(icao24="ghijkl")  # non-hex chars

    def test_lat_lon_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_record(lat=91.0)
        with pytest.raises(ValidationError):
            _minimal_record(lon=-181.0)

    def test_track_deg_bounds_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _minimal_record(track_deg=361.0)
        with pytest.raises(ValidationError):
            _minimal_record(track_deg=-1.0)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrajectoryRecord.model_validate(
                {
                    "icao24": "abcdef",
                    "timestamp_us": 0,
                    "lat": 0.0,
                    "lon": 0.0,
                    "source": "opensky",
                    "unknown_field": "x",
                }
            )

    def test_optional_fields_default_none(self) -> None:
        rec = _minimal_record()
        assert rec.alt_geom_m is None
        assert rec.alt_baro_m is None
        assert rec.callsign is None
        assert rec.provenance is None

    def test_provenance_accepts_bool_map(self) -> None:
        rec = _minimal_record(provenance={"lat": True, "lon": False})
        assert rec.provenance == {"lat": True, "lon": False}


class TestParquetSchemaParity:
    def test_pydantic_fields_match_parquet_columns(self) -> None:
        pydantic_fields = set(TrajectoryRecord.model_fields)
        parquet_columns = {f.name for f in TRAJECTORY_PARQUET_SCHEMA}
        assert pydantic_fields == parquet_columns, (
            f"schema drift: pydantic={pydantic_fields}, parquet={parquet_columns}"
        )

    def test_required_columns_non_nullable(self) -> None:
        required = {"icao24", "timestamp_us", "lat", "lon", "source"}
        for field in TRAJECTORY_PARQUET_SCHEMA:
            if field.name in required:
                assert not field.nullable, f"{field.name} must be non-nullable"


class TestCategoryMapping:
    @pytest.mark.parametrize(
        ("category", "expected"),
        [
            ("A1", "light-fixed-wing"),
            ("A4", "heavy-fixed-wing"),
            ("A7", "rotorcraft"),
            ("B1", "glider-or-balloon-or-uav"),
            ("C2", "other"),
        ],
    )
    def test_spec_scenario_mappings(self, category: str, expected: str) -> None:
        assert map_category(category) == expected

    def test_unknown_returns_other(self) -> None:
        assert map_category("ZZ") == "other"
        assert map_category("") == "other"

    def test_null_returns_other_without_raising(self) -> None:
        assert map_category(None) == "other"

    def test_case_insensitive(self) -> None:
        assert map_category("a1") == "light-fixed-wing"

    def test_all_returned_values_are_valid_classes(self) -> None:
        for cat in ["A1", "A2", "A3", "A7", "B1", "C0", None]:
            assert map_category(cat) in THRESH_CLASSES
