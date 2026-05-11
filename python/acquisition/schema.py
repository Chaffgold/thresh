"""Canonical trajectory schema.

The acquisition layer translates OpenSky and ADS-B Exchange records
into a single schema. The Pydantic model is the validation surface;
the PyArrow schema is the on-disk Parquet contract. The two are kept
in lockstep — every field in :class:`TrajectoryRecord` has a matching
column in :data:`TRAJECTORY_PARQUET_SCHEMA`.

See ``openspec/changes/flight-data-training-pipeline/specs/flight-data-acquisition/spec.md``
for the authoritative requirements. ADS-B reports are **system-level
truth** and are never consumed as raw sensor measurements.
"""

from __future__ import annotations

from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

Source = Literal["opensky", "adsbx"]


class TrajectoryRecord(BaseModel):
    """One observation of one aircraft at one instant in time.

    All units are SI (m, m/s) unless otherwise noted; timestamps are
    UTC microseconds since the UNIX epoch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    icao24: str = Field(min_length=6, max_length=6, pattern=r"^[0-9a-f]{6}$")
    """Lowercase hex 24-bit ICAO Mode-S address (e.g. ``a3c8b7``)."""

    timestamp_us: int = Field(ge=0)
    """UTC microseconds since the UNIX epoch."""

    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)

    alt_geom_m: float | None = None
    """Geometric (GNSS) altitude in metres above WGS84 ellipsoid."""

    alt_baro_m: float | None = None
    """Barometric altitude in metres above MSL."""

    vel_ground_mps: float | None = None
    """Ground speed in m/s."""

    track_deg: float | None = Field(default=None, ge=0.0, le=360.0)
    """True track (course) in degrees, 0 to 360."""

    vrate_mps: float | None = None
    """Vertical rate in m/s (positive = climb)."""

    category: str | None = None
    """ADS-B emitter category code (e.g. ``A1``, ``A4``, ``B6``)."""

    callsign: str | None = None

    quality_nic: int | None = Field(default=None, ge=0, le=11)
    """Navigation Integrity Category (0 to 11)."""

    quality_nac_p: int | None = Field(default=None, ge=0, le=11)
    """Navigation Accuracy Category for position (0 to 11)."""

    source: Source
    """Which provider this record came from."""

    provenance: dict[str, bool] | None = None
    """Optional per-field MLAT/TIS-B provenance (ADSBx only)."""


TRAJECTORY_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("icao24", pa.string(), nullable=False),
        pa.field("timestamp_us", pa.int64(), nullable=False),
        pa.field("lat", pa.float64(), nullable=False),
        pa.field("lon", pa.float64(), nullable=False),
        pa.field("alt_geom_m", pa.float64(), nullable=True),
        pa.field("alt_baro_m", pa.float64(), nullable=True),
        pa.field("vel_ground_mps", pa.float64(), nullable=True),
        pa.field("track_deg", pa.float64(), nullable=True),
        pa.field("vrate_mps", pa.float64(), nullable=True),
        pa.field("category", pa.string(), nullable=True),
        pa.field("callsign", pa.string(), nullable=True),
        pa.field("quality_nic", pa.int8(), nullable=True),
        pa.field("quality_nac_p", pa.int8(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
        pa.field("provenance", pa.map_(pa.string(), pa.bool_()), nullable=True),
    ]
)
"""Canonical PyArrow schema for trajectory Parquet files."""


_CATEGORY_MAP: dict[str, str] = {
    "A1": "light-fixed-wing",
    "A2": "light-fixed-wing",
    "A3": "heavy-fixed-wing",
    "A4": "heavy-fixed-wing",
    "A5": "heavy-fixed-wing",
    "A7": "rotorcraft",
    "B1": "glider-or-balloon-or-uav",
    "B2": "glider-or-balloon-or-uav",
    "B6": "glider-or-balloon-or-uav",
}

THRESH_CLASSES: tuple[str, ...] = (
    "light-fixed-wing",
    "heavy-fixed-wing",
    "rotorcraft",
    "glider-or-balloon-or-uav",
    "other",
)
"""The five thresh-internal detection classes."""


def map_category(category: str | None) -> str:
    """Map an ADS-B emitter category to the thresh class enum.

    Unknown or null categories map to ``"other"`` (never raises). See
    design.md Decision 8.
    """
    if category is None:
        return "other"
    return _CATEGORY_MAP.get(category.upper(), "other")
