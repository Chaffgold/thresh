"""OpenSky Network ingestion.

Two entry points:

* :func:`fetch_state_vectors` polls the public ``/api/states/all`` REST
  endpoint over a developer-supplied time range and bounding box. The
  endpoint returns one snapshot per call; this helper steps through
  the range at ``poll_interval_s``.
* :func:`load_zenodo_dump` reads a previously-downloaded Zenodo
  trajectory file (Parquet or CSV) and yields canonical records,
  validating the file's SHA-256 against the provided checksum.

Both emit :class:`TrajectoryRecord` instances with ``source="opensky"``.
Records are system-level truth and are never consumed as raw sensor
measurements (design.md Decision 4).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import pyarrow.parquet as pq

from acquisition.schema import TrajectoryRecord

OPENSKY_BASE_URL = "https://opensky-network.org/api"
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_S = 1.0
DEFAULT_POLL_INTERVAL_S = 10.0


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """WGS84 bounding box (degrees)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.lat_min <= self.lat_max <= 90.0):
            raise ValueError(f"invalid latitude range: [{self.lat_min}, {self.lat_max}]")
        if not (-180.0 <= self.lon_min <= self.lon_max <= 180.0):
            raise ValueError(f"invalid longitude range: [{self.lon_min}, {self.lon_max}]")

    def as_query_params(self) -> dict[str, str]:
        return {
            "lamin": str(self.lat_min),
            "lamax": str(self.lat_max),
            "lomin": str(self.lon_min),
            "lomax": str(self.lon_max),
        }


@dataclass(frozen=True, slots=True)
class TimeRange:
    """UTC time range in epoch seconds, half-open ``[start, end)``."""

    start_s: int
    end_s: int

    def __post_init__(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(f"empty time range: [{self.start_s}, {self.end_s})")


class OpenSkyError(RuntimeError):
    """Non-recoverable error talking to the OpenSky API."""


# OpenSky /api/states/all response columns, in response order (per the
# OpenSky-Network REST docs: https://openskynetwork.github.io/opensky-api/rest.html).
# Observed: rows have 17 columns without ``extended=1`` and 18, ending in
# ``category``, with it.
_STATE_COLUMNS = (
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source",
    "category",
)

# Column name → row index. :func:`state_row_to_record` reads a state-vector
# row through this, so the column order is written once, in ``_STATE_COLUMNS``.
_COL = {name: index for index, name in enumerate(_STATE_COLUMNS)}


def _emitter_category(value: object) -> str | None:
    """Map OpenSky's integer enumeration, not raw ADS-B category numbers.

    The REST contract reserves both 0 and 1 for missing information, then
    enumerates A1..A7 (2..8), B1..B7 (9..15), and C1..C5 (16..20).
    Already-canonical strings remain supported for imported records.
    """
    if isinstance(value, str):
        return value
    # Do not coerce booleans or fractional values into aircraft categories.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 2 <= value <= 8:
        return f"A{value - 1}"
    if 9 <= value <= 15:
        return f"B{value - 8}"
    if 16 <= value <= 20:
        return f"C{value - 15}"
    return None


def state_row_to_record(row: list[Any], fetch_time_s: int) -> TrajectoryRecord | None:
    """Translate one OpenSky state-vector row to a canonical record.

    Returns ``None`` for rows that lack a position (the OpenSky feed
    can emit aircraft with no lat/lon — they have a Mode-S contact
    but no ADS-B position).
    """
    icao24 = cast("str | None", row[_COL["icao24"]])
    longitude = cast("float | None", row[_COL["longitude"]])
    latitude = cast("float | None", row[_COL["latitude"]])
    if icao24 is None or longitude is None or latitude is None:
        return None

    time_position = cast("int | None", row[_COL["time_position"]])
    timestamp_s = time_position if time_position is not None else fetch_time_s

    callsign_raw = cast("str | None", row[_COL["callsign"]])
    callsign = callsign_raw.strip() if callsign_raw else None

    category_raw: Any = row[_COL["category"]] if len(row) > _COL["category"] else None
    category = _emitter_category(category_raw)

    return TrajectoryRecord(
        icao24=icao24.lower().strip(),
        timestamp_us=timestamp_s * 1_000_000,
        lat=latitude,
        lon=longitude,
        alt_geom_m=cast("float | None", row[_COL["geo_altitude"]]),
        alt_baro_m=cast("float | None", row[_COL["baro_altitude"]]),
        vel_ground_mps=cast("float | None", row[_COL["velocity"]]),
        track_deg=cast("float | None", row[_COL["true_track"]]),
        vrate_mps=cast("float | None", row[_COL["vertical_rate"]]),
        category=category,
        callsign=callsign,
        quality_nic=None,
        quality_nac_p=None,
        source="opensky",
        provenance=None,
    )


def fetch_state_vectors(
    bbox: BoundingBox,
    time_range: TimeRange,
    *,
    auth: httpx.Auth | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    client: httpx.Client | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
) -> Iterator[TrajectoryRecord]:
    """Yield canonical records covering ``bbox`` over ``time_range``.

    The OpenSky public REST endpoint returns one snapshot per call;
    this function polls every ``poll_interval_s`` seconds from
    ``time_range.start_s`` to ``time_range.end_s``. Transient HTTP
    failures are retried up to ``retries`` times with exponential
    backoff. Non-recoverable errors raise :class:`OpenSkyError` with
    the response body included.

    Historical windows require authentication: pass ``auth`` from
    :func:`acquisition.opensky_auth.build_auth`. Anonymous callers are
    limited to the current snapshot (:func:`fetch_current_states`).
    """
    owns_client = client is None
    http = client or httpx.Client(base_url=OPENSKY_BASE_URL, timeout=30.0, auth=auth)
    try:
        for t in range(time_range.start_s, time_range.end_s, max(int(poll_interval_s), 1)):
            payload = _fetch_one_snapshot(http, bbox, t, retries=retries, backoff_s=backoff_s)
            states = payload.get("states") or []
            payload_time = int(payload.get("time", t))
            for row in states:
                rec = state_row_to_record(row, payload_time)
                if rec is not None:
                    yield rec
    finally:
        if owns_client:
            http.close()


def fetch_current_states(
    bbox: BoundingBox,
    *,
    auth: httpx.Auth | None = None,
    client: httpx.Client | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
) -> list[TrajectoryRecord]:
    """Fetch one *current* ``/states/all`` snapshot (no ``time`` parameter).

    This is the only form anonymous access reliably supports: sending
    ``time`` even slightly in the past can be rejected with HTTP 403
    "Authenticate to get historical data" once a slow request makes the
    caller's clock lag. Live pollers (``acquisition.opensky_cli``) MUST
    use this instead of :func:`fetch_state_vectors`.
    """
    owns_client = client is None
    http = client or httpx.Client(base_url=OPENSKY_BASE_URL, timeout=30.0, auth=auth)
    try:
        payload = _fetch_one_snapshot(http, bbox, None, retries=retries, backoff_s=backoff_s)
    finally:
        if owns_client:
            http.close()
    states = payload.get("states") or []
    payload_time = int(payload.get("time", time.time()))
    records = (state_row_to_record(row, payload_time) for row in states)
    return [rec for rec in records if rec is not None]


def _fetch_one_snapshot(
    http: httpx.Client,
    bbox: BoundingBox,
    timestamp_s: int | None,
    *,
    retries: int,
    backoff_s: float,
) -> dict[str, Any]:
    params = bbox.as_query_params()
    if timestamp_s is not None:
        params["time"] = str(timestamp_s)
    # Without extended=1 the API omits column 17 (aircraft category),
    # which feeds map_category / the thresh class labels downstream.
    params["extended"] = "1"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = http.get("/states/all", params=params)
            if resp.status_code >= 500 or resp.status_code == 429:
                raise httpx.HTTPStatusError(
                    f"transient {resp.status_code}", request=resp.request, response=resp
                )
            if resp.status_code >= 400:
                raise OpenSkyError(f"HTTP {resp.status_code} from OpenSky: {resp.text[:500]}")
            return cast("dict[str, Any]", resp.json())
        except (httpx.HTTPError, OpenSkyError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_s * (2**attempt))
            else:
                raise OpenSkyError(
                    f"OpenSky fetch failed after {retries + 1} attempts: {exc}"
                ) from exc
    raise OpenSkyError(f"unreachable: {last_error}")


def load_zenodo_dump(
    path: Path,
    *,
    sha256: str | None = None,
) -> Iterator[TrajectoryRecord]:
    """Read a Zenodo-published OpenSky trajectory dump.

    The Parquet file's SHA-256 is validated against ``sha256`` when
    provided (spec scenario "Loading a Zenodo trajectory dump"). The
    file is expected to use the canonical schema columns; rows
    failing validation raise via ``model_validate``.
    """
    if sha256 is not None:
        _verify_sha256(path, sha256)

    # Use ParquetFile rather than read_table so the Zenodo loader works
    # against files that happen to live under a Hive-partitioned tree
    # (e.g. when re-reading our own outputs); see storage.read_partition.
    table = pq.ParquetFile(path).read()
    for row in table.to_pylist():
        row_typed = cast("dict[str, Any]", row)
        # Ensure source is recorded even when the file omits it.
        row_typed.setdefault("source", "opensky")
        prov_raw = row_typed.get("provenance")
        if prov_raw is not None and not isinstance(prov_raw, dict):
            row_typed["provenance"] = dict(prov_raw)
        yield TrajectoryRecord.model_validate(row_typed)


def _verify_sha256(path: Path, expected_hex: str) -> None:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual != expected_hex:
        raise OpenSkyError(f"checksum mismatch for {path}: expected {expected_hex}, got {actual}")
