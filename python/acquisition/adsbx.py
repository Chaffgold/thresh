"""ADS-B Exchange v2 ingestion.

Polls the gateway's geospatial-filtering endpoints (currently only
``/api/aircraft/v2/airport/{icao}`` is implemented) and translates
readsb-style aircraft records into the canonical trajectory schema.

Records are system-level truth (design.md Decision 4) and are never
consumed downstream as raw sensor measurements. ADSBx data is not
redistributed under their TOS — this module is shipped as the
acquisition recipe; the developer supplies their own API key.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from acquisition.schema import TrajectoryRecord

ADSBX_BASE_URL = "https://gateway.adsbexchange.com/api/aircraft/v2"
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_S = 1.0

# Unit conversions
FEET_TO_METRES = 0.3048
KNOTS_TO_MPS = 0.5144444
FEET_PER_MIN_TO_MPS = 0.00508


class AdsbxError(RuntimeError):
    """Non-recoverable error talking to the ADSBx gateway."""


class AdsbxRateLimited(AdsbxError):
    """Surfaced after repeated 429 responses (likely a misconfigured rate limit)."""


@dataclass(frozen=True, slots=True)
class AdsbxResponse:
    """One snapshot of aircraft visible at a poll time."""

    snapshot_time_us: int
    """Server-reported ``now`` (UTC microseconds since UNIX epoch)."""

    records: list[TrajectoryRecord]


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any, lo: int, hi: int) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if lo <= value <= hi:
            return value
        return None
    return None


def _build_provenance(record: dict[str, Any]) -> dict[str, bool] | None:
    """Mark which fields came from MLAT or TIS-B per the source record.

    Returns ``None`` when neither annotation list is present, otherwise
    a dict whose keys are the canonical-schema field names with
    ``True`` indicating "this field originated from MLAT or TIS-B".
    """
    mlat = record.get("mlat")
    tisb = record.get("tisb")
    if not mlat and not tisb:
        return None

    field_map = {
        "lat": "lat",
        "lon": "lon",
        "alt_baro": "alt_baro_m",
        "alt_geom": "alt_geom_m",
        "gs": "vel_ground_mps",
        "track": "track_deg",
        "baro_rate": "vrate_mps",
        "geom_rate": "vrate_mps",
    }
    provenance: dict[str, bool] = {}
    for source_field, canonical_field in field_map.items():
        if (mlat and source_field in mlat) or (tisb and source_field in tisb):
            provenance[canonical_field] = True
    return provenance or None


def _record_timestamp_us(record: dict[str, Any], snapshot_time_us: int) -> int:
    """Compute UTC microseconds for this record.

    ADSBx reports ``seen_pos`` as seconds since the position update.
    If absent we fall back to the snapshot time.
    """
    seen_pos = _coerce_float(record.get("seen_pos"))
    if seen_pos is None:
        return snapshot_time_us
    return snapshot_time_us - int(seen_pos * 1_000_000)


def aircraft_record_to_canonical(
    record: dict[str, Any],
    snapshot_time_us: int,
) -> TrajectoryRecord | None:
    """Translate one ADSBx aircraft record to the canonical schema.

    Returns ``None`` for records lacking position data (the feed
    includes Mode-S contacts with no ADS-B position).
    """
    icao24 = record.get("hex")
    lat = _coerce_float(record.get("lat"))
    lon = _coerce_float(record.get("lon"))
    if not isinstance(icao24, str) or lat is None or lon is None:
        return None

    callsign_raw = record.get("flight")
    callsign = callsign_raw.strip() if isinstance(callsign_raw, str) else None

    alt_baro_ft = _coerce_float(record.get("alt_baro"))
    alt_geom_ft = _coerce_float(record.get("alt_geom"))
    gs_knots = _coerce_float(record.get("gs"))
    track_deg = _coerce_float(record.get("track"))
    # geom_rate (GNSS-derived) preferred; fall back to baro_rate.
    vrate_fpm = _coerce_float(record.get("geom_rate"))
    if vrate_fpm is None:
        vrate_fpm = _coerce_float(record.get("baro_rate"))

    return TrajectoryRecord(
        icao24=icao24.lower().strip(),
        timestamp_us=_record_timestamp_us(record, snapshot_time_us),
        lat=lat,
        lon=lon,
        alt_geom_m=(alt_geom_ft * FEET_TO_METRES) if alt_geom_ft is not None else None,
        alt_baro_m=(alt_baro_ft * FEET_TO_METRES) if alt_baro_ft is not None else None,
        vel_ground_mps=(gs_knots * KNOTS_TO_MPS) if gs_knots is not None else None,
        track_deg=track_deg if track_deg is not None and 0.0 <= track_deg <= 360.0 else None,
        vrate_mps=(vrate_fpm * FEET_PER_MIN_TO_MPS) if vrate_fpm is not None else None,
        category=record.get("category") if isinstance(record.get("category"), str) else None,
        callsign=callsign,
        quality_nic=_coerce_int(record.get("nic"), 0, 11),
        quality_nac_p=_coerce_int(record.get("nac_p"), 0, 11),
        source="adsbx",
        provenance=_build_provenance(record),
    )


def fetch_airport(
    airport_icao: str,
    api_key: str,
    *,
    client: httpx.Client | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
) -> AdsbxResponse:
    """Fetch one snapshot of aircraft visible at ``airport_icao``.

    Hits ``/api/aircraft/v2/airport/{icao}``. Translates the
    readsb-style payload into canonical records. Transient HTTP
    failures retry with exponential backoff. 429 responses honour
    ``Retry-After`` once; persistent rate-limiting raises
    :class:`AdsbxRateLimited`.
    """
    owns_client = client is None
    http = client or httpx.Client(
        base_url=ADSBX_BASE_URL,
        timeout=30.0,
        headers={"x-rapidapi-key": api_key},
    )
    try:
        payload = _fetch_with_retry(
            http,
            path=f"/airport/{airport_icao}",
            retries=retries,
            backoff_s=backoff_s,
        )
        snapshot_time_us = int(payload.get("now", time.time() * 1000)) * 1000
        aircraft = payload.get("ac") or []
        canonical: list[TrajectoryRecord] = []
        for record in aircraft:
            rec = aircraft_record_to_canonical(record, snapshot_time_us)
            if rec is not None:
                canonical.append(rec)
        return AdsbxResponse(snapshot_time_us=snapshot_time_us, records=canonical)
    finally:
        if owns_client:
            http.close()


def _fetch_with_retry(
    http: httpx.Client,
    *,
    path: str,
    retries: int,
    backoff_s: float,
) -> dict[str, Any]:
    consecutive_429 = 0
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = http.get(path)
            if resp.status_code == 429:
                consecutive_429 += 1
                if consecutive_429 >= 3:
                    raise AdsbxRateLimited(
                        "three consecutive 429 responses from ADSBx; "
                        "check the configured rate limit"
                    )
                retry_after = float(resp.headers.get("Retry-After", "1"))
                time.sleep(retry_after)
                continue
            consecutive_429 = 0
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"transient {resp.status_code}", request=resp.request, response=resp
                )
            if resp.status_code >= 400:
                raise AdsbxError(f"HTTP {resp.status_code} from ADSBx: {resp.text[:500]}")
            payload: Any = resp.json()
            if not isinstance(payload, dict):
                raise AdsbxError(f"unexpected ADSBx payload type: {type(payload).__name__}")
            return payload
        except (httpx.HTTPError, AdsbxError) as exc:
            if isinstance(exc, AdsbxRateLimited):
                raise
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_s * (2**attempt))
            else:
                raise AdsbxError(
                    f"ADSBx fetch failed after {retries + 1} attempts: {exc}"
                ) from exc
    raise AdsbxError(f"unreachable: {last_error}")


def fetch_airport_iter(
    airport_icao: str,
    api_key: str,
    *,
    client: httpx.Client | None = None,
) -> Iterator[TrajectoryRecord]:
    """Convenience iterator wrapper around :func:`fetch_airport`."""
    response = fetch_airport(airport_icao, api_key, client=client)
    yield from response.records
