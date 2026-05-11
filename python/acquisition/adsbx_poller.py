"""ADS-B Exchange polling scheduler.

Continuously fetches snapshots from a single airport over a duration
at a configurable rate, deduplicates on ``(icao24, timestamp_us)``,
and appends to per-airport partitioned Parquet files.

Designed for offline collection by a developer with an ADSBx API
key — never invoked by CI (no shared API key).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from acquisition.adsbx import AdsbxRateLimited, fetch_airport
from acquisition.schema import TrajectoryRecord
from acquisition.storage import write_partition

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PollerStats:
    """Mutable counters kept for the duration of a polling run."""

    snapshots_attempted: int = 0
    snapshots_succeeded: int = 0
    records_appended: int = 0
    records_deduplicated: int = 0
    rate_limit_backoffs: int = 0


@dataclass(slots=True)
class PollResult:
    """Final outcome of a polling run."""

    airport_icao: str
    started_at_us: int
    ended_at_us: int
    files_written: list[Path] = field(default_factory=list)
    stats: PollerStats = field(default_factory=PollerStats)


def _deduplicate(
    new_records: list[TrajectoryRecord],
    seen: set[tuple[str, int]],
    stats: PollerStats,
) -> list[TrajectoryRecord]:
    fresh: list[TrajectoryRecord] = []
    for r in new_records:
        key = (r.icao24, r.timestamp_us)
        if key in seen:
            stats.records_deduplicated += 1
            continue
        seen.add(key)
        fresh.append(r)
    return fresh


def _airport_partition_dir(root: Path, airport_icao: str) -> Path:
    """Per-airport sub-partition under the canonical layout."""
    return root / f"airport={airport_icao.upper()}"


def _sleep_until_next_tick(last_tick_s: float, period_s: float) -> float:
    """Sleep until ``last_tick_s + period_s``; return the new tick time."""
    next_tick = last_tick_s + period_s
    delay = next_tick - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    return time.monotonic()


def _ticks(
    duration_s: float,
    period_s: float,
) -> Iterator[None]:
    """Yield once per tick for up to ``duration_s`` wall-clock seconds."""
    deadline = time.monotonic() + duration_s
    last_tick = time.monotonic() - period_s  # fire immediately first
    while time.monotonic() < deadline:
        last_tick = _sleep_until_next_tick(last_tick, period_s)
        yield


def run(
    airport_icao: str,
    api_key: str,
    root: Path,
    *,
    rate_limit_hz: float = 1.0,
    duration_s: float = 60.0,
    client: httpx.Client | None = None,
) -> PollResult:
    """Poll ``/airport/{icao}`` at ``rate_limit_hz`` for ``duration_s`` seconds.

    Each snapshot's records are deduplicated against everything seen
    so far in this run, then written under
    ``<root>/source=adsbx/airport=<ICAO>/date=YYYY-MM-DD/...``.
    Rate-limit responses (HTTP 429) are surfaced via the underlying
    fetch's :class:`AdsbxRateLimited` after three consecutive
    rejections.
    """
    period_s = 1.0 / rate_limit_hz if rate_limit_hz > 0 else 1.0
    started_us = int(time.time() * 1_000_000)
    stats = PollerStats()
    seen: set[tuple[str, int]] = set()
    pending: list[TrajectoryRecord] = []

    for _ in _ticks(duration_s, period_s):
        stats.snapshots_attempted += 1
        try:
            response = fetch_airport(airport_icao, api_key, client=client)
        except AdsbxRateLimited:
            stats.rate_limit_backoffs += 1
            raise
        except Exception:
            logger.exception("ADSBx fetch failed; continuing")
            continue
        stats.snapshots_succeeded += 1
        fresh = _deduplicate(response.records, seen, stats)
        pending.extend(fresh)

    files_written: list[Path] = []
    if pending:
        airport_root = _airport_partition_dir(root, airport_icao)
        files_written = write_partition(pending, root=airport_root, source="adsbx")
        stats.records_appended = len(pending)

    return PollResult(
        airport_icao=airport_icao,
        started_at_us=started_us,
        ended_at_us=int(time.time() * 1_000_000),
        files_written=files_written,
        stats=stats,
    )


def coalesce_records(
    raw_snapshots: list[list[TrajectoryRecord]],
) -> tuple[list[TrajectoryRecord], int]:
    """Merge multiple snapshots' records into one deduplicated list.

    Returns ``(unique_records, dedup_count)``. Useful as a building
    block for tests and offline analysis pipelines that already have
    snapshots in memory.
    """
    seen: set[tuple[str, int]] = set()
    unique: list[TrajectoryRecord] = []
    duplicates = 0
    for snapshot in raw_snapshots:
        for r in snapshot:
            key = (r.icao24, r.timestamp_us)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            unique.append(r)
    return unique, duplicates


__all__ = [
    "PollResult",
    "PollerStats",
    "coalesce_records",
    "run",
]
