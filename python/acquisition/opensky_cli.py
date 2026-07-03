"""Small CLI for capturing an OpenSky-derived trajectory sample.

Live-polls the public ``/api/states/all`` endpoint over a bounding box
and writes a single flat Parquet file in the canonical trajectory
schema (task 2.7: the checked-in CI dry-run sample under
``test-data/trajectories/opensky-sample.parquet``).

Run from the ``python/`` directory::

    uv run python -m acquisition.opensky_cli \\
        --bbox 49.0 51.0 7.0 10.0 --duration-s 300

Anonymous OpenSky access ignores the ``time`` query parameter and
always returns the *current* snapshot, so unlike
:func:`acquisition.opensky.fetch_state_vectors` (which steps through a
historical range for authenticated replay) this CLI paces itself
against the wall clock: one snapshot per ``--poll-interval-s``,
deduplicated on ``(icao24, timestamp_us)``.

The output records attribution obligations documented in
``LICENSING.md`` — "This work uses data from the OpenSky Network".
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from acquisition.opensky import (
    DEFAULT_POLL_INTERVAL_S,
    OPENSKY_BASE_URL,
    BoundingBox,
    TimeRange,
    fetch_state_vectors,
)
from acquisition.schema import TrajectoryRecord
from acquisition.storage import records_to_table

DEFAULT_OUT = (
    Path(__file__).resolve().parents[2] / "test-data" / "trajectories" / "opensky-sample.parquet"
)
"""Anchored to the repo layout, not the caller's CWD."""
SAMPLE_BUDGET_BYTES = 5 * 1024 * 1024
"""Design Decision 7: checked-in samples stay under 5 MB."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture an OpenSky /states/all sample into canonical trajectory Parquet."
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="WGS84 bounding box in degrees.",
    )
    window = parser.add_mutually_exclusive_group()
    window.add_argument(
        "--time",
        nargs=2,
        type=int,
        metavar=("START_S", "END_S"),
        help="UTC epoch-second window [start, end). Historical windows need "
        "authenticated access; anonymous polling always sees the current snapshot.",
    )
    window.add_argument(
        "--duration-s",
        type=int,
        default=300,
        help="Capture live from now for this many seconds (default: 300).",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=DEFAULT_POLL_INTERVAL_S,
        help="Seconds between snapshots (default: %(default)s; anonymous "
        "OpenSky resolution is 10 s).",
    )
    parser.add_argument(
        "--credentials",
        type=str,
        default=None,
        metavar="USER:PASS",
        help="Optional OpenSky basic-auth credentials.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output Parquet path (default: %(default)s).",
    )
    return parser.parse_args(argv)


def resolve_window(args: argparse.Namespace, now_s: int) -> TimeRange:
    """Turn --time/--duration-s into a half-open epoch-second window."""
    if args.time is not None:
        start_s, end_s = args.time
        return TimeRange(start_s=start_s, end_s=end_s)
    return TimeRange(start_s=now_s, end_s=now_s + args.duration_s)


def capture(
    bbox: BoundingBox,
    window: TimeRange,
    *,
    poll_interval_s: float,
    credentials: tuple[str, str] | None,
    client: httpx.Client | None = None,
) -> list[TrajectoryRecord]:
    """Collect one snapshot per poll interval across ``window``.

    Ticks in the future are waited out against the wall clock so a
    live anonymous capture actually spans the window instead of
    re-reading one snapshot back-to-back.
    """
    owns_client = client is None
    http = client or httpx.Client(base_url=OPENSKY_BASE_URL, timeout=30.0, auth=credentials)
    records: list[TrajectoryRecord] = []
    step_s = max(int(poll_interval_s), 1)
    try:
        for tick_s in range(window.start_s, window.end_s, step_s):
            wait_s = tick_s - time.time()
            if wait_s > 0:
                time.sleep(wait_s)
            snapshot = TimeRange(start_s=tick_s, end_s=tick_s + 1)
            records.extend(fetch_state_vectors(bbox, snapshot, poll_interval_s=1.0, client=http))
    finally:
        if owns_client:
            http.close()
    return records


def dedup_sort(records: list[TrajectoryRecord]) -> list[TrajectoryRecord]:
    """Drop duplicate ``(icao24, timestamp_us)`` observations and sort.

    Anonymous re-polls return aircraft whose position has not updated
    since the previous snapshot; the canonical dedup key collapses
    them (spec: dedup on ``(icao24, timestamp_us)``).
    """
    unique: dict[tuple[str, int], TrajectoryRecord] = {}
    for record in records:
        unique.setdefault((record.icao24, record.timestamp_us), record)
    return [unique[key] for key in sorted(unique)]


def write_sample(records: list[TrajectoryRecord], out: Path) -> tuple[int, str]:
    """Write a flat canonical-schema Parquet file; return (bytes, sha256)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(records_to_table(records), out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    return out.stat().st_size, digest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    lat_min, lat_max, lon_min, lon_max = args.bbox
    bbox = BoundingBox(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
    window = resolve_window(args, now_s=int(time.time()))

    credentials: tuple[str, str] | None = None
    if args.credentials is not None:
        user, _, password = args.credentials.partition(":")
        if not user or not password:
            raise SystemExit("--credentials must be USER:PASS")
        credentials = (user, password)

    raw = capture(
        bbox,
        window,
        poll_interval_s=args.poll_interval_s,
        credentials=credentials,
    )
    records = dedup_sort(raw)
    if not records:
        raise SystemExit("no records captured: empty bbox/window or OpenSky outage")

    size_bytes, digest = write_sample(records, args.out)
    aircraft = len({r.icao24 for r in records})
    print(f"wrote {args.out}: {len(records)} records ({len(raw)} raw), {aircraft} aircraft")
    print(f"size: {size_bytes} bytes, sha256: {digest}")
    if size_bytes > SAMPLE_BUDGET_BYTES:
        print(
            f"warning: exceeds the {SAMPLE_BUDGET_BYTES} byte checked-in sample budget "
            "(design Decision 7); narrow the bbox or window before committing",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
