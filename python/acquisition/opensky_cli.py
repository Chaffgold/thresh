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
    OpenSkyError,
    TimeRange,
    fetch_current_states,
    fetch_state_vectors,
)
from acquisition.opensky_auth import (
    DEFAULT_CREDENTIALS_PATH,
    ENV_CLIENT_ID,
    ENV_CLIENT_SECRET,
    OpenSkyAuthError,
    build_auth,
    load_credentials,
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
        help=argparse.SUPPRESS,  # removed; kept only to give a migration error
    )
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"OpenSky OAuth2 credentials JSON (default: {DEFAULT_CREDENTIALS_PATH} "
        f"if present). Overridden by the {ENV_CLIENT_ID}/{ENV_CLIENT_SECRET} "
        "environment variables. Secrets are never accepted as flags — argv is "
        "visible to other users via `ps`.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output Parquet path (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    if args.credentials is not None:
        parser.error(
            "--credentials (basic auth) was removed: OpenSky's REST API now requires "
            f"OAuth2. Set {ENV_CLIENT_ID}/{ENV_CLIENT_SECRET} in the environment, or "
            "pass --credentials-file with a credentials.json from "
            "https://opensky-network.org/my-opensky/api-client . "
            "Do not pass secrets as flags — argv is visible to other users via `ps`."
        )
    return args


def resolve_window(args: argparse.Namespace, now_s: int) -> TimeRange:
    """Turn --time/--duration-s into a half-open epoch-second window."""
    if args.time is not None:
        start_s, end_s = args.time
        return TimeRange(start_s=start_s, end_s=end_s)
    return TimeRange(start_s=now_s, end_s=now_s + args.duration_s)


def capture(
    bbox: BoundingBox,
    duration_s: int,
    *,
    poll_interval_s: float,
    auth: httpx.Auth | None = None,
    client: httpx.Client | None = None,
    retries: int = 3,
    backoff_s: float = 1.0,
) -> list[TrajectoryRecord]:
    """Live-poll *current* snapshots for ``duration_s``, one per interval.

    Pacing uses monotonic-clock deltas (never absolute epoch ticks, which
    break if NTP steps the clock mid-run) and each poll omits the ``time``
    parameter (anonymous access 403s on even slightly-stale times).
    A mid-run OpenSky error keeps the records collected so far instead of
    losing the whole capture.
    """
    owns_client = client is None
    http = client or httpx.Client(base_url=OPENSKY_BASE_URL, timeout=30.0, auth=auth)
    records: list[TrajectoryRecord] = []
    step_s = max(poll_interval_s, 1.0)
    polls = max(int(duration_s / step_s), 1)
    started = time.monotonic()
    try:
        for poll in range(polls):
            wait_s = started + poll * step_s - time.monotonic()
            if wait_s > 0:
                time.sleep(wait_s)
            try:
                records.extend(
                    fetch_current_states(bbox, client=http, retries=retries, backoff_s=backoff_s)
                )
            except OpenSkyError as exc:
                print(
                    f"warning: poll {poll + 1}/{polls} failed ({exc}); "
                    f"keeping {len(records)} records collected so far",
                    file=sys.stderr,
                    flush=True,
                )
                break
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

    try:
        credentials = load_credentials(path=args.credentials_file)
    except OpenSkyAuthError as exc:
        raise SystemExit(str(exc)) from exc
    auth = build_auth(credentials)
    if credentials is None:
        print(
            "note: no OpenSky credentials found — polling anonymously "
            f"(~400 credits/day). Set {ENV_CLIENT_ID}/{ENV_CLIENT_SECRET} or place "
            f"credentials.json at {DEFAULT_CREDENTIALS_PATH} for 4,000/day.",
            file=sys.stderr,
        )

    if args.time is not None:
        # Historical replay of an explicit window (needs authenticated access).
        if auth is None:
            raise SystemExit(
                "--time requires authentication: anonymous access always returns the "
                f"current snapshot. Set {ENV_CLIENT_ID}/{ENV_CLIENT_SECRET} first."
            )
        window = resolve_window(args, now_s=int(time.time()))
        raw = list(
            fetch_state_vectors(
                bbox,
                window,
                poll_interval_s=args.poll_interval_s,
                auth=auth,
            )
        )
    else:
        raw = capture(
            bbox,
            args.duration_s,
            poll_interval_s=args.poll_interval_s,
            auth=auth,
        )
    records = dedup_sort(raw)
    if not records:
        raise SystemExit("no records captured: empty bbox/window or OpenSky outage")

    size_bytes, digest = write_sample(records, args.out)
    aircraft = len({r.icao24 for r in records})
    print(
        f"wrote {args.out}: {len(records)} records ({len(raw)} raw), {aircraft} aircraft",
        flush=True,
    )
    print(f"size: {size_bytes} bytes, sha256: {digest}", flush=True)
    if size_bytes > SAMPLE_BUDGET_BYTES:
        print(
            f"warning: exceeds the {SAMPLE_BUDGET_BYTES} byte checked-in sample budget "
            "(design Decision 7); narrow the bbox or window before committing",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
