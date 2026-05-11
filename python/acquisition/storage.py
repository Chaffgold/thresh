"""Partitioned Parquet storage for canonical trajectory records.

Files are written at::

    <root>/source=<opensky|adsbx>/date=YYYY-MM-DD/trajectories.parquet

Each writer flush asserts that timestamps within each ``icao24`` are
monotonically increasing (per spec scenario "Writing a day's worth
of OpenSky data").
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from acquisition.schema import TRAJECTORY_PARQUET_SCHEMA, Source, TrajectoryRecord


def _date_partition(timestamp_us: int) -> str:
    return datetime.fromtimestamp(timestamp_us / 1_000_000, tz=UTC).strftime("%Y-%m-%d")


def _records_to_arrays(records: list[TrajectoryRecord]) -> dict[str, list[Any]]:
    arrays: dict[str, list[Any]] = {field.name: [] for field in TRAJECTORY_PARQUET_SCHEMA}
    for r in records:
        arrays["icao24"].append(r.icao24)
        arrays["timestamp_us"].append(r.timestamp_us)
        arrays["lat"].append(r.lat)
        arrays["lon"].append(r.lon)
        arrays["alt_geom_m"].append(r.alt_geom_m)
        arrays["alt_baro_m"].append(r.alt_baro_m)
        arrays["vel_ground_mps"].append(r.vel_ground_mps)
        arrays["track_deg"].append(r.track_deg)
        arrays["vrate_mps"].append(r.vrate_mps)
        arrays["category"].append(r.category)
        arrays["callsign"].append(r.callsign)
        arrays["quality_nic"].append(r.quality_nic)
        arrays["quality_nac_p"].append(r.quality_nac_p)
        arrays["source"].append(r.source)
        arrays["provenance"].append(
            list(r.provenance.items()) if r.provenance is not None else None
        )
    return arrays


def _assert_monotonic_per_icao(records: list[TrajectoryRecord]) -> None:
    last_seen: dict[str, int] = {}
    for r in records:
        prev = last_seen.get(r.icao24)
        if prev is not None and r.timestamp_us < prev:
            raise ValueError(
                f"non-monotonic timestamps for icao24={r.icao24}: "
                f"saw {r.timestamp_us} after {prev}"
            )
        last_seen[r.icao24] = r.timestamp_us


def records_to_table(records: list[TrajectoryRecord]) -> pa.Table:
    """Convert canonical records to an Arrow table matching the schema.

    Arrays are constructed with explicit types so pyarrow does not
    dictionary-encode columns where the input happens to be uniform
    (e.g. ``source`` will always be a single value within one file).
    """
    arrays_dict = _records_to_arrays(records)
    pa_arrays = [
        pa.array(arrays_dict[field.name], type=field.type) for field in TRAJECTORY_PARQUET_SCHEMA
    ]
    return pa.Table.from_arrays(pa_arrays, schema=TRAJECTORY_PARQUET_SCHEMA)


def write_partition(
    records: Iterable[TrajectoryRecord],
    root: Path,
    source: Source,
) -> list[Path]:
    """Write ``records`` under ``<root>/source=<source>/date=YYYY-MM-DD/``.

    Records may span multiple UTC days; one file is written per day.
    Records within each day-partition must be monotonic per icao24
    (enforced; raises ``ValueError`` on violation).

    Returns the list of file paths written.
    """
    by_day: dict[str, list[TrajectoryRecord]] = {}
    for r in records:
        if r.source != source:
            raise ValueError(
                f"record source mismatch: got '{r.source}', writer expected '{source}'"
            )
        by_day.setdefault(_date_partition(r.timestamp_us), []).append(r)

    written: list[Path] = []
    for date, day_records in by_day.items():
        day_records.sort(key=lambda r: (r.icao24, r.timestamp_us))
        _assert_monotonic_per_icao(day_records)
        out_dir = root / f"source={source}" / f"date={date}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "trajectories.parquet"
        pq.write_table(records_to_table(day_records), out_path)
        written.append(out_path)
    return written


def read_partition(path: Path) -> list[TrajectoryRecord]:
    """Read a parquet partition back into canonical records.

    Uses :class:`ParquetFile` directly rather than the higher-level
    dataset API: the latter would walk up the ``source=…/date=…/``
    directory structure and infer Hive partition columns, which
    collides with the in-file ``source`` column we deliberately keep
    in the canonical schema.

    Used by tests and downstream consumers that want validated
    Pydantic objects rather than raw Arrow rows.
    """
    table = pq.ParquetFile(path).read()
    records: list[TrajectoryRecord] = []
    rows = table.to_pylist()
    for row in rows:
        prov_raw = row.get("provenance")
        if prov_raw is not None:
            row["provenance"] = dict(prov_raw)
        records.append(TrajectoryRecord.model_validate(row))
    return records
