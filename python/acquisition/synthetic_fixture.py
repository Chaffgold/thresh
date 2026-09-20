"""Generate the wholly synthetic canonical trajectory fixture without a network.

Run from ``python/`` with ``uv run python -m acquisition.synthetic_fixture``.
All identifiers, timestamps and coordinates are fixed invented values. The
``source="opensky"`` field emulates the existing provider schema; it does not
describe external data provenance. Every record and the Parquet metadata are
explicitly marked synthetic. This fixture only validates schema compatibility,
not API parity, realism, or real-data model acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow.parquet as pq

from acquisition.schema import TrajectoryRecord
from acquisition.storage import records_to_table

DEFAULT_OUT = (
    Path(__file__).resolve().parents[2] / "test-data" / "trajectories" / "synthetic-sample.parquet"
)
FIXTURE_METADATA = {
    b"thresh.fixture.origin": b"synthetic",
    b"thresh.fixture.generator": b"acquisition.synthetic_fixture",
}


def synthetic_records() -> list[TrajectoryRecord]:
    """Return three invented linear trajectories, each with 30 ten-second steps."""
    records: list[TrajectoryRecord] = []
    for aircraft, category in enumerate(("A1", "A4", "A7")):
        for step in range(30):
            altitude = 1000.0 + aircraft * 100.0 + step * 5.0
            records.append(
                TrajectoryRecord(
                    icao24=f"{aircraft + 1:06x}",
                    timestamp_us=(1_700_000_000 + step * 10) * 1_000_000,
                    lat=10.0 + aircraft * 0.01,
                    lon=20.0 + aircraft * 0.01 + step * 0.001,
                    alt_geom_m=altitude,
                    alt_baro_m=altitude - 25.0,
                    vel_ground_mps=11.0,
                    track_deg=90.0,
                    vrate_mps=0.5,
                    category=category,
                    callsign=f"SYNTH{aircraft + 1:03d}",
                    source="opensky",
                    provenance={"synthetic": True},
                )
            )
    return records


def write_fixture(out: Path) -> tuple[int, str]:
    """Write synthetic rows and provenance metadata; return (bytes, sha256)."""
    table = records_to_table(synthetic_records()).replace_schema_metadata(FIXTURE_METADATA)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return out.stat().st_size, hashlib.sha256(out.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    size_bytes, digest = write_fixture(args.out)
    print(f"wrote synthetic fixture {args.out}: {size_bytes} bytes, sha256: {digest}")


if __name__ == "__main__":
    main()
