"""Per-ICAO track stitching from point-in-time state-vector snapshots.

A "track" is a contiguous run of observations for a single ``icao24``
with no gap larger than ``gap_threshold_seconds`` (default 60 s). The
``stitch_tracks`` function partitions a stream of records into tracks
and yields one :class:`Track` per contiguous run.

See ``openspec/.../specs/flight-data-acquisition/spec.md`` → "Track
stitching from snapshots" for the authoritative behaviour.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

from acquisition.schema import TrajectoryRecord

DEFAULT_GAP_THRESHOLD_S: float = 60.0


@dataclass(frozen=True, slots=True)
class Track:
    """A contiguous run of observations for a single aircraft."""

    track_id: str
    """Stable identifier of the form ``{icao24}-{first_timestamp_iso}``."""

    icao24: str
    records: tuple[TrajectoryRecord, ...]

    @property
    def start_us(self) -> int:
        return self.records[0].timestamp_us

    @property
    def end_us(self) -> int:
        return self.records[-1].timestamp_us


def _make_track_id(icao24: str, first_us: int) -> str:
    first_iso = datetime.fromtimestamp(first_us / 1_000_000, tz=UTC).isoformat()
    return f"{icao24}-{first_iso}"


def stitch_tracks(
    records: Iterable[TrajectoryRecord],
    gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S,
) -> Iterator[Track]:
    """Split ``records`` into per-ICAO tracks on gaps > ``gap_threshold_s``.

    Records do not need to be pre-sorted; this function groups by
    ``icao24`` first, then sorts each group by timestamp, then splits.

    Yields tracks in undefined order across distinct ``icao24``s, but
    chronological order within each ``icao24``.
    """
    gap_threshold_us = int(gap_threshold_s * 1_000_000)

    by_icao: dict[str, list[TrajectoryRecord]] = {}
    for record in records:
        by_icao.setdefault(record.icao24, []).append(record)

    for icao24, recs in by_icao.items():
        recs.sort(key=lambda r: r.timestamp_us)
        current: list[TrajectoryRecord] = []
        for rec in recs:
            if current and (rec.timestamp_us - current[-1].timestamp_us) > gap_threshold_us:
                yield Track(
                    track_id=_make_track_id(icao24, current[0].timestamp_us),
                    icao24=icao24,
                    records=tuple(current),
                )
                current = []
            current.append(rec)
        if current:
            yield Track(
                track_id=_make_track_id(icao24, current[0].timestamp_us),
                icao24=icao24,
                records=tuple(current),
            )
