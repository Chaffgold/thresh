"""Flight-data acquisition clients and canonical trajectory schema.

OpenSky Network (historical, redistributable) and ADS-B Exchange v2
(live, geographically scoped) are translated into a single canonical
Parquet/Arrow schema. Records are system-level / track-level truth;
they are never used as raw sensor measurements. See
`openspec/changes/flight-data-training-pipeline/specs/flight-data-acquisition/spec.md`.
"""
