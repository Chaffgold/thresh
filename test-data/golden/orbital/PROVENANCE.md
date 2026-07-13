# SGP4 Golden Fixture Provenance

Tier-3 golden vectors for design Decision 8 of the
`orbital-ballistic-filter-models` change: TEME position/velocity samples
at t0 and t0 + {60, 300, 600} s, propagated by the AIAA-verified [`sgp4`
crate](https://crates.io/crates/sgp4) and committed so CI never
regenerates them. Units in the JSON files are metres and metres/second.

Consumed by the default-feature envelope test
`crates/thresh-data/tests/golden_orbital.rs`: `KeplerJ2` initialized from
each fixture's t0 state must stay within 5 km position error of the SGP4
samples out to 600 s. The envelope is deliberately generous — SGP4
carries drag and short-periodic terms that Kepler+J2 osculating physics
legitimately lacks; exactness is certified by golden tiers 1 and 2.

## Generation record

- `sgp4` crate version: **1.2.2** (from the workspace `Cargo.lock` at generation time)
- Generator: `crates/thresh-data/src/bin/gen_sgp4_fixtures.rs` (feature-gated, never in CI)
- Regeneration command (run from the workspace root, then commit the output):

```sh
cargo run -p thresh-data --features orbital --bin gen-sgp4-fixtures
```

Repeated runs at the same `sgp4` version are bit-identical (constant
inline TLEs, deterministic `serde_json` float formatting).

## Objects and TLEs (verbatim)

### ISS (ZARYA) — `sgp4-25544-iss.json`

Source: cached repo TLE `crates/thresh-data/scenarios/orbital-iss.tle`.

```text
1 25544U 98067A   24001.00000000  .00016717  00000-0  10270-3 0  9026
2 25544  51.6400 208.9163 0006703  30.1579 330.0018 15.49560455    18
```

### AIAA object 06251 (DELTA 1 DEB) — `sgp4-06251-delta1deb.json`

Source: AIAA-2006-6753 verification TLE set (sgp4 crate `test_cases.toml`).

```text
1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  3985
2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  6774
```

### AIAA object 28057 (sun-synchronous LEO) — `sgp4-28057-sunsync.json`

Source: AIAA-2006-6753 verification TLE set (sgp4 crate `test_cases.toml`).

```text
1 28057U 03049A   06177.78615833  .00000060  00000-0  35940-4 0  1836
2 28057  98.4283 247.6961 0000884  88.1964 271.9322 14.35478080140550
```

