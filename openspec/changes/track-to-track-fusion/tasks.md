# Distributed Track-to-Track Fusion — Tasks

## 1. Core types

- [x] 1.1 Add `TrackExchange` struct (track_id, source_id, state, covariance, timestamp) to `crates/thresh-fusion/src/t2t.rs`; derives `Debug`, `Clone`
- [x] 1.2 Add `From<&Track>` impl for `TrackExchange` to convert internal tracks to exchange format
- [x] 1.3 Add `FusionMode` enum (`Naive`, `CovarianceIntersection`) to `crates/thresh-fusion/src/t2t.rs`

## 2. Track-to-track association

- [x] 2.1 Implement `augmented_mahalanobis` distance function in `crates/thresh-fusion/src/t2t.rs`
- [x] 2.2 Implement `augmented_mahalanobis_with_cross_cov(x1, P1, x2, P2, P12) -> f64` for the cross-covariance-aware variant
- [x] 2.3 Implement `t2t_association` function building cost matrix and calling Hungarian solver in `crates/thresh-fusion/src/t2t.rs`
- [x] 2.4 Unit tests: two identical tracks associate, two distant tracks do not

## 3. Temporal alignment

- [x] 3.1 Implement `extrapolate_track(exchange: &TrackExchange, target_time: f64, f: &DMatrix, q: &DMatrix) -> TrackExchange` in `crates/thresh-fusion/src/t2t.rs`
- [x] 3.2 Implement batch alignment: `align_to_common_time(tracks: &mut [TrackExchange], f, q)` extrapolating all tracks to the latest timestamp
- [x] 3.3 Unit test: extrapolation of a constant-velocity track matches manual F*x + Q computation

## 4. Federated fusion manager

- [x] 4.1 Implement `FederatedFusionManager` struct with per-source track storage in `crates/thresh-fusion/src/t2t.rs`
- [x] 4.2 Implement `FederatedFusionManager::fuse()` orchestrating associate -> fuse -> birth for all sites
- [x] 4.3 Implement naive fusion mode: information filter sum `P_fused^{-1} = P1^{-1} + P2^{-1}` as `fuse_naive`
- [x] 4.4 Implement `fuse_covariance_intersection` reusing existing CI module for pairwise T2T fusion
- [ ] 4.5 ~~Implement optimal fusion with cross-covariance bookkeeping (opt-in mode)~~ **Deferred** — requires bookkeeping infrastructure not yet in place

## 5. Lifecycle and output

- [x] 5.1 Implement fused track birth from unmatched incoming tracks
- [x] 5.2 Implement fused track coasting and deletion when no source updates arrive within a configurable timeout
- [x] 5.3 Add `FederatedFusionManager::get_fused_tracks() -> &[TrackExchange]` for reading the common operating picture

## 6. Integration and testing

- [x] 6.1 Integration test: two simulated radar sites tracking the same three targets, federated fusion produces three fused tracks
- [x] 6.2 Integration test: asynchronous updates (site A at 1 Hz, site B at 2 Hz) produce temporally coherent fused output
- [x] 6.3 Add module to `crates/thresh-fusion/src/lib.rs`: `pub mod t2t;`
