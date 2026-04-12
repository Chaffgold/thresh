# Distributed Track-to-Track Fusion — Tasks

## 1. Core types

- [ ] 1.1 Add `TrackExchange` struct to `crates/thresh-core/src/track.rs` with `track_id`, `source_id`, `state`, `covariance`, `timestamp`, `class`, `confidence` fields; derive `Serialize`/`Deserialize`
- [ ] 1.2 Add `From<&Track>` impl for `TrackExchange` to convert internal tracks to exchange format
- [ ] 1.3 Add `FusionMode` enum (`Naive`, `CovarianceIntersection`, `OptimalCrossCovariance`) to `crates/thresh-fusion/src/federated.rs`

## 2. Track-to-track association

- [ ] 2.1 Create `crates/thresh-association/src/t2t.rs` with `augmented_mahalanobis(x1, P1, x2, P2) -> f64` distance function
- [ ] 2.2 Implement `augmented_mahalanobis_with_cross_cov(x1, P1, x2, P2, P12) -> f64` for the cross-covariance-aware variant
- [ ] 2.3 Implement `T2TAssociator` struct with `associate(source_tracks: &[Vec<TrackExchange>], gate: f64) -> AssignmentResult` building cost matrix and calling Hungarian solver
- [ ] 2.4 Unit tests: two identical tracks associate, two distant tracks do not, cross-covariance variant produces tighter distance

## 3. Temporal alignment

- [ ] 3.1 Implement `extrapolate_track(exchange: &TrackExchange, target_time: f64, model: &dyn MotionModel) -> TrackExchange` in `crates/thresh-fusion/src/temporal.rs`
- [ ] 3.2 Implement batch alignment: `align_to_common_time(tracks: &mut [TrackExchange])` extrapolating all tracks to the latest timestamp
- [ ] 3.3 Unit test: extrapolation of a constant-velocity track matches manual F*x + Q computation

## 4. Federated fusion manager

- [ ] 4.1 Implement `FederatedFusionManager` struct with fused track table, per-source history, and `FusionMode` config
- [ ] 4.2 Implement `FederatedFusionManager::update(&mut self, incoming: Vec<Vec<TrackExchange>>)` orchestrating align -> associate -> fuse -> lifecycle
- [ ] 4.3 Implement naive fusion mode: information filter sum `P_fused^{-1} = P1^{-1} + P2^{-1}`
- [ ] 4.4 Extend existing `covariance_intersection` module to support the federated pairwise fusion case
- [ ] 4.5 Implement optimal fusion with cross-covariance bookkeeping (opt-in mode)

## 5. Lifecycle and output

- [ ] 5.1 Implement fused track birth from unmatched incoming tracks
- [ ] 5.2 Implement fused track coasting and deletion when no source updates arrive within a configurable timeout
- [ ] 5.3 Add `FederatedFusionManager::get_fused_tracks() -> Vec<TrackExchange>` for reading the common operating picture

## 6. Integration and testing

- [ ] 6.1 Integration test: two simulated radar sites tracking the same three targets, federated fusion produces three fused tracks
- [ ] 6.2 Integration test: asynchronous updates (site A at 1 Hz, site B at 2 Hz) produce temporally coherent fused output
- [ ] 6.3 Add module to `crates/thresh-fusion/src/lib.rs`: `pub mod federated;` and `pub mod temporal;`
