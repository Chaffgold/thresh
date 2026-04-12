## 1. JPDA Core (thresh-association)

- [ ] 1.1 Add `crates/thresh-association/src/jpda.rs` module. Define `JpdaConfig` struct with fields: `detection_prob: f64`, `clutter_density: f64`, `gate_threshold: f64`.
- [ ] 1.2 Define `JpdaAssociator` struct holding `JpdaConfig` and reusing `MahalanobisGating` for gate computation.
- [ ] 1.3 Implement gated pair enumeration: for each track, identify all detections within the Mahalanobis gate. Build a bipartite gating graph.
- [ ] 1.4 Implement independent cluster decomposition: partition tracks into independent clusters based on shared gated detections (connected components of the gating graph).
- [ ] 1.5 Implement joint event enumeration within each cluster: enumerate all feasible joint association events where each detection is assigned to at most one track, including a missed-detection event for each track.
- [ ] 1.6 Implement event probability computation: for each joint event, compute its probability from the product of individual track-detection likelihoods (Gaussian innovation likelihood), detection probability, and clutter density.
- [ ] 1.7 Implement marginal association probability computation: for each track-detection pair, sum the probabilities of all joint events that include that pair. Include the missed-detection marginal probability (sum of events where the track has no detection).
- [ ] 1.8 Verify that marginal probabilities sum to 1.0 for each track (all detection associations + missed detection).

## 2. JPDA Update (thresh-association)

- [ ] 2.1 Implement merged innovation computation: `v = sum_j(beta_j * v_j)` where `beta_j` is the association probability and `v_j` is the innovation for detection j.
- [ ] 2.2 Implement spread-of-innovations matrix: `Pv = sum_j(beta_j * v_j * v_j^T) - v * v^T`.
- [ ] 2.3 Implement JPDA state update: `x+ = x- + K * v` using the standard Kalman gain and merged innovation.
- [ ] 2.4 Implement JPDA covariance update with spread term: `P+ = beta_0 * P- + (1 - beta_0) * P_KF + K * Pv * K^T` where `P_KF` is the single-detection Kalman covariance.
- [ ] 2.5 Expose `JpdaAssociator::associate_and_update(tracks, detections, h, r) -> Vec<JpdaResult>` as the public API, returning updated state/covariance per track.

## 3. MHT Hypothesis Tree (thresh-association)

- [ ] 3.1 Add `crates/thresh-association/src/mht.rs` module. Define `MhtConfig` struct with fields: `n_scan: usize`, `k_best: usize`, `detection_prob: f64`, `clutter_density: f64`, `gate_threshold: f64`.
- [ ] 3.2 Define `Hypothesis` struct: `assignment: Vec<(TrackId, Option<usize>)>`, `score: f64`, `parent: Option<usize>`, `scan_index: usize`, `track_states: HashMap<TrackId, (DVector<f64>, DMatrix<f64>)>`.
- [ ] 3.3 Define `HypothesisTree` struct holding `Vec<Hypothesis>`, current scan index, and `MhtConfig`.
- [ ] 3.4 Implement hypothesis expansion: given parent hypotheses and a new scan of detections, generate all feasible child hypotheses. Each child assigns each detection to at most one track (or false alarm) and each track to at most one detection (or missed detection).
- [ ] 3.5 Implement cumulative score computation for each child hypothesis: parent score + sum of log-likelihoods for each track-detection assignment in the new scan.

## 4. MHT Pruning (thresh-association)

- [ ] 4.1 Implement k-best pruning: after hypothesis expansion, sort leaf hypotheses by score (descending), retain top k, discard the rest and deallocate their track states.
- [ ] 4.2 Implement N-scan pruning: walk back N scans from each leaf. For scan `t - N`, identify assignments that all surviving hypotheses agree on. Collapse those agreed-upon assignments, freeing the corresponding tree nodes.
- [ ] 4.3 Implement memory reclamation: after pruning, remove orphaned (unreachable) hypothesis nodes from the tree vector and compact indices.
- [ ] 4.4 Add hypothesis count monitoring: log a warning if the hypothesis count exceeds 2 * k_best after expansion (indicates pruning may be insufficient).

## 5. MHT Track Extraction (thresh-association)

- [ ] 5.1 Implement `HypothesisTree::best_hypothesis() -> &Hypothesis` returning the highest-scoring leaf.
- [ ] 5.2 Implement `HypothesisTree::extract_tracks() -> Vec<(TrackId, DVector<f64>, DMatrix<f64>)>` returning track states from the best hypothesis.
- [ ] 5.3 Implement track ID consistency: when extracting tracks, ensure IDs are consistent across timesteps by tracing the best hypothesis's association history.

## 6. Tracker Integration (thresh-tracker)

- [ ] 6.1 Define `AssociationStrategy` enum in thresh-tracker: `Hungarian`, `Jpda { detection_prob, clutter_density }`, `Mht { n_scan, k_best, detection_prob, clutter_density }`.
- [ ] 6.2 Add `association_strategy: AssociationStrategy` field to `MultiObjectTracker` and corresponding builder/constructor methods.
- [ ] 6.3 Refactor `MultiObjectTracker::step()` to dispatch association based on the strategy enum: Hungarian uses existing code, JPDA calls `JpdaAssociator`, MHT calls `HypothesisTree`.
- [ ] 6.4 For JPDA: after `associate_and_update`, map results back to Track updates (state, covariance, hit/miss for lifecycle).
- [ ] 6.5 For MHT: after `extract_tracks`, synchronize the tracker's track list with the MHT output (birth new tracks, update existing, mark lost).
- [ ] 6.6 Ensure backward compatibility: default `AssociationStrategy::Hungarian` so existing code continues to work unchanged.

## 7. Tests

- [ ] 7.1 Unit test: JPDA with a single track and single detection produces association probability ~1.0 (equivalent to Hungarian).
- [ ] 7.2 Unit test: JPDA with two closely-spaced tracks and one ambiguous detection produces association probabilities that sum to 1.0 per track.
- [ ] 7.3 Unit test: JPDA merged innovation is the weighted average of individual innovations.
- [ ] 7.4 Unit test: JPDA covariance update includes the spread-of-innovations term (covariance is larger than single-detection update).
- [ ] 7.5 Unit test: MHT hypothesis expansion produces the correct number of child hypotheses for a small example (2 tracks, 2 detections).
- [ ] 7.6 Unit test: k-best pruning retains exactly k hypotheses and discards the lowest-scoring ones.
- [ ] 7.7 Unit test: N-scan pruning collapses agreed-upon old assignments.
- [ ] 7.8 Integration test: JPDA on a crossing-tracks scenario (two targets cross paths) produces better MOTA than Hungarian.
- [ ] 7.9 Integration test: MHT on a dense clutter scenario (high false alarm rate) maintains track continuity better than Hungarian.
