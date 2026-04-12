## 1. JPDA Core (thresh-association)

- [x] 1.1 Add `crates/thresh-association/src/jpda.rs` module. Define `JpdaTrack` struct and `JpdaResult` struct. Parameters (`p_detection`, `clutter_density`, `gate`) are passed to `jpda_probabilities()` directly.
- [x] 1.2 Implemented as a free function `jpda_probabilities()` reusing `MahalanobisGating` (`mahalanobis_squared`) for gate computation.
- [x] 1.3 Implement gated pair enumeration: for each track, identify all detections within the Mahalanobis gate. Build a bipartite gating graph.
- [ ] 1.4 Implement independent cluster decomposition: partition tracks into independent clusters based on shared gated detections (connected components of the gating graph).
- [x] 1.5 Implement joint event enumeration within each cluster: enumerate all feasible joint association events where each detection is assigned to at most one track, including a missed-detection event for each track.
- [x] 1.6 Implement event probability computation: for each joint event, compute its probability from the product of individual track-detection likelihoods (Gaussian innovation likelihood), detection probability, and clutter density.
- [x] 1.7 Implement marginal association probability computation: for each track-detection pair, sum the probabilities of all joint events that include that pair. Include the missed-detection marginal probability (sum of events where the track has no detection).
- [x] 1.8 Verify that marginal probabilities sum to 1.0 for each track (all detection associations + missed detection).

## 2. JPDA Update (thresh-association)

- [x] 2.1 Implement merged innovation computation: `jpda_combined_innovation()` computes `v = sum_j(beta_j * v_j)`.
- [x] 2.2 Implement spread-of-innovations matrix: `jpda_covariance_correction()` computes `Pv = sum_j(beta_j * v_j * v_j^T) - v * v^T`.
- [x] 2.3 Implement JPDA state update: `x+ = x- + K * v` using the standard Kalman gain and merged innovation.
- [x] 2.4 Implement JPDA covariance update with spread term: `P+ = beta_0 * P- + (1 - beta_0) * P_KF + K * Pv * K^T` where `P_KF` is the single-detection Kalman covariance.
- [x] 2.5 Expose `JpdaAssociator::associate_and_update(tracks, detections, h, r) -> Vec<JpdaResult>` as the public API, returning updated state/covariance per track.

## 3. MHT Hypothesis Tree (thresh-association)

- [x] 3.1 Add `crates/thresh-association/src/mht.rs` module. Config is passed via `HypothesisTree::new(max_hypotheses, n_scan_depth)`.
- [x] 3.2 Define `Hypothesis` struct with `assignments: Vec<(usize, Option<usize>)>` and `log_likelihood: f64`.
- [x] 3.3 Define `HypothesisTree` struct holding `Vec<Hypothesis>`, `max_hypotheses`, and `n_scan_depth`.
- [x] 3.4 Implement hypothesis expansion via `HypothesisTree::expand()` with recursive enumeration of feasible joint events.
- [x] 3.5 Implement cumulative score computation for each child hypothesis: parent score + sum of log-likelihoods for each track-detection assignment in the new scan.

## 4. MHT Pruning (thresh-association)

- [x] 4.1 Implement k-best pruning via `HypothesisTree::prune_k_best()`: sort by score descending, retain top k.
- [x] 4.2 Implement N-scan pruning: walk back N scans from each leaf. For scan `t - N`, identify assignments that all surviving hypotheses agree on. Collapse those agreed-upon assignments, freeing the corresponding tree nodes.
- [x] 4.3 Implement memory reclamation: after pruning, remove orphaned (unreachable) hypothesis nodes from the tree vector and compact indices.
- [x] 4.4 Add hypothesis count monitoring: log a warning if the hypothesis count exceeds 2 * k_best after expansion (indicates pruning may be insufficient).

## 5. MHT Track Extraction (thresh-association)

- [x] 5.1 Implement `HypothesisTree::best_hypothesis() -> Option<&Hypothesis>` returning the highest-scoring leaf.
- [x] 5.2 Implement `HypothesisTree::marginal_probabilities()` returning per-pair association probabilities across all hypotheses.
- [x] 5.3 Implement track ID consistency: when extracting tracks, ensure IDs are consistent across timesteps by tracing the best hypothesis's association history.

## 6. Tracker Integration (thresh-tracker)

- [ ] 6.1 Define `AssociationStrategy` enum in thresh-tracker: `Hungarian`, `Jpda { detection_prob, clutter_density }`, `Mht { n_scan, k_best, detection_prob, clutter_density }`.
- [ ] 6.2 Add `association_strategy: AssociationStrategy` field to `MultiObjectTracker` and corresponding builder/constructor methods.
- [ ] 6.3 Refactor `MultiObjectTracker::step()` to dispatch association based on the strategy enum: Hungarian uses existing code, JPDA calls `JpdaAssociator`, MHT calls `HypothesisTree`.
- [ ] 6.4 For JPDA: after `associate_and_update`, map results back to Track updates (state, covariance, hit/miss for lifecycle).
- [ ] 6.5 For MHT: after `extract_tracks`, synchronize the tracker's track list with the MHT output (birth new tracks, update existing, mark lost).
- [ ] 6.6 Ensure backward compatibility: default `AssociationStrategy::Hungarian` so existing code continues to work unchanged.

## 7. Tests

- [x] 7.1 Unit test: JPDA with a single track and single detection produces association probability ~1.0 (`test_jpda_single_track_single_det`).
- [x] 7.2 Unit test: JPDA with multiple tracks produces association probabilities that sum to 1.0 per track (`test_jpda_probabilities_sum_to_one`).
- [x] 7.3 Unit test: JPDA merged innovation is the weighted average of individual innovations (`test_jpda_combined_innovation`).
- [x] 7.4 Unit test: JPDA spread-of-innovations covariance correction is PSD (`test_jpda_covariance_correction_positive_semidefinite`).
- [x] 7.5 Unit test: MHT hypothesis expansion produces the correct number of child hypotheses for a small example (`test_mht_expand_generates_hypotheses`).
- [x] 7.6 Unit test: k-best pruning retains exactly k hypotheses and discards the lowest-scoring ones (`test_mht_prune_k_best`).
- [x] 7.7 Unit test: N-scan pruning collapses agreed-upon old assignments.
- [ ] 7.8 Integration test: JPDA on a crossing-tracks scenario (two targets cross paths) produces better MOTA than Hungarian.
- [ ] 7.9 Integration test: MHT on a dense clutter scenario (high false alarm rate) maintains track continuity better than Hungarian.
