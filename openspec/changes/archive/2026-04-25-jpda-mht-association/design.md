## Context

thresh-association currently implements the Hungarian algorithm for single-assignment data association and Mahalanobis-distance gating for filtering infeasible track-detection pairs. The Hungarian algorithm makes hard (irrevocable) assignments at each timestep: each detection is assigned to at most one track, and each track receives at most one detection. This works well in sparse, well-separated target scenarios but degrades in dense environments with closely-spaced targets, crossing tracks, or high false alarm rates.

Two classical alternatives exist: Joint Probabilistic Data Association (JPDA) and Multi-Hypothesis Tracking (MHT). JPDA handles ambiguity by computing soft association probabilities and updating each track with a probability-weighted combination of all feasible detections. MHT handles ambiguity by deferring hard decisions, maintaining multiple global association hypotheses over a sliding window. Both are currently available in thresh only via the Stone Soup PyO3 bridge (`thresh-bridge`), which adds a Python runtime dependency and limits deployment to environments with Python installed.

The existing gating infrastructure (`MahalanobisGating`) and cost matrix computation can be reused by both JPDA and MHT, since both algorithms operate on the same set of gated track-detection pairs.

## Goals / Non-Goals

**Goals:**
- Implement JPDA with merged-measurement update in thresh-association
- Implement hypothesis-oriented MHT with N-scan and k-best pruning in thresh-association
- Reuse existing Mahalanobis gating infrastructure for both algorithms
- Add `AssociationStrategy` enum to `MultiObjectTracker` for selecting the association method
- Integrate JPDA and MHT into the tracker's `step()` method
- Provide configurable parameters (detection probability, false alarm density, N-scan depth, k-best count)
- Test against cluttered scenarios where Hungarian performance degrades

**Non-Goals:**
- Probabilistic MHT (PMHT) -- a fundamentally different algorithm
- Track-oriented MHT (only hypothesis-oriented)
- GPU acceleration of combinatorial hypothesis enumeration
- Learned association (GNN-based matching)
- Replacing Hungarian as the default (it remains the default for backward compatibility)

## Decisions

### 1. JPDA lives in `thresh-association/src/jpda.rs`

**Decision:** Add a `jpda` module to thresh-association with a `JpdaAssociator` struct that implements the JPDA algorithm. It reuses `MahalanobisGating` for filtering and computes association probabilities from the gated pairs.

**Rationale:** JPDA is a data association algorithm, not a filter or tracker. It belongs in thresh-association alongside the Hungarian implementation. The struct-based design allows configuration (detection probability, clutter density) to be set at construction and reused across timesteps.

### 2. JPDA association probability computation via joint event enumeration

**Decision:** Compute marginal association probabilities by:
1. Enumerate all feasible joint association events (each detection assigned to at most one track) within the gated set
2. Compute each event's probability from individual track-detection likelihoods
3. Marginalize to get per-pair probabilities by summing event probabilities

For tractability, limit enumeration to events involving tracks with overlapping gates. Tracks with non-overlapping gates are independent and can use direct (non-combinatorial) probability computation.

**Rationale:** Exact enumeration is combinatorially expensive in the worst case (O(M! / (M-T)!) for M detections and T tracks) but tractable when gating limits the number of feasible pairs per track. In practice, with Mahalanobis gating, each track typically has 1-5 gated detections, making exact enumeration feasible for scenarios with up to ~50 tracks. For larger scenarios, the independent-gates optimization reduces the problem to small independent subsets.

### 3. JPDA merged-measurement update

**Decision:** The JPDA update for each track computes:
- Merged innovation: `v = sum_j(beta_j * v_j)` where `beta_j` is the association probability for detection j and `v_j` is the innovation
- State update: `x+ = x- + K * v` (standard Kalman gain times merged innovation)
- Covariance update with spread of innovations: `P+ = beta_0 * P- + (1 - beta_0) * P_KF + K * Pv * K^T` where `beta_0` is the missed-detection probability, `P_KF` is the standard Kalman update covariance, and `Pv = sum_j(beta_j * v_j * v_j^T) - v * v^T` is the spread of innovations

**Rationale:** The merged-measurement approach (Bar-Shalom & Li, 2009) avoids maintaining per-detection filter copies. The spread-of-innovations term inflates the covariance to reflect association uncertainty, which is the key advantage of JPDA over hard assignment.

### 4. MHT hypothesis tree structure

**Decision:** Represent the hypothesis tree as a `Vec<Hypothesis>` where each `Hypothesis` contains:
- A global association assignment (track-detection mapping for one scan)
- A cumulative log-likelihood score
- A parent index (for tree traversal and N-scan pruning)
- Per-track filter states (cloned at each branch point)

```
pub struct Hypothesis {
    pub assignment: Vec<(TrackId, Option<DetectionIdx>)>,
    pub score: f64,
    pub parent: Option<usize>,
    pub scan_index: usize,
    pub track_states: HashMap<TrackId, (DVector<f64>, DMatrix<f64>)>,
}
```

**Rationale:** A flat vector with parent pointers is simpler and more cache-friendly than a recursive tree structure. N-scan pruning operates by walking parent pointers, and k-best pruning operates by sorting the vector by score. Storing per-hypothesis track states is memory-intensive but necessary for correctness -- different hypotheses may have different track states due to different association histories.

### 5. N-scan pruning collapses old agreement

**Decision:** After each scan, walk back N levels from each leaf hypothesis. If all surviving hypotheses agree on the assignment at scan `t - N`, collapse those hypotheses by removing the agreed-upon scan from the tree and merging the common ancestor. Effectively, this commits to associations that are N scans old.

**Rationale:** N-scan pruning bounds the tree depth, preventing unbounded memory growth. N=2 or N=3 is standard in the literature. The key insight is that if all surviving hypotheses agree on an old assignment, there is no remaining ambiguity about that assignment and it can be committed.

### 6. K-best pruning via sorted truncation

**Decision:** After hypothesis expansion at each scan, sort all leaf hypotheses by cumulative log-likelihood and retain only the top k. Default k=100.

**Rationale:** K-best pruning bounds the tree width. It is simpler than Murty's algorithm (which finds the k-best assignments optimally) but sufficient when combined with N-scan pruning. The log-likelihood scoring naturally favors hypotheses with consistent, high-quality associations.

### 7. `AssociationStrategy` enum on `MultiObjectTracker`

**Decision:** Add an enum to the tracker configuration:

```
pub enum AssociationStrategy {
    Hungarian,
    Jpda { detection_prob: f64, clutter_density: f64 },
    Mht { n_scan: usize, k_best: usize, detection_prob: f64, clutter_density: f64 },
}
```

The tracker's `step()` method dispatches to the appropriate association algorithm based on this enum.

**Rationale:** A configuration enum is simpler than a trait object for the three known strategies. It keeps the tracker's public API clean and makes strategy selection a construction-time decision. Default is `Hungarian` for backward compatibility.

### 8. Missed-detection and false-alarm modeling

**Decision:** Both JPDA and MHT require two parameters beyond the gating threshold:
- `detection_prob` (Pd): probability that a true target generates a detection (default: 0.9)
- `clutter_density` (lambda): spatial density of false alarms per unit volume (default: 1e-6)

These parameterize the missed-detection hypothesis probability and the false-alarm likelihood.

**Rationale:** These are standard parameters in Bayesian data association (Bar-Shalom & Li). They must be configured per-scenario since they depend on the sensor characteristics and the environment.

## Risks / Trade-offs

**[Risk] Combinatorial explosion in MHT.** Without careful pruning, the hypothesis tree grows exponentially with each scan. Mitigation: N-scan pruning (bounds depth) + k-best pruning (bounds width) together keep the tree manageable. Default parameters (N=3, k=100) are conservative. Add monitoring for hypothesis count and warn if pruning is insufficient.

**[Risk] JPDA joint event enumeration is expensive for dense scenarios.** With many tracks and many detections per gate, the number of joint events grows factorially. Mitigation: the independent-gates optimization partitions tracks into independent clusters. Within each cluster, the enumeration is over a small number of tracks. For worst-case dense clusters, consider adding an approximate JPDA variant (e.g., cheap JPDA or suboptimal JPDA) in a future iteration.

**[Trade-off] Per-hypothesis track state storage in MHT.** Storing full filter state (state vector + covariance) per hypothesis per track is memory-intensive. For 100 hypotheses and 50 tracks, this is 100 * 50 = 5000 state/covariance pairs. Mitigation: use shared state for hypotheses that share the same track history (copy-on-write semantics via `Arc`). This deduplicates storage for tracks that are not involved in the ambiguous association.

**[Trade-off] Hungarian remains the default.** New users will get Hungarian unless they explicitly opt into JPDA or MHT. This avoids surprising behavior changes but means users in cluttered scenarios must know to switch. Documentation and examples should guide the choice.

**[Risk] Numerical stability of log-likelihood accumulation in MHT.** Cumulative log-likelihoods can diverge over long sequences, causing overflow in score differences. Mitigation: periodically re-normalize scores relative to the best hypothesis.

## Open Questions

- Should JPDA support a configurable maximum number of joint events before falling back to approximate computation?
- Should MHT support track birth and death within the hypothesis tree, or handle these outside the tree?
- Should we implement Murty's algorithm for optimal k-best assignment enumeration, or is sorted truncation sufficient?
- Should the `AssociationStrategy` be changeable at runtime (e.g., switch from Hungarian to JPDA mid-scenario)?
- What is the right default for k-best? Literature uses 50-200 depending on scenario complexity.
