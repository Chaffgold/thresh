# Distributed Track-to-Track Fusion — Design

## Context

thresh-fusion currently implements centralized measurement-level fusion (centralized fuser, information filter, covariance intersection) and sensor registration. All existing fusion assumes raw measurements are available at a single node. For distributed multi-site architectures (OTHR networks, coalition environments), raw data cannot be shared — only track-level state estimates. Track-to-track fusion (T2TF) is the missing capability. thresh-core has track types (`TrackState`, `Track`) and thresh-association has Mahalanobis gating, but neither supports cross-site track matching.

## Goals / Non-Goals

**Goals:**
- `TrackExchange` struct in thresh-core for inter-site track state messages
- `T2TAssociator` in thresh-association for track-to-track matching via augmented-state Mahalanobis distance
- `FederatedFusionManager` in thresh-fusion accepting `Vec<Vec<TrackExchange>>` from N sites, performing association and fusion
- Cross-covariance-aware covariance intersection as the primary fusion algorithm
- Asynchronous update handling for tracks arriving at different rates from different sites
- Naive fusion mode (ignoring cross-covariance) for simplicity and as a comparison baseline

**Non-Goals:**
- Network transport, serialization wire formats, or message brokers
- Byzantine fault tolerance or adversarial track detection
- Multi-level fusion hierarchy (only single fusion node)
- Modifying existing centralized fusion algorithms

## Decisions

### TrackExchange type

```rust
pub struct TrackExchange {
    pub track_id: TrackId,
    pub source_id: u32,
    pub state: DVector<f64>,
    pub covariance: DMatrix<f64>,
    pub timestamp: f64,
    pub class: TargetClass,
    pub confidence: f64,
}
```

Lives in `crates/thresh-core/src/track.rs` alongside existing track types. Derives `Serialize`/`Deserialize` so callers can use any serialization format they choose. Does not include process noise or cross-covariance — those are managed internally by the fusion manager.

### Track-to-track association

`T2TAssociator` in `crates/thresh-association/src/t2t.rs`. Uses augmented-state Mahalanobis distance: for two tracks with states `x1`, `x2` and covariances `P1`, `P2`, the test statistic is `(x1 - x2)^T (P1 + P2)^{-1} (x1 - x2)`. When cross-covariance `P12` is available, use the full augmented form: stack states and use the block covariance matrix `[[P1, P12], [P12^T, P2]]`.

Association is framed as a linear assignment problem: build a cost matrix of Mahalanobis distances between all track pairs from different sources, gate at a chi-squared threshold, and solve with the existing Hungarian algorithm. This reuses `hungarian_assignment` from thresh-association.

### Federated fusion manager

`FederatedFusionManager` in `crates/thresh-fusion/src/federated.rs`. Maintains:
- A fused track table (the common operating picture)
- Per-source track histories for temporal alignment
- Optional cross-covariance bookkeeping between source-fused track pairs

Workflow per update cycle:
1. Receive `Vec<TrackExchange>` from one or more sources
2. Temporally align: extrapolate each incoming track to the fusion time using constant-velocity prediction
3. Associate incoming tracks to fused tracks (and to each other if from different sources)
4. Fuse matched pairs via covariance intersection (robust, does not require cross-covariance knowledge) or optimal fusion (if cross-covariance is tracked)
5. Birth new fused tracks from unmatched incoming tracks
6. Coast/delete fused tracks with no updates for configurable duration

### Temporal alignment

Incoming tracks may have different timestamps. Before association, extrapolate each track's state forward to a common fusion time `t_fuse` (the latest timestamp among all incoming tracks in the batch). Use linear prediction: `x(t_fuse) = F(dt) * x(t)` with the same constant-velocity model used by the tracker. Covariance grows: `P(t_fuse) = F * P * F^T + Q(dt)`. This is a first-order approximation; higher-order models are a future extension.

### Covariance intersection

The primary fusion method. For two estimates `(x1, P1)` and `(x2, P2)`:
```
P_fused^{-1} = w * P1^{-1} + (1-w) * P2^{-1}
x_fused = P_fused * (w * P1^{-1} * x1 + (1-w) * P2^{-1} * x2)
```
where `w in [0,1]` is chosen to minimize `det(P_fused)` or `trace(P_fused)`. Reuse the existing `covariance_intersection` module in thresh-fusion, extending it to handle the multi-source case by sequential pairwise fusion.

### Fusion modes

- **Naive**: Treat incoming track covariances as independent. Simple `P_fused^{-1} = P1^{-1} + P2^{-1}` (information filter sum). Fast but overconfident when tracks share common process noise.
- **Covariance intersection**: Conservative, no cross-covariance needed. Default mode.
- **Optimal with cross-covariance**: Bookkeep `P12` matrices. Most accurate but highest memory/compute cost. Opt-in via configuration.

## Risks / Trade-offs

- **Cross-covariance bookkeeping cost**: Storing `P12` for every source-fused track pair is O(N_sources * N_tracks * state_dim^2). For large track counts, this dominates memory. Covariance intersection avoids this but is conservative.
- **Temporal alignment error**: Linear extrapolation is inaccurate for maneuvering targets. Accepting this for now; future work could use IMM-predicted states.
- **Association ambiguity**: When multiple sources track the same target with different track IDs, augmented-state Mahalanobis can fail if covariances are very different. The gating threshold must be tuned per deployment.
- **No network layer**: Callers must implement their own transport. This is intentional (out of scope) but means integration requires more user code.

## Open Questions

1. Should `FederatedFusionManager` own the temporal extrapolation model, or should callers extrapolate tracks before submission?
2. How should track deletion propagate — if one source drops a track, should the fused track coast until all sources drop it?
3. Should we support more than pairwise CI (e.g., batch CI over N sources simultaneously)?
