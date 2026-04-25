# JPDA and MHT Data Association

## What

Add Joint Probabilistic Data Association (JPDA) and Multi-Hypothesis Tracking (MHT) to thresh-association as alternatives to the current single-assignment Hungarian algorithm. JPDA computes soft association probabilities across all track-detection pairs. MHT maintains a hypothesis tree with deferred hard decisions. Both integrate into the tracker via a new `AssociationStrategy` enum.

## Why

The current Hungarian algorithm assumes each detection maps to at most one track and makes hard (irrevocable) assignments at every timestep. In dense or cluttered environments with closely-spaced targets, crossing tracks, or high false alarm rates, this single-assignment approach breaks down. JPDA handles ambiguity by updating each track with a probability-weighted combination of all feasible detections, preserving uncertainty about the correct association. MHT handles ambiguity by deferring hard decisions, maintaining multiple association hypotheses over a sliding window and pruning unlikely branches. These are the two standard alternatives to Hungarian in production tracking systems and are currently only available in thresh via the Stone Soup PyO3 bridge, which adds a Python runtime dependency.

## How

- **JPDA:** Compute association probabilities for all track-detection pairs within the existing gating infrastructure. For each track, compute a probability-weighted innovation from all gated detections (including a missed-detection hypothesis). Update the track state and covariance using the combined innovation (merged measurement approach).
- **MHT:** Maintain a hypothesis tree where each node represents a global association hypothesis at one timestep. At each scan, expand the tree with all feasible association events. Prune via N-scan pruning (collapse hypotheses that agree on associations older than N scans) and k-best pruning (retain only the top k global hypotheses by cumulative likelihood). Extract the best current hypothesis for track state output.
- Add `AssociationStrategy` enum (`Hungarian`, `Jpda`, `Mht`) to `MultiObjectTracker` configuration, with dispatch in the `step()` method.

## Out of scope

- Probabilistic MHT (PMHT) -- a different algorithm despite the similar name
- Track-oriented MHT (only hypothesis-oriented MHT is implemented)
- GPU acceleration of the combinatorial hypothesis generation
- Learned association (GNN-based matching)
- Integration with the Stone Soup bridge (the native implementations replace the need for it)

## Affected crates

- thresh-association: JPDA probability computation, JPDA merged-measurement update, MHT hypothesis tree, N-scan pruning, k-best pruning, `AssociationStrategy` enum
- thresh-tracker: JPDA/MHT-aware step dispatch in `MultiObjectTracker`, configuration for strategy selection
