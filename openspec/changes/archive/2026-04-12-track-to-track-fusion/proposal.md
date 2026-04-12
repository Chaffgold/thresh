# Distributed Track-to-Track Fusion

## What

Add distributed and federated fusion algorithms to thresh-fusion so multiple independent tracker instances (e.g., geographically separated radar sites) can fuse their track-level outputs into a unified common operating picture without sharing raw measurements. This complements the existing centralized measurement-level fusion with a track-level alternative for bandwidth-constrained and organizational-boundary scenarios.

## Why

Centralized fusion, already implemented in thresh-fusion, requires all raw measurements to be available at a single processing node. This assumption breaks in multi-site OTHR networks, coalition environments, and distributed C2 architectures where raw data cannot be shared due to bandwidth, latency, or security constraints. Track-to-track fusion (T2TF) is the standard solution: each site runs its own tracker and exports track state estimates, which a fusion node combines. Without T2TF, thresh cannot support the distributed multi-sensor architectures that are its primary use case.

## How

- Implement cross-covariance estimation for track-to-track fusion, handling the common process noise correlation between independently filtered tracks
- Add track-to-track association using augmented-state Mahalanobis distance to match tracks from different sources that observe the same target
- Build a `FederatedFusionManager` that accepts asynchronous track updates from multiple sources, handles temporal alignment via state extrapolation, and produces fused track outputs
- Support both naive (ignoring cross-covariance) and optimal (with cross-covariance bookkeeping) fusion modes, with covariance intersection as a robust fallback
- Add track exchange types to thresh-core for serializable track state messages between fusion nodes

## Out of scope

- Network transport, serialization protocols, or wire formats (fusion operates on in-memory track structs)
- Byzantine fault tolerance or adversarial track injection detection
- Real-time communication layer or message broker integration
- More than two levels of fusion hierarchy

## Affected crates

- thresh-fusion: T2TF algorithms, cross-covariance estimation, federated fusion manager
- thresh-tracker: multi-site tracker manager that coordinates local tracking with fusion output
- thresh-core: track exchange types, track source metadata, temporal alignment utilities
