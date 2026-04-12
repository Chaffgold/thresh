# Real-Time Streaming Tracker API

## What

Add an async streaming API to thresh-tracker so the tracker can consume live sensor feeds with backpressure, frame dropping, and configurable latency budgets, replacing the need for callers to implement their own buffering and timing logic around the current batch-oriented `step()` interface.

## Why

The current `step(detections, dt)` API requires the caller to batch detections per frame and drive the tracker synchronously. For live radar, ADS-B, or multi-sensor feeds, this means the caller must implement temporal binning (grouping measurements into frames), frame-rate management (deciding when to run prediction vs. update), backpressure handling (what to do when detections arrive faster than the tracker can process), and missed-frame logic. Every integration rebuilds this same scaffolding. A streaming API that handles these concerns inside the tracker makes live integration straightforward.

## How

- Implement a `StreamingTracker` wrapper around `MultiObjectTracker` that owns an async processing loop
- Accept individual measurements via a tokio mpsc channel with configurable buffer depth for backpressure
- Implement temporal binning: accumulate measurements within a configurable frame window, then execute predict+update
- Add frame-rate decimation: when measurements arrive faster than the latency budget allows, drop oldest pending frames and predict forward
- Expose track outputs via a tokio broadcast channel so multiple consumers (logger, display, fusion node) can subscribe
- Gate the tokio dependency behind an optional `streaming` feature flag to keep the sync API dependency-free

## Out of scope

- Network I/O, socket management, or protocol handling (the streaming API operates on local async channels, not network connections)
- Distributed tracking across multiple processes or machines
- GUI, dashboard, or real-time visualization
- Guaranteed real-time scheduling or RTOS integration

## Affected crates

- thresh-tracker: `StreamingTracker` wrapper, temporal binning, frame budget logic, feature-gated tokio dependency
- thresh-core: async-compatible measurement types, timestamp ordering utilities
