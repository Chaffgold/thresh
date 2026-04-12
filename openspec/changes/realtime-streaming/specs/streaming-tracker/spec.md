## Capability: Real-Time Streaming Tracker

### Overview

An async streaming wrapper around MultiObjectTracker accepts individual measurements on a tokio mpsc channel and emits track updates on a broadcast channel, handling temporal framing and latency management.

## ADDED Requirements

### Requirement: Async channel-based streaming tracker

The system MUST provide an async streaming wrapper around MultiObjectTracker that accepts individual measurements on a tokio mpsc channel and emits track updates on a broadcast channel.

#### Scenario: Temporal binning and stale frame dropping under load

**WHEN** measurements arrive faster than the configured frame rate

**THEN** the streaming tracker bins measurements into temporal frames

**SHALL** bin them into temporal frames and drop stale frames that exceed the latency budget
