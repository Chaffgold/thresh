## Capability: Track Visualization Dashboard

### Overview

A native desktop visualization application that displays real-time or recorded tracker state on interactive 2D/3D plots, including track trails, measurement scatter, association lines, and MOT metric summaries.

## ADDED Requirements

### Requirement: 2D bird's-eye-view track plot

The system MUST render a 2D plot showing track position trails (color-coded by track ID), measurement scatter points, and association lines connecting measurements to their assigned tracks.

#### Scenario: Displaying a multi-target tracking scenario

**WHEN** the visualization receives TrackSnapshot data containing 10 active tracks over 100 timesteps

**THEN** it renders each track's position history as a color-coded trail on the 2D plot, displays current-timestep measurements as scatter points, and draws association lines from each measurement to its assigned track

**SHALL** update the display at a minimum of 30 frames per second during live streaming and allow pan/zoom interaction without dropping frames

### Requirement: Real-time metric sidebar

The system MUST display a metric sidebar showing current MOT metrics (MOTA, MOTP, track count, confirmed/tentative/lost breakdown) updated at each timestep.

#### Scenario: Live metric updates during streaming

**WHEN** the visualization is connected to a live streaming tracker via broadcast channel

**THEN** the metric sidebar updates at each timestep with current MOTA, MOTP, total track count, and counts of confirmed, tentative, and lost tracks

**SHALL** display metrics with no more than one-timestep latency from the tracker's broadcast

### Requirement: Playback from recorded scenarios

The system MUST support loading a JSON recording of TrackSnapshot data and providing playback controls including play, pause, step forward, step backward, speed control, and seek to a specific timestep.

#### Scenario: Stepping through a recorded scenario

**WHEN** a user loads a JSON recording file and clicks the step-forward button

**THEN** the visualization advances exactly one timestep, updating the plot and metrics to reflect the new timestep's data

**SHALL** allow stepping backward to any previously viewed timestep without reloading the recording file

### Requirement: TrackSnapshot export from tracker

The system MUST provide a `TrackSnapshot` type in thresh-tracker that captures the full tracker state at a single timestep (all tracks with state, covariance, status, associations, and optional metrics) and supports serialization to JSON.

#### Scenario: Recording a tracking session

**WHEN** a user enables snapshot recording on a `MultiObjectTracker` instance

**THEN** the tracker emits a `TrackSnapshot` at each `step()` call, serializable to JSON for later playback

**SHALL** include all track states, covariances, lifecycle statuses, and association decisions in each snapshot
