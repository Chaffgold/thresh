## Capability: crates.io Publishing Workflow

### Overview

A fully automated workflow for publishing all thresh workspace crates to crates.io in correct dependency order, with metadata validation, dry-run verification, and CI integration.

## ADDED Requirements

### Requirement: Complete crate metadata

All workspace crates intended for publication MUST have complete `[package]` metadata including `description`, `license`, `repository`, `documentation`, `readme`, `keywords`, and `categories` fields.

#### Scenario: Metadata validation in CI

**WHEN** a pull request is opened that modifies any Cargo.toml file

**THEN** a CI check validates that all publishable crates have complete metadata by running `cargo publish --dry-run` for each crate

**SHALL** fail the CI check if any required metadata field is missing or if `cargo publish --dry-run` reports errors

### Requirement: Topological publish ordering

The publishing workflow MUST publish crates in dependency order so that each crate's dependencies are already available on crates.io before it is published.

#### Scenario: Publishing the full workspace

**WHEN** a version tag (e.g., `v0.3.0`) is pushed to the repository

**THEN** the GitHub Actions workflow publishes crates in topological order: thresh-core, thresh-filter, thresh-association, thresh-fusion, thresh-tracker, thresh-inference, thresh-synth, thresh-eval, thresh-bridge, thresh-data, thresh, thresh-py

**SHALL** wait for crates.io index propagation between publishes (at minimum 30 seconds) and abort the remaining sequence if any publish fails

### Requirement: Workspace metadata inheritance

The workspace MUST use `[workspace.package]` for shared metadata fields to ensure consistency across crates and reduce duplication.

#### Scenario: Shared fields inherited by member crates

**WHEN** a new crate is added to the workspace

**THEN** it inherits `license`, `repository`, `authors`, `edition`, and `rust-version` from `[workspace.package]`

**SHALL** only need to specify crate-specific fields (`description`, `keywords`, `categories`) in its own Cargo.toml
