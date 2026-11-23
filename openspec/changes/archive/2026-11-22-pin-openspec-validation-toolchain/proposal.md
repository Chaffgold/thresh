# Pin OpenSpec Validation Toolchain

## Why

The dependency PRs #162 and #163 fail existing OpenSpec checks while local
OpenSpec 1.13.2 passes. CI installs a floating `@latest` and does not record its
validator version, making failures difficult to reproduce.

## What Changes

- Pin the CI OpenSpec installation to the verified 1.13.2 baseline and print/check
  the installed version before validation.
- Document the matching local installation and all existing validation commands.
- Keep live-artifact strict validation, archived-task validation, and active
  metadata checks intact; do not rewrite existing specs to accommodate drift.
- Validate using an isolated published package installation and GitHub checks
  before updating/merging the dependency PRs.

## Capabilities

### New Capabilities

None. This is tooling and contributor documentation only (`skip_specs: true`).

### Modified Capabilities

None. No product requirements or strict validation gates change.

## Impact

Only the CI OpenSpec job, contributor instructions, and this tracking change.
No Rust/Python dependency or runtime changes, generated skill refresh, bridge
implementation, provider data, model acceptance, or publication are included.
Historical CI logs do not identify the installed OpenSpec versions; the exact
cause of their differing diagnostics must not be stated as proven version drift.
