# Design

## Context

See `proposal.md` for motivation. The existing OpenSpec CI job uses Node 22,
installs `@fission-ai/openspec@latest`, then runs live, archived, and metadata
validation. Local 1.13.2 passes all three gates; npm publishes that version with
Node >=20.19.0 support.

## Goals / Non-Goals

Make the installed CI validator reproducible and visible without changing product
requirements, archive policy, validation strictness, or generated agent files.

## Decisions

- Set `OPENSPEC_VERSION: '1.13.2'` on the existing CI job, install that exact
  package, print its reported version, and fail if it differs. Retain Node 22.
  A floating tag cannot provide the same reproducibility.
- Keep the three validation steps unchanged. A pin is not permission to remove
  metadata fields, shorten requirements, or skip archives just to get green CI.
- Document the same exact install and the three commands for contributors.
  No new runtime script, package manifest, or automatic global-tool update is
  needed for this small tooling correction.
- Mark this tracking change `skip_specs: true`; no capability behavior changes.

## Risks / Trade-offs

- A pin will age: upgrade it in a reviewed PR with all strict gates exercised.
- Local global installations can differ: show the required version and command
  in contributor instructions; validate a fresh published installation on Node 22.
- Historical CI versions are unknown: describe differing diagnostics as observed,
  not proof that a specific OpenSpec release caused them.

## Migration Plan

Merge this focused CI repair only after checks pass, then update the dependency
PR branches and require fresh checks before their merges. Revert the focused
change if needed; never bypass validation. Bridge work remains on its own branch.
