# crates.io Publishing Workflow

## What

Polish crate metadata across all workspace members and add a cargo-release + GitHub Actions workflow for publishing all workspace crates to crates.io in correct dependency order. This includes completing Cargo.toml metadata (descriptions, license, repository, documentation URLs), verifying publish ordering, configuring cargo-release for workspace version management, and creating a CI workflow that automates the publish sequence.

## Why

thresh is currently only usable by cloning the repository. Publishing to crates.io enables the Rust ecosystem to use individual thresh crates as dependencies (`cargo add thresh-filter`, `cargo add thresh-association`, etc.) without pulling the full monorepo. This is a prerequisite for ecosystem adoption and for downstream projects that want to depend on specific thresh capabilities without vendoring. It also enables semver-based dependency management and lets Rust documentation infrastructure (docs.rs) automatically host API docs for each crate.

## How

- Audit and complete `[package]` metadata in every workspace Cargo.toml: `description`, `license`, `repository`, `documentation`, `readme`, `keywords`, `categories`
- Use workspace-level metadata inheritance (`[workspace.package]`) for shared fields (license, repository, authors, edition, rust-version)
- Set `publish = true` (or remove `publish = false`) on all crates intended for publishing; keep `publish = false` on internal-only crates if any
- Verify and document the topological publish order based on inter-crate dependencies
- Add `cargo-release` configuration (`release.toml`) for coordinated version bumping and publish
- Create a GitHub Actions workflow (`.github/workflows/publish.yml`) triggered by version tags that publishes crates in dependency order with appropriate delays between publishes for crates.io index propagation
- Add a dry-run CI job that validates publishability on every PR

## Out of scope

- Publishing pre-built binaries (covered by existing release workflow)
- Publishing to alternative registries (only crates.io)
- Automatic changelog generation (manual for now)
- Publishing Python wheels to PyPI (separate concern)

## Affected crates

- All workspace Cargo.toml files (metadata completion)
- Workspace root Cargo.toml (workspace.package inheritance)
- `.github/workflows/` (new publish workflow)
- `release.toml` (new cargo-release configuration)
