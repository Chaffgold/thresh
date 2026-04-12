## Context

thresh is a Cargo workspace with 12 crates (thresh-core, thresh-filter, thresh-association, thresh-fusion, thresh-tracker, thresh-inference, thresh-bridge, thresh-synth, thresh-eval, thresh-data, thresh, thresh-py). Currently, the only way to use thresh is to clone the repository and depend on path-based crate references. Publishing to crates.io would enable standard Rust dependency management (`cargo add thresh-filter`) and automatic docs.rs hosting.

The workspace already uses `[workspace.package]` for edition and rust-version inheritance, but other metadata fields (description, license, keywords, categories) are incomplete or missing. Some crates have `publish = false` set during early development. The inter-crate dependency graph defines a strict topological ordering that must be respected during publishing.

## Goals / Non-Goals

**Goals:**
- Complete `[package]` metadata for all publishable crates
- Maximize use of `[workspace.package]` inheritance for shared fields
- Configure `cargo-release` for coordinated version management
- Create a GitHub Actions workflow for automated publishing in dependency order
- Add a dry-run CI check on PRs to catch publishability issues early
- Document the publish process for maintainers

**Non-Goals:**
- Publishing pre-built binaries (existing release workflow)
- Publishing to alternative registries
- Automatic changelog generation
- Publishing Python wheels (PyPI is a separate concern)
- Crate-level semver independence (all crates share the workspace version)

## Decisions

### 1. All crates share a single workspace version

**Decision:** Use a single version number for all workspace crates, managed via `[workspace.package] version = "0.3.0"` and inherited by all members via `version.workspace = true`.

**Rationale:** Unified versioning simplifies the release process and avoids confusion about which crate versions are compatible. Since the crates are tightly coupled (thresh-tracker depends on thresh-filter which depends on thresh-core), independent versioning would create a combinatorial compatibility matrix. Workspace-level versioning means `cargo add thresh-filter@0.3` and `cargo add thresh-tracker@0.3` are guaranteed compatible.

### 2. Workspace-level metadata inheritance

**Decision:** Set the following in `[workspace.package]`:
- `version`: shared across all crates
- `edition`: "2021"
- `rust-version`: minimum supported Rust version
- `license`: "MIT OR Apache-2.0"
- `repository`: GitHub URL
- `authors`: project authors

Each crate specifies only: `description` (unique per crate), `keywords` (crate-specific), `categories` (crate-specific), `readme` (crate-specific if present).

**Rationale:** Minimizes duplication and ensures consistency. Crate-specific fields like `description` and `keywords` must differ because they describe different capabilities.

### 3. Topological publish order

**Decision:** Publish crates in this order, derived from the dependency graph:

1. thresh-core
2. thresh-filter
3. thresh-association
4. thresh-fusion
5. thresh-tracker
6. thresh-inference
7. thresh-synth
8. thresh-eval
9. thresh-bridge
10. thresh-data
11. thresh (umbrella)
12. thresh-py

**Rationale:** Each crate must be available on the crates.io index before crates that depend on it are published. This order satisfies all `[dependencies]` constraints.

### 4. `cargo-release` configuration

**Decision:** Add a `release.toml` at the workspace root with:

```toml
[workspace]
shared-version = "workspace"
consolidate-commits = true
push = true
publish = true
tag = true
tag-prefix = "v"

[[package]]
name = "thresh-py"
publish = false  # published via maturin, not cargo

[[package]]
name = "thresh-bridge"
publish = false  # requires PyO3, not suitable for crates.io default build
```

**Rationale:** `cargo-release` handles version bumping, git tagging, and publish orchestration. The `consolidate-commits` option creates a single version-bump commit. thresh-py is excluded because it is a Python extension module published via maturin to PyPI. thresh-bridge may be excluded if its PyO3 dependency creates publishability issues.

### 5. GitHub Actions publish workflow

**Decision:** Create `.github/workflows/publish.yml` triggered by `push` to tags matching `v*`. The workflow:
1. Checks out the repository
2. Installs Rust stable toolchain
3. Runs `cargo publish --dry-run` for each crate in order (validation pass)
4. Publishes each crate in topological order with 45-second delays between publishes for index propagation
5. Uses the `CARGO_REGISTRY_TOKEN` secret for authentication

**Rationale:** Tag-triggered publishing matches the existing release workflow pattern. The dry-run pass catches issues before any crate is published, avoiding partial publishes. The 45-second delay between publishes accounts for crates.io index propagation latency (the index is typically updated within 30 seconds).

### 6. Dry-run CI check on PRs

**Decision:** Add a job to the existing CI workflow that runs `cargo publish --dry-run -p <crate>` for each publishable crate on every PR that modifies Cargo.toml files.

**Rationale:** Catches metadata issues, missing files (README), and dependency specification errors before they reach the main branch. Only runs on Cargo.toml changes to avoid unnecessary CI time.

### 7. Feature-gated crates and publishability

**Decision:** Crates with heavy optional dependencies (thresh-inference with `onnx`, thresh-bridge with `stonesoup`) are published with those features disabled by default. Users who need ORT or PyO3 enable the features in their Cargo.toml.

**Rationale:** crates.io builds must succeed without native libraries (ORT, Python). Default features must be buildable with just `cargo add <crate>`. Optional features are documented in each crate's README.

## Risks / Trade-offs

**[Risk] Name squatting.** The `thresh-*` namespace may already be taken on crates.io. Mitigation: check name availability for all 12 crate names before starting the metadata work.

**[Risk] Partial publish failure.** If crate N fails to publish, crates 1 through N-1 are already published and cannot be unpublished (only yanked). Mitigation: the dry-run pass catches most issues. If a partial publish does occur, yank the incomplete set and retry after fixing the issue.

**[Trade-off] Unified versioning vs semver independence.** All crates share a version, so a patch fix in thresh-core bumps the version for all crates even if they are unchanged. This is standard for tightly-coupled workspace crates (e.g., tokio, bevy) and avoids compatibility confusion.

**[Trade-off] Excluding thresh-bridge from crates.io.** The PyO3 dependency makes thresh-bridge difficult to publish (requires Python headers at build time). Excluding it means users must clone the repo to use the Stone Soup bridge. Acceptable because JPDA/MHT native implementations reduce the need for the bridge.

**[Risk] Documentation builds on docs.rs.** docs.rs builds with default features only. Crates with feature-gated modules will have incomplete documentation on docs.rs unless `[package.metadata.docs.rs]` is configured with `all-features = true` or specific feature lists. Mitigation: add docs.rs metadata to each crate's Cargo.toml.

## Open Questions

- Should thresh-bridge be published with `publish = false` or published with a note that it requires Python?
- Should we reserve the crate names on crates.io early (with a placeholder publish) to prevent squatting?
- What is the minimum supported Rust version (MSRV) we should commit to for crates.io users?
- Should we use `cargo-semver-checks` in CI to validate semver compliance between versions?
- Should the publish workflow support publishing a subset of crates (e.g., only those that changed)?
