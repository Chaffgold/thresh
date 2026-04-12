## 1. Metadata Audit

- [ ] 1.1 Check crates.io name availability for all 12 crate names: thresh-core, thresh-filter, thresh-association, thresh-fusion, thresh-tracker, thresh-inference, thresh-bridge, thresh-synth, thresh-eval, thresh-data, thresh, thresh-py.
- [ ] 1.2 Verify current `[package]` metadata completeness for each crate: identify missing fields (description, license, repository, documentation, readme, keywords, categories).
- [ ] 1.3 Draft unique `description` strings for each crate (concise, under 200 chars, explains the crate's purpose).
- [ ] 1.4 Select appropriate `keywords` (max 5 per crate) and `categories` from the crates.io taxonomy for each crate.

## 2. Workspace Metadata Inheritance

- [ ] 2.1 Add or update `[workspace.package]` in root Cargo.toml with shared fields: `version = "0.3.0"`, `edition = "2021"`, `rust-version`, `license = "MIT OR Apache-2.0"`, `repository`, `authors`.
- [ ] 2.2 Update each member Cargo.toml to use `version.workspace = true`, `edition.workspace = true`, `license.workspace = true`, `repository.workspace = true`, `authors.workspace = true`.
- [ ] 2.3 Add crate-specific fields to each member Cargo.toml: `description`, `keywords`, `categories`, `readme` (if the crate has its own README).
- [ ] 2.4 Verify workspace inheritance works: `cargo metadata --format-version=1` shows correct resolved metadata for all crates.

## 3. Publish Flags

- [ ] 3.1 Set `publish = true` (or remove `publish = false`) on all crates intended for crates.io.
- [ ] 3.2 Set `publish = false` on thresh-py (published via maturin to PyPI, not crates.io).
- [ ] 3.3 Decide and set `publish` flag for thresh-bridge (PyO3 dependency makes crates.io publication problematic).

## 4. docs.rs Configuration

- [ ] 4.1 Add `[package.metadata.docs.rs]` to each crate's Cargo.toml specifying features to enable for docs.rs builds (e.g., `all-features = true` or specific feature lists).
- [ ] 4.2 Verify docs build correctly with `cargo doc --workspace --no-deps` under the docs.rs-equivalent feature configuration.

## 5. cargo-release Configuration

- [ ] 5.1 Add `release.toml` at workspace root with `shared-version = "workspace"`, `consolidate-commits = true`, `push = true`, `publish = true`, `tag = true`, `tag-prefix = "v"`.
- [ ] 5.2 Add per-package overrides in `release.toml` for crates excluded from publishing (thresh-py, possibly thresh-bridge).
- [ ] 5.3 Test cargo-release dry run: `cargo release --workspace --dry-run patch` to verify version bumping and publish order.

## 6. GitHub Actions Publish Workflow

- [ ] 6.1 Create `.github/workflows/publish.yml` triggered by `push` to tags matching `v*`.
- [ ] 6.2 Implement validation pass: run `cargo publish --dry-run -p <crate>` for each publishable crate in topological order.
- [ ] 6.3 Implement publish pass: run `cargo publish -p <crate>` for each publishable crate in topological order, with 45-second delays between publishes for index propagation.
- [ ] 6.4 Add `CARGO_REGISTRY_TOKEN` secret usage and document how to set it in the repository settings.
- [ ] 6.5 Add error handling: if any publish fails, annotate the workflow run with which crates succeeded and which failed.

## 7. CI Dry-Run Check

- [ ] 7.1 Add a dry-run job to the existing CI workflow that runs on PRs modifying `**/Cargo.toml` files.
- [ ] 7.2 The job runs `cargo publish --dry-run -p <crate>` for each publishable crate and reports any errors.

## 8. Documentation and Verification

- [ ] 8.1 Run a full dry-run publish sequence locally: `cargo publish --dry-run -p thresh-core && ... && cargo publish --dry-run -p thresh` to verify the complete chain.
- [ ] 8.2 Verify each crate's README exists and is referenced in Cargo.toml (crates.io displays the README on the crate page).
- [ ] 8.3 Add a PUBLISHING.md document (or section in CONTRIBUTING.md) describing the release process for maintainers.
