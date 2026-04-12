# Contributing to thresh

Thank you for your interest in contributing to thresh!

## Getting Started

1. Fork the repository and clone your fork.
2. Create a feature branch from `develop`: `git checkout -b feature/your-feature develop`
3. Install pre-commit hooks: `pre-commit install`
4. Make your changes and ensure all checks pass before pushing.

## Development Workflow

We use **Gitflow**: feature branches are created from `develop`, PRs target `develop`, and releases are cut from `develop` → `main`.

### Build and Test

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo doc --workspace --no-deps
```

CI enforces `RUSTFLAGS=-Dwarnings` — all warnings are compile errors.

### Feature-Gated Code

Several crates have optional features that pull in heavy dependencies (PyO3, ONNX Runtime). When modifying feature-gated code, test with the feature enabled:

```sh
cargo test -p thresh-data --features adsb
cargo test -p thresh-data --features orbital
cargo clippy -p thresh-synth --features rcs-compute --all-targets -- -D warnings
```

### Code Style

- **nalgebra** for all matrix/vector math.
- Keep function cognitive complexity ≤ 15 (SonarCloud `rust:S3776`). When a function grows past this, use the **phase-helper decomposition** pattern documented in `CLAUDE.md`.
- Don't add features, refactor code, or make improvements beyond what was asked.
- Prefer editing existing files over creating new ones.

### OpenSpec

Design specs live in `openspec/`. Validate before committing:

```sh
openspec validate --all --strict --no-interactive
```

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(tracker): add stereographic projection variant
fix(hungarian): correct augmenting-path search for non-greedy matching
docs(readme): add sensor fidelity levels table
refactor(adsb): extract ground-truth interpolation into helper
test(orbital): add ISS propagation accuracy test
```

## Pull Requests

- Keep PRs focused — one logical change per PR.
- Target the `develop` branch.
- Include a test plan in the PR description.
- Ensure all CI checks pass before requesting review.
- Squash-merge is the default merge strategy.

## Reporting Issues

Open an issue at https://github.com/Chaffgold/thresh/issues with:
- A clear description of the problem or feature request.
- Steps to reproduce (for bugs).
- Expected vs actual behaviour.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
