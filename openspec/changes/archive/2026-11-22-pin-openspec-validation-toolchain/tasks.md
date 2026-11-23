# Tasks

## 1. Reproducible validation toolchain

- [x] 1.1 Pin the existing CI job to OpenSpec 1.13.2, print/assert the installed version, and preserve all three validation steps; verify an isolated published installation works on Node 22 and a mismatched reported version fails the assertion.
- [x] 1.2 Document the matching exact installation, version check, live/archive/metadata commands, and reviewed-upgrade policy in CONTRIBUTING.md; verify those commands against the same published baseline.

## 2. Integration verification

- [x] 2.1 Run strict live and archived validation plus every active metadata check, review the workflow diff and artifact scope independently, and verify there are no product-spec edits, validation bypasses, generated skill changes, or unrelated dependency changes.
