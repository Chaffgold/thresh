## Context

SonarCloud's community Rust plugin implements cognitive complexity per the Sonar whitepaper (G. Ann Campbell, "Cognitive Complexity: A new way of measuring understandability"). Unlike cyclomatic complexity, which counts linearly independent paths, cognitive complexity penalizes nesting, recursion, and flow-break statements more heavily — a triple-nested loop with `break` and `continue` scores dramatically higher than three sequential loops. The rule enforcing this is `rust:S3776`, and the thresh project quality gate requires each function to score ≤ 15.

Six functions currently exceed this threshold. One — the Hungarian assignment in `thresh-association` — sits at 113, making it a major outlier. The other five are clustered between 16 and 40. We have already applied a proven decomposition pattern to several tracker `step()` implementations in this codebase, extracting named phase helpers (`predict_phase`, `build_cost_matrix_phase`, `update_matched`, `update_lifecycle`) without changing behavior. The same pattern works for most of the flagged functions; the Hungarian case will require more careful structural work because the algorithm's state is more tightly coupled than a sequential tracker pipeline.

## Goals / Non-Goals

Goals:
- Bring all six flagged functions to cognitive complexity ≤ 15 as reported by SonarCloud.
- Preserve exact functional behavior — these are diff-only refactors, not rewrites.
- Maintain or improve test coverage over the refactored code; add regression tests where the refactor exposes edge cases that weren't previously pinned.
- Leave the top-level functions readable as a sequence of named phases, so reviewers can follow the logic without drilling into every helper.

Non-Goals:
- Don't rewrite algorithms or change any public APIs. The Hungarian implementation remains Jonker-Volgenant; the RK4 propagator remains RK4.
- Don't chase sub-15 scores obsessively. Landing at 14 or 15 is fine — the goal is to clear the quality gate, not to golf the complexity metric.
- Don't add new features or expand scope. Any feature work motivated by what we learn during the refactor goes into a separate change proposal.
- Don't touch functions that aren't flagged, even if they look similar to flagged ones.

## Decisions

### 1. Decomposition pattern: phase helpers with descriptive names

**Decision:** Extract phase helpers using descriptive names that match what the phase does (e.g., `predict_phase`, `build_cost_matrix_phase`, `update_matched`, `update_lifecycle`). The top-level function becomes a short sequence of calls to these helpers plus a small amount of glue.

**Rationale:** This is the pattern we already used successfully on the great-circle and recentered-ENU tracker `step()` functions. It has three concrete benefits: (1) the top-level function reads like a table of contents, (2) each helper is independently testable when its inputs and outputs are well-defined, and (3) reviewers can ignore helpers whose names aren't relevant to the change they're looking at. It also keeps the diff localized — we're moving lines into functions, not rewriting them.

**Alternatives considered:**
- *Inline closures*: Would reduce complexity scores but not improve reviewability; closures with captured state are harder to reason about than functions with explicit parameters.
- *Trait-based dispatch*: Adds indirection (two files to read instead of one, virtual dispatch in hot paths) without a corresponding benefit for functions that are only called from one place.

### 2. Hungarian algorithm: full structural refactor with comparison harness

**Decision:** Split the Jonker-Volgenant implementation into three phases — cost-matrix row/column reduction, augmenting-path search, and label/slack updates — each as a separate function with its own unit tests. During the transition, keep both the old and new implementations behind a `#[cfg(test)]` comparison harness that runs both against random cost matrices and asserts identical assignment outputs for a large number of iterations before the old implementation is removed.

**Rationale:** At cognitive complexity 113, the current function is effectively unreviewable — any change to the matching logic is a ticking bomb. A phase-based decomposition is not optional here; it's the only way to make the algorithm maintainable. The comparison harness is essential because the Hungarian algorithm has subtle failure modes (non-unique optimal assignments, numerical ties, degenerate inputs) that aren't always caught by the existing association tests. Running both implementations in parallel on random inputs is the cheapest way to gain confidence that the refactor is behavior-preserving.

**Alternatives considered:**
- *Swap in a crate implementation (`pathfinding`, `lapjv`)*: Would eliminate the complexity entirely, but changes the dependency footprint and may introduce behavioral differences we don't want to chase down. Out of scope for a maintainability change.
- *Refactor in place without a harness*: Risks silently introducing wrong-assignment bugs that only manifest in production scenarios. Not worth the shortcut.

### 3. Test complexity: extract scenario helpers, not monolithic loops

**Decision:** For the stereographic long-traverse test (complexity 21), extract per-step helpers that mirror the structure we used in the recentered-ENU long-traverse test: a measurement-generation helper, a step helper, and a final-error computation helper. The test itself should read as a scenario script, not a monolith.

**Rationale:** The test already passes — the point of refactoring it is so that future changes to the stereographic tracker don't have to wade through a 100-line test function to understand what's being asserted. Keeping tests simple is just as important as keeping production code simple; complex tests are a leading indicator of regressions that slip through because the reviewer didn't fully parse the assertion.

## Risks / Trade-offs

**[Risk] Hungarian refactor silently changes the association math.** At complexity 113 there is real surface area for introducing a subtle bug — an off-by-one in the augmenting path, a missed label update, a wrong sign on a slack computation. Mitigation: the `#[cfg(test)]` comparison harness described in Decision 2 runs both implementations against randomized cost matrices and asserts equality. The old implementation stays in the tree until the harness has accumulated enough iterations (≥10,000) to give confidence in equivalence. We also add unit tests for each extracted phase helper so that any future regression points at the right location.

**[Trade-off] More function definitions means more small allocations and potential indirection overhead.** In theory, more functions mean more stack frames and more call sites. In practice, all the extracted helpers are crate-local and inlineable, so the optimizer should collapse them to identical codegen. Mitigation: if any benchmark regresses, add `#[inline]` to the affected helpers or verify with `cargo asm`.

**[Risk] Refactoring test helpers can silently change what the test asserts.** Pulling test code into helper functions risks losing an assertion or weakening a check without anyone noticing. Mitigation: run the test suite before and after each refactor and confirm identical pass/fail behavior. Push assertions down into the extracted helpers so that a failure message points at the specific phase, not a generic line in a 100-line test body.

**[Trade-off] The comparison harness for the Hungarian algorithm lives in the tree until we retire the old implementation.** This temporarily doubles the amount of association code. Mitigation: accept the short-term duplication, schedule the retirement in task 1.6, and don't leave the harness behind once confidence is established.

## Open Questions

- Should we add a clippy lint or a CI job that prevents new `rust:S3776` violations from creeping in between SonarCloud runs? SonarCloud runs on PR builds, but a local pre-push signal would be cheaper.
- Is 15 the right threshold for this project, or should we tune it based on what SonarCloud flags on comparable mature Rust projects? 15 is the Sonar default; some projects relax it to 20 for Rust specifically because pattern-matching and iterator chains can inflate the score.
- Should the decomposition pattern (phase helpers with descriptive names) be documented in `CLAUDE.md` or a short style guide so that new contributors pick it up without needing to reverse-engineer it from the tracker refactors?
