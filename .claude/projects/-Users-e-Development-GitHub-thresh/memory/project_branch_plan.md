---
name: Current branch and PR plan
description: Active worktree layout and PR sequence for finishing transformer-fusion-tracker then moving to test-data-pipeline
type: project
---

Active worktree layout as of 2026-04-09:

1. `thresh-worktrees/finish-transformer-fusion` (branch: `feature/finish-transformer-fusion`) — finish 8 remaining transformer-fusion-tracker tasks → PR to develop
2. `thresh/` main repo (branch: `feature/test-data-pipeline`) — coordinate transforms + data pipeline work (parked until #1 is merged)

**Why:** Transformer-fusion-tracker is nearly complete (8 tasks left). Clean it up first so develop is current before starting new feature work.

**How to apply:** Complete finish-transformer-fusion → PR to develop → then switch to test-data-pipeline for coordinate transforms (including new ECI tasks 2.5–2.8).
