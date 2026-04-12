---
name: ECI coordinate system needed
description: ECI (J2000/GCRF) coordinate frame must be added to thresh-core coords — tracked in test-data-pipeline OpenSpec tasks 2.5–2.8, 2.11–2.12
type: project
---

ECI (J2000/GCRF) coordinate system needs to be added to thresh-core alongside existing polar/Cartesian transforms.

**Why:** Required for orbital tracking (SGP4 outputs TEME which is ECI-family), high-fidelity simulation (nyx-space outputs ECI), and ground station observation modeling. Both test-data-pipeline and hifi-sensor-simulation depend on it.

**How to apply:** Tasks 2.5–2.8 and 2.11–2.12 in `openspec/changes/test-data-pipeline/tasks.md` cover the implementation. Work happens on `feature/test-data-pipeline` branch after transformer-fusion-tracker is PR'd.
