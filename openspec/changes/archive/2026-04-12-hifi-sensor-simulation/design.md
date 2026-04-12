## Context

The thresh-synth crate currently generates measurements using fixed detection probability and Gaussian noise. Real sensor physics are significantly more complex: radar detection depends on target RCS (which fluctuates and varies with aspect angle), the radar equation (SNR decays as R⁴), and atmospheric attenuation. Aircraft trajectories are constrained by aerodynamics, not just kinematic primitives. This change upgrades the synthetic pipeline to physics-based fidelity while keeping the simple generators as the default.

## Goals / Non-Goals

**Goals:**
- Implement Swerling I-IV RCS fluctuation models in pure Rust
- Implement the radar equation with configurable transmitter parameters
- Provide RCS lookup tables (aspect angle → dBsm) for common target archetypes
- Bridge to JSBSim for realistic 6DOF aircraft/missile trajectories
- Bridge to nyx-space or Orekit for high-fidelity orbital propagation
- Model EO/IR sensor physics (blackbody, atmospheric transmission, NETD)
- Bridge to RadarSimPy or mpar-sim for full radar scene simulation

**Non-Goals:**
- Full electromagnetic simulation in real-time (offline RCS computation only)
- Hardware-in-the-loop simulation
- Electronic warfare / jamming modeling (deferred)
- Synthetic Aperture Radar (SAR) imaging simulation

## Decisions

### 1. Layered fidelity in thresh-synth

**Decision:** Add high-fidelity modules alongside existing simple generators. Users choose fidelity level per scenario.

- **Level 0** (existing): Fixed P_d, Gaussian noise. No deps.
- **Level 1** (new, pure Rust): Swerling + radar equation + RCS tables. No external deps.
- **Level 2** (new, feature-gated): JSBSim trajectories, nyx-space orbits, RadarSimPy scenes. Requires Python/external libs.

### 2. nyx-space over Orekit for orbital propagation

**Decision:** Prefer nyx-space (pure Rust crate) for orbital propagation. Fall back to Orekit (PyO3) only if nyx-space lacks needed force models.

**Rationale:** nyx-space is native Rust, no Python dependency, and supports EGM96, drag (NRLMSISE-00), SRP, third-body perturbations. Orekit has broader feature set but requires JVM via Python bridge — heavy.

### 3. Offline RCS computation, runtime lookup

**Decision:** Use PyPOFacets/Open RCS offline to pre-compute RCS-vs-angle tables, save as JSON. Load tables at runtime in Rust. Never run EM solvers during tracking.

**Rationale:** Physical optics RCS computation is slow (seconds per angle). Tracking needs RCS values at kHz rates. Pre-computed tables bridge this gap. Ship default tables for common archetypes in the repo.

### 4. Open source tool selection

**Decision:** Primary bridges ordered by preference:

| Capability | Primary | Backup | Rationale |
|---|---|---|---|
| Flight dynamics | JSBSim (C++/Python, mature) | — | NASA heritage, F-16/737 models built-in |
| Orbital propagation | nyx-space (Rust) | Orekit (Java/Python) | Native Rust preferred |
| RCS from geometry | PyPOFacets (Python) | Open RCS (Python) | Simpler API, PO method |
| Full radar sim | RadarSimPy (Python/C++) | mpar-sim (Python) | Ray tracing + scene support |

### 5. IR/EO modeling scope

**Decision:** Implement simplified but physically grounded IR model: blackbody target signature → atmospheric attenuation (Beer-Lambert with band-specific coefficients) → sensor NETD → SNR → P_d. Skip full MODTRAN-level atmospheric modeling.

**Rationale:** Full atmospheric radiative transfer (MODTRAN/libRadtran) is overkill for tracking validation. The simplified model captures the dominant effects (range-dependent attenuation, band selection) without adding a heavy dependency.

## Risks / Trade-offs

**[Risk] JSBSim Python bindings stability** → Mitigation: Pin to known-good version, wrap in robust error handling. JSBSim C++ core is very stable (NASA heritage).

**[Risk] nyx-space API churn** → Mitigation: The crate is actively developed. Pin version, abstract behind our own trait so switching to Orekit is straightforward.

**[Risk] RCS table accuracy** → Mitigation: Physical optics is accurate for electrically large targets (aircraft, missiles at GHz frequencies). For small/resonant targets, note accuracy limitations in documentation.

**[Trade-off] Fidelity vs complexity** → Level 1 (Swerling + radar equation) gives 80% of the benefit with zero external deps. Level 2 bridges add realism but require Python ecosystem. Most users will use Level 1.

## Open Questions

- Should we ship pre-computed RCS tables for default target archetypes, or require users to generate their own?
- What JSBSim aircraft models beyond F-16 and 737 are useful? Missile models?
- Should the radar equation support bistatic configurations, or monostatic only for now?
- Is libRadtran worth bridging for atmospheric modeling, or is Beer-Lambert sufficient?

## Reference: Open Source Tools

- [mpar-sim](https://github.com/ShaneFlandermeyer/mpar-sim) — Multi-function Phased Array Radar Simulator (Python)
- [Open RCS](https://github.com/comp-ime-eb-br/open-rcs) — RCS computation from 3D targets (Python)
- [PyPOFacets](https://github.com/gems-uff/pypofacets) — Physical Optics RCS from faceted models (Python)
- [RadarSimPy](https://github.com/radarsimx/radarsimpy) — Radar scene simulator (Python + C++)
- [JSBSim](https://github.com/JSBSim-Team/jsbsim) — Flight dynamics model (C++ with Python bindings)
- [nyx-space](https://github.com/nyx-space/nyx) — Astrodynamics toolkit (Rust)
- [Orekit](https://www.orekit.org/) — Astrodynamics library (Java, Python wrapper)
- [OpenEMS](https://openems.de/) — FDTD EM solver (C++ with Python/MATLAB)
- [Arraytool](https://zinka.wordpress.com/) — Phased array analysis (Python)
