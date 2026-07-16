//! Shared orbital and ballistic force math and state representation.
//!
//! Single source of truth for the two-body/J2 gravitational accelerations,
//! the piecewise-exponential atmosphere with ballistic-coefficient drag, and
//! the closure-based fixed-step RK4 integrator. Both the `thresh-filter`
//! motion models and the `thresh-synth` truth generator delegate here so
//! that prediction and truth use bitwise-identical physics (design
//! Decision 1 of the `orbital-ballistic-filter-models` change). The
//! frame-disciplined [`OrbitalState`] representation type (design
//! Decision 2) lives in [`state`].
//!
//! The `orbit-propagation-fidelity` change adds the high-fidelity tier on
//! top — all additive, nothing above is altered: analytic Sun/Moon
//! ephemerides ([`ephemeris`]), closed-form J3/J4 zonals (in [`gravity`]),
//! truncated EGM96 spherical harmonics ([`egm96`]), Harris-Priester
//! density (in [`atmosphere`]), cannonball SRP with cylindrical eclipse
//! ([`srp`]), lunisolar third-body ([`third_body`]), the composed
//! time-aware force stack ([`force_config`]), and the adaptive
//! Dormand–Prince RK5(4) integrator ([`dormand_prince`]).

pub mod atmosphere;
pub mod dormand_prince;
pub mod egm96;
pub mod ephemeris;
pub mod force_config;
pub mod gravity;
pub mod integrate;
pub mod srp;
pub mod state;
pub mod third_body;

pub use atmosphere::{ATMOSPHERE_TABLE, atmosphere_density, drag_acceleration};
pub use gravity::{GravityModel, j2_acceleration, two_body_acceleration};
pub use integrate::{rk4_stage, rk4_step};
pub use state::{
    ElementError, Frame, OrbitalElements, OrbitalFrame, OrbitalState, cartesian_to_keplerian,
    keplerian_to_cartesian,
};
