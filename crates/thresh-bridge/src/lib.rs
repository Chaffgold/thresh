//! PyO3 bridge to Stone Soup JPDA and Gaussian Mixture PHD reference algorithms.
//!
//! The bridge provides JPDA and GM-PHD references only; it has no MHT or IMM
//! entry point. Those contracts are native, in the `mht-tracking`
//! (`thresh-association`, `thresh-tracker`) and `imm-estimation`
//! (`thresh-filter`) capabilities, and both have open interim gaps.
//!
//! All modules are gated behind the `stonesoup` Cargo feature which pulls in
//! `pyo3`.  When the feature is disabled, this crate is intentionally empty
//! so that downstream crates can unconditionally depend on it without requiring
//! a Python installation.

#[cfg(feature = "stonesoup")]
pub mod context;
#[cfg(feature = "stonesoup")]
pub mod convert;
#[cfg(feature = "stonesoup")]
mod covariance;
#[cfg(feature = "stonesoup")]
pub mod detection;
#[cfg(feature = "stonesoup")]
pub mod error;
#[cfg(feature = "stonesoup")]
pub mod jpda;
#[cfg(feature = "stonesoup")]
mod observation;
#[cfg(feature = "stonesoup")]
pub mod phd;
#[cfg(feature = "stonesoup")]
mod runtime;
#[cfg(feature = "stonesoup")]
pub mod time;
