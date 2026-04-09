//! PyO3 bridge to Stone Soup for advanced tracking algorithms (JPDA, MHT, IMM).
//!
//! All modules are gated behind the `stonesoup` Cargo feature which pulls in
//! `pyo3`.  When the feature is disabled, this crate is intentionally empty
//! so that downstream crates can unconditionally depend on it without requiring
//! a Python installation.

#[cfg(feature = "stonesoup")]
pub mod convert;
#[cfg(feature = "stonesoup")]
pub mod detection;
#[cfg(feature = "stonesoup")]
pub mod error;
#[cfg(feature = "stonesoup")]
pub mod imm;
#[cfg(feature = "stonesoup")]
pub mod jpda;
#[cfg(feature = "stonesoup")]
pub mod mht;
#[cfg(feature = "stonesoup")]
pub mod phd;
