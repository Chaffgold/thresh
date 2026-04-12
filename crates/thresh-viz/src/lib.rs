//! Track visualization dashboard for the thresh tracking framework.
//!
//! The data layer ([`recording`]) works without any GUI dependencies.
//! Enable the `gui` feature to get the interactive egui application.

pub mod recording;

#[cfg(feature = "gui")]
pub mod app;
