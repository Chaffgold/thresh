//! Track visualization dashboard for the thresh tracking framework.
//!
//! The data layer ([`recording`], [`events`]) works without any GUI
//! dependencies. Enable the `gui` feature to get the interactive egui
//! application.

pub mod events;
pub mod recording;

#[cfg(feature = "gui")]
pub mod app;
