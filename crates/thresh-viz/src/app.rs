//! Interactive egui visualization application.
//!
//! This module is only available when the `gui` feature is enabled.

#[cfg(feature = "gui")]
use eframe::egui;

use crate::recording::{Recording, VizFrame};

/// Main visualization application state.
pub struct ThreshVizApp {
    /// Loaded recording (if any).
    recording: Option<Recording>,
    /// Index of the currently displayed frame.
    current_frame: usize,
    /// Whether playback is active.
    playing: bool,
}

impl ThreshVizApp {
    /// Create a new application, optionally with a pre-loaded recording.
    pub fn new(recording: Option<Recording>) -> Self {
        Self {
            recording,
            current_frame: 0,
            playing: false,
        }
    }

    /// Get the current frame, if a recording is loaded and the index is valid.
    pub fn current_viz_frame(&self) -> Option<&VizFrame> {
        self.recording
            .as_ref()
            .and_then(|r| r.frames.get(self.current_frame))
    }

    /// Total number of frames in the loaded recording.
    pub fn total_frames(&self) -> usize {
        self.recording
            .as_ref()
            .map(|r| r.frame_count())
            .unwrap_or(0)
    }

    /// Advance to the next frame (wraps around).
    pub fn step_forward(&mut self) {
        let total = self.total_frames();
        if total > 0 {
            self.current_frame = (self.current_frame + 1) % total;
        }
    }

    /// Step backward one frame (wraps around).
    pub fn step_backward(&mut self) {
        let total = self.total_frames();
        if total > 0 {
            self.current_frame = if self.current_frame == 0 {
                total - 1
            } else {
                self.current_frame - 1
            };
        }
    }

    /// Seek to a specific frame index, clamped to valid range.
    pub fn seek(&mut self, frame: usize) {
        let total = self.total_frames();
        self.current_frame = frame.min(total.saturating_sub(1));
    }

    /// Toggle play/pause.
    pub fn toggle_play(&mut self) {
        self.playing = !self.playing;
    }

    /// Whether playback is currently active.
    pub fn is_playing(&self) -> bool {
        self.playing
    }
}

#[cfg(feature = "gui")]
impl eframe::App for ThreshVizApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("thresh-viz");
            ui.label("Track visualization dashboard — coming soon.");

            if let Some(frame) = self.current_viz_frame() {
                ui.label(format!(
                    "Frame {}/{} — t={:.2}s — {} tracks, {} detections",
                    self.current_frame + 1,
                    self.total_frames(),
                    frame.timestamp,
                    frame.tracks.len(),
                    frame.detections.len(),
                ));
            } else {
                ui.label("No recording loaded.");
            }
        });
    }
}
