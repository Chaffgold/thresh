//! Interactive egui visualization application.
//!
//! This module is only available when the `gui` feature is enabled.

use eframe::egui;
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotPoints, Points};

use crate::recording::Recording;

/// Deterministic hash-based color for a track ID.
fn id_to_color(id: u64) -> egui::Color32 {
    let hue = ((id.wrapping_mul(2_654_435_761)) % 360) as f32;
    egui::ecolor::Hsva::new(hue / 360.0, 0.7, 0.9, 1.0).into()
}

/// Main visualization application state.
pub struct ThreshVizApp {
    /// Loaded recording (if any).
    recording: Option<Recording>,
    /// Index of the currently displayed frame.
    current_frame: usize,
    /// Whether playback is active.
    playing: bool,
    /// Playback speed multiplier (1.0 = realtime).
    playback_speed: f32,
    /// Show raw detection markers.
    show_detections: bool,
    /// Show ground-truth markers.
    show_ground_truth: bool,
    /// Show track trail lines.
    show_trails: bool,
    /// Number of past frames to include in trails.
    trail_length: usize,
    /// Currently selected track ID (for highlighting).
    selected_track: Option<u64>,
    /// Accumulated time since last frame advance (seconds).
    accumulated_dt: f64,
}

impl ThreshVizApp {
    /// Create a new application, optionally with a pre-loaded recording.
    pub fn new(recording: Option<Recording>) -> Self {
        Self {
            recording,
            current_frame: 0,
            playing: false,
            playback_speed: 1.0,
            show_detections: true,
            show_ground_truth: true,
            show_trails: true,
            trail_length: 20,
            selected_track: None,
            accumulated_dt: 0.0,
        }
    }

    /// Total number of frames in the loaded recording.
    fn total_frames(&self) -> usize {
        self.recording
            .as_ref()
            .map(|r| r.frame_count())
            .unwrap_or(0)
    }

    /// Advance to the next frame, wrapping at the end.
    fn advance_frame(&mut self) {
        let total = self.total_frames();
        if total > 0 {
            self.current_frame = (self.current_frame + 1) % total;
        }
    }

    /// Jump to the last frame.
    fn go_to_last_frame(&mut self) {
        let total = self.total_frames();
        if total > 0 {
            self.current_frame = total - 1;
        }
    }

    /// Collect trail positions for a track across recent frames.
    fn get_trail(&self, track_id: u64, recording: &Recording) -> Vec<[f64; 2]> {
        let start = self.current_frame.saturating_sub(self.trail_length);
        let mut trail = Vec::new();
        for i in start..=self.current_frame {
            if let Some(t) = recording.frames[i].tracks.iter().find(|t| t.id == track_id) {
                trail.push([t.position[0], t.position[1]]);
            }
        }
        trail
    }

    /// Render the playback controls and display options.
    fn render_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Playback");
        ui.horizontal(|ui| {
            if ui.button("\u{23EE}").clicked() {
                self.current_frame = 0;
            }
            if ui.button("\u{25C0}").clicked() {
                self.current_frame = self.current_frame.saturating_sub(1);
            }
            let play_label = if self.playing { "\u{23F8}" } else { "\u{25B6}" };
            if ui.button(play_label).clicked() {
                self.playing = !self.playing;
            }
            if ui.button("\u{23ED}").clicked() {
                self.go_to_last_frame();
            }
        });

        if let Some(rec) = &self.recording {
            let max = rec.frame_count().saturating_sub(1);
            ui.add(egui::Slider::new(&mut self.current_frame, 0..=max).text("Frame"));
            if let Some(frame) = rec.frames.get(self.current_frame) {
                ui.label(format!("Time: {:.2}s", frame.timestamp));
            }
        }

        ui.separator();
        ui.heading("Display");
        ui.checkbox(&mut self.show_detections, "Show detections");
        ui.checkbox(&mut self.show_ground_truth, "Show ground truth");
        ui.checkbox(&mut self.show_trails, "Show track trails");
        ui.add(egui::Slider::new(&mut self.trail_length, 1..=100).text("Trail length"));
        ui.add(
            egui::Slider::new(&mut self.playback_speed, 0.1..=5.0)
                .text("Speed")
                .logarithmic(true),
        );
    }

    /// Render summary metrics for the current frame.
    fn render_metrics(&self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Metrics");
        if let Some(rec) = &self.recording
            && let Some(frame) = rec.frames.get(self.current_frame)
        {
            ui.label(format!(
                "Tracks: {} confirmed, {} tentative",
                frame.confirmed_track_count(),
                frame.tentative_track_count(),
            ));
            ui.label(format!("Detections: {}", frame.detections.len()));
            ui.label(format!("Ground truth: {}", frame.ground_truth.len()));
        }
    }

    /// Render the scrollable track list with selection.
    fn render_track_list(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Tracks");

        // Clone the data we need to avoid borrow issues.
        let tracks: Vec<_> = self
            .recording
            .as_ref()
            .and_then(|rec| rec.frames.get(self.current_frame))
            .map(|frame| {
                frame
                    .tracks
                    .iter()
                    .map(|t| (t.id, t.position, t.is_confirmed, t.class_label.clone()))
                    .collect()
            })
            .unwrap_or_default();

        egui::ScrollArea::vertical()
            .max_height(200.0)
            .show(ui, |ui| {
                for (id, pos, confirmed, class) in &tracks {
                    let selected = self.selected_track == Some(*id);
                    let status = if *confirmed { "\u{2713}" } else { "?" };
                    let class_str = class
                        .as_ref()
                        .map(|c| format!(" ({c})"))
                        .unwrap_or_default();
                    let label =
                        format!("{status} ID {id} [{:.0}, {:.0}]{class_str}", pos[0], pos[1]);
                    if ui.selectable_label(selected, label).clicked() {
                        self.selected_track = if selected { None } else { Some(*id) };
                    }
                }
            });
    }

    /// Render the main 2D plot.
    fn render_plot(&self, ui: &mut egui::Ui) {
        let Some(recording) = &self.recording else {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() / 3.0);
                ui.label("No recording loaded. Use --recording <file.json>");
            });
            return;
        };

        let Some(frame) = recording.frames.get(self.current_frame) else {
            ui.label("Frame index out of range.");
            return;
        };

        Plot::new("track_plot")
            .data_aspect(1.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                // Confirmed tracks as colored circles.
                for track in &frame.tracks {
                    let color = id_to_color(track.id);
                    let radius = if self.selected_track == Some(track.id) {
                        9.0
                    } else {
                        6.0
                    };
                    plot_ui.points(
                        Points::new(vec![[track.position[0], track.position[1]]])
                            .radius(radius)
                            .color(color)
                            .name(format!("Track {}", track.id)),
                    );
                }

                // Track trails (previous N frames).
                if self.show_trails {
                    for track in &frame.tracks {
                        let trail = self.get_trail(track.id, recording);
                        if trail.len() >= 2 {
                            plot_ui.line(
                                Line::new(PlotPoints::new(trail))
                                    .color(id_to_color(track.id))
                                    .width(1.5),
                            );
                        }
                    }
                }

                // Detections as gray diamonds.
                if self.show_detections {
                    let det_points: Vec<[f64; 2]> = frame
                        .detections
                        .iter()
                        .map(|d| [d.position[0], d.position[1]])
                        .collect();
                    if !det_points.is_empty() {
                        plot_ui.points(
                            Points::new(det_points)
                                .radius(4.0)
                                .color(egui::Color32::GRAY)
                                .shape(MarkerShape::Diamond)
                                .name("Detections"),
                        );
                    }
                }

                // Ground truth as green crosses.
                if self.show_ground_truth {
                    let gt_points: Vec<[f64; 2]> = frame
                        .ground_truth
                        .iter()
                        .map(|g| [g.position[0], g.position[1]])
                        .collect();
                    if !gt_points.is_empty() {
                        plot_ui.points(
                            Points::new(gt_points)
                                .radius(5.0)
                                .color(egui::Color32::GREEN)
                                .shape(MarkerShape::Cross)
                                .name("Ground Truth"),
                        );
                    }
                }
            });
    }
}

impl eframe::App for ThreshVizApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Left panel: controls, metrics, track list.
        egui::SidePanel::left("controls")
            .min_width(220.0)
            .show(ctx, |ui| {
                self.render_controls(ui);
                self.render_metrics(ui);
                self.render_track_list(ui);
            });

        // Central panel: 2D plot.
        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_plot(ui);
        });

        // Auto-advance when playing.
        if self.playing {
            ctx.request_repaint();
            let dt = ctx.input(|i| i.stable_dt) as f64;
            self.accumulated_dt += dt * self.playback_speed as f64;

            // Compute the inter-frame interval from the recording.
            let interval = self
                .recording
                .as_ref()
                .and_then(|rec| {
                    if rec.frames.len() >= 2 {
                        Some(rec.duration() / (rec.frames.len() - 1) as f64)
                    } else {
                        None
                    }
                })
                .unwrap_or(1.0 / 30.0);

            while self.accumulated_dt >= interval {
                self.accumulated_dt -= interval;
                self.advance_frame();
            }
        }
    }
}
