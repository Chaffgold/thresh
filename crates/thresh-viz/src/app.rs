//! Interactive egui visualization application.
//!
//! This module is only available when the `gui` feature is enabled. It
//! supports both offline JSON `Recording` playback and live ingest from
//! a `StreamingTracker` via [`crate::streaming::SnapshotBridge`].

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotPoints, Points};

use thresh_eval::{MotMetrics, MotMetricsBuilder};

use crate::events::diff_snapshots;
use crate::geom::{ellipse_axes, ellipse_polyline};
use crate::recording::{LifecycleEvent, Recording, VizDetection, VizFrame, VizGroundTruth};
use crate::streaming::{ConnectionStatus, SnapshotBridge};

/// Default position-distance threshold (meters) for the per-frame MOT
/// metrics builder when computing live MOTA/MOTP/IDF1.
const METRICS_DISTANCE_THRESHOLD: f64 = 5.0;

/// Maximum lifecycle events retained in the GUI event log.
const EVENT_LOG_CAPACITY: usize = 200;

/// Toast display duration for the screenshot confirmation message.
const TOAST_DURATION_SECS: f64 = 2.0;

/// Centralized hotkey bindings, also used to render the help overlay.
#[derive(Debug, Clone, Copy)]
pub struct KeyBindings {
    pub play_pause: egui::Key,
    pub step_forward: egui::Key,
    pub step_back: egui::Key,
    pub zoom_in: egui::Key,
    pub zoom_out: egui::Key,
    pub screenshot: egui::Key,
    pub toggle_ellipses: egui::Key,
    pub toggle_associations: egui::Key,
    pub toggle_event_log: egui::Key,
    pub toggle_help: egui::Key,
}

impl Default for KeyBindings {
    fn default() -> Self {
        Self {
            play_pause: egui::Key::Space,
            step_forward: egui::Key::ArrowRight,
            step_back: egui::Key::ArrowLeft,
            zoom_in: egui::Key::Plus,
            zoom_out: egui::Key::Minus,
            screenshot: egui::Key::S,
            toggle_ellipses: egui::Key::E,
            toggle_associations: egui::Key::A,
            toggle_event_log: egui::Key::L,
            toggle_help: egui::Key::Questionmark,
        }
    }
}

impl KeyBindings {
    /// Catalog of (key label, action description) used by the help
    /// overlay. Order matches the way users typically discover features.
    pub fn catalog(&self) -> Vec<(String, &'static str)> {
        vec![
            (format!("{:?}", self.play_pause), "Play / pause"),
            (format!("{:?}", self.step_forward), "Step forward"),
            (format!("{:?}", self.step_back), "Step backward"),
            (
                format!("{:?} / {:?}", self.zoom_in, self.zoom_out),
                "Zoom in / out (or scroll wheel)",
            ),
            ("Drag".to_string(), "Pan"),
            (format!("{:?}", self.screenshot), "Screenshot to PNG"),
            (
                format!("{:?}", self.toggle_associations),
                "Toggle association lines",
            ),
            (
                format!("{:?}", self.toggle_ellipses),
                "Toggle covariance ellipses",
            ),
            (
                format!("{:?}", self.toggle_event_log),
                "Toggle lifecycle event log",
            ),
            (
                format!("{:?} or Esc", self.toggle_help),
                "Toggle this help overlay",
            ),
        ]
    }
}

/// Source of visualization frames.
pub enum VizSource {
    /// No data source attached.
    None,
    /// Pre-recorded JSON file.
    Recording(Recording),
    /// Live broadcast bridge from a `StreamingTracker`.
    Live(LiveSource),
}

/// Live snapshot ingest state.
pub struct LiveSource {
    pub bridge: SnapshotBridge,
    /// Frames accumulated from the bridge so far.
    pub frames: Vec<VizFrame>,
    /// Reusable scratch buffer for `drain_into` to avoid per-frame
    /// allocation.
    drain_scratch: Vec<thresh_tracker::streaming::TrackSnapshot>,
}

impl LiveSource {
    pub fn new(bridge: SnapshotBridge) -> Self {
        Self {
            bridge,
            frames: Vec::new(),
            drain_scratch: Vec::with_capacity(64),
        }
    }
}

/// Main visualization application state.
pub struct ThreshVizApp {
    source: VizSource,
    current_frame: usize,
    playing: bool,
    playback_speed: f32,
    accumulated_dt: f64,

    show_detections: bool,
    show_ground_truth: bool,
    show_trails: bool,
    show_associations: bool,
    show_ellipses: bool,
    show_event_log: bool,
    show_help_overlay: bool,
    trail_length: usize,
    selected_track: Option<u64>,

    keys: KeyBindings,

    metrics_builder: MotMetricsBuilder,
    last_metrics: Option<MotMetrics>,
    metrics_frame_index: Option<usize>,

    event_log: VecDeque<(f64, LifecycleEvent)>,

    screenshot_dir: PathBuf,
    toast: Option<(String, Instant)>,
}

impl ThreshVizApp {
    /// Create a new app with a recording (or none).
    pub fn new(recording: Option<Recording>) -> Self {
        let source = match recording {
            Some(r) => VizSource::Recording(r),
            None => VizSource::None,
        };
        Self::from_source(source)
    }

    /// Create a new app from a live streaming bridge.
    pub fn live(bridge: SnapshotBridge) -> Self {
        Self::from_source(VizSource::Live(LiveSource::new(bridge)))
    }

    /// Build with an explicit source.
    pub fn from_source(source: VizSource) -> Self {
        Self {
            source,
            current_frame: 0,
            playing: false,
            playback_speed: 1.0,
            accumulated_dt: 0.0,
            show_detections: true,
            show_ground_truth: true,
            show_trails: true,
            show_associations: true,
            show_ellipses: false,
            show_event_log: true,
            show_help_overlay: false,
            trail_length: 20,
            selected_track: None,
            keys: KeyBindings::default(),
            metrics_builder: MotMetricsBuilder::new(METRICS_DISTANCE_THRESHOLD),
            last_metrics: None,
            metrics_frame_index: None,
            event_log: VecDeque::with_capacity(EVENT_LOG_CAPACITY),
            screenshot_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            toast: None,
        }
    }

    /// Override the screenshot output directory.
    pub fn with_screenshot_dir(mut self, dir: PathBuf) -> Self {
        self.screenshot_dir = dir;
        self
    }

    /// Access the current `KeyBindings` (useful for tests).
    pub fn keys(&self) -> &KeyBindings {
        &self.keys
    }

    /// True if the help overlay is currently visible.
    pub fn help_overlay_open(&self) -> bool {
        self.show_help_overlay
    }

    /// True if covariance ellipses are currently rendered.
    pub fn ellipses_shown(&self) -> bool {
        self.show_ellipses
    }

    /// True if association lines are currently rendered.
    pub fn associations_shown(&self) -> bool {
        self.show_associations
    }

    /// True if the lifecycle event log panel is currently visible.
    pub fn event_log_visible(&self) -> bool {
        self.show_event_log
    }

    /// Total number of frames currently available in the source.
    fn total_frames(&self) -> usize {
        match &self.source {
            VizSource::None => 0,
            VizSource::Recording(r) => r.frame_count(),
            VizSource::Live(l) => l.frames.len(),
        }
    }

    /// Borrow all frames in the current source.
    fn frames(&self) -> &[VizFrame] {
        match &self.source {
            VizSource::None => &[],
            VizSource::Recording(r) => &r.frames,
            VizSource::Live(l) => &l.frames,
        }
    }

    /// Get the currently displayed frame, if any.
    fn current(&self) -> Option<&VizFrame> {
        self.frames().get(self.current_frame)
    }

    /// True if the current source is live (vs recording).
    fn is_live(&self) -> bool {
        matches!(self.source, VizSource::Live(_))
    }

    /// True if ground truth is available in the current source.
    /// Live streams that don't carry GT make MOT metrics show "n/a".
    fn has_ground_truth(&self) -> bool {
        self.frames().iter().any(|f| !f.ground_truth.is_empty())
    }

    /// Drain the bridge if live and recompute metrics for any new frames.
    fn ingest_live(&mut self) {
        let prev_len = self.frames().len();
        if let VizSource::Live(live) = &mut self.source {
            live.drain_scratch.clear();
            live.bridge.drain_into(&mut live.drain_scratch);
            for snap in live.drain_scratch.drain(..) {
                live.frames
                    .push(track_snapshot_to_viz_frame(snap, live.frames.last()));
            }
        }
        let new_len = self.frames().len();
        if new_len > prev_len {
            // For live streams, follow the latest frame.
            if self.is_live() {
                self.current_frame = new_len - 1;
            }
            self.update_metrics_for_new_frames(prev_len, new_len);
        }
    }

    fn update_metrics_for_new_frames(&mut self, prev_len: usize, new_len: usize) {
        if !self.has_ground_truth() {
            return;
        }
        // Snapshot the relevant per-frame data into owned values before
        // mutating self — keeps the borrow checker happy even though
        // `frames()` returns a slice rooted in `self.source`.
        let snapshots: Vec<(f64, thresh_eval::matching::FrameData, Vec<LifecycleEvent>)> = self
            .frames()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i >= prev_len && *i < new_len)
            .map(|(_, f)| {
                let fd = thresh_eval::matching::FrameData {
                    gt: f.ground_truth.iter().map(|g| (g.id, g.position)).collect(),
                    tracks: f.tracks.iter().map(|t| (t.id, t.position)).collect(),
                };
                (f.timestamp, fd, f.events.clone())
            })
            .collect();

        for (i, (timestamp, fd, events)) in snapshots.into_iter().enumerate() {
            self.last_metrics = Some(self.metrics_builder.update(&fd));
            self.metrics_frame_index = Some(prev_len + i);
            for ev in events {
                if self.event_log.len() == EVENT_LOG_CAPACITY {
                    self.event_log.pop_front();
                }
                self.event_log.push_back((timestamp, ev));
            }
        }
    }

    /// Recompute metrics from scratch for recordings (called once on
    /// load) so the current-frame display has running totals.
    fn rebuild_metrics_for_recording(&mut self) {
        if !matches!(self.source, VizSource::Recording(_)) {
            return;
        }
        self.metrics_builder.reset();
        self.event_log.clear();
        self.last_metrics = None;
        self.metrics_frame_index = None;
        let total = self.total_frames();
        if self.has_ground_truth() {
            self.update_metrics_for_new_frames(0, total);
            // Reset the displayed metrics frame to the current view.
            self.metrics_frame_index = Some(self.current_frame.min(total.saturating_sub(1)));
        }
    }

    /// Apply hotkey input to mutable app state.
    fn handle_input(&mut self, ctx: &egui::Context) {
        let pressed: std::collections::HashSet<egui::Key> =
            ctx.input(|i| i.events.iter().filter_map(key_pressed).collect());
        let k = self.keys;
        if pressed.contains(&k.toggle_help) {
            self.show_help_overlay = !self.show_help_overlay;
        }
        if pressed.contains(&egui::Key::Escape) && self.show_help_overlay {
            self.show_help_overlay = false;
        }
        if pressed.contains(&k.play_pause) {
            self.playing = !self.playing;
        }
        if pressed.contains(&k.step_forward) {
            self.step_forward();
        }
        if pressed.contains(&k.step_back) {
            self.step_back();
        }
        if pressed.contains(&k.toggle_ellipses) {
            self.show_ellipses = !self.show_ellipses;
        }
        if pressed.contains(&k.toggle_associations) {
            self.show_associations = !self.show_associations;
        }
        if pressed.contains(&k.toggle_event_log) {
            self.show_event_log = !self.show_event_log;
        }
        if pressed.contains(&k.screenshot) {
            self.request_screenshot(ctx);
        }
    }

    /// Request the next frame's framebuffer for screenshot export.
    fn request_screenshot(&mut self, ctx: &egui::Context) {
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
    }

    /// Process a captured framebuffer: encode to PNG, write to disk,
    /// surface a toast with the absolute path.
    fn handle_captured_screenshot(&mut self, image: &egui::ColorImage) {
        match save_color_image_as_png(image, &self.screenshot_dir) {
            Ok(path) => {
                self.toast = Some((
                    format!("Screenshot saved: {}", path.display()),
                    Instant::now(),
                ));
            }
            Err(e) => {
                self.toast = Some((format!("Screenshot failed: {e}"), Instant::now()));
            }
        }
    }

    fn step_forward(&mut self) {
        let total = self.total_frames();
        if total > 0 {
            self.current_frame = (self.current_frame + 1).min(total - 1);
        }
    }

    fn step_back(&mut self) {
        self.current_frame = self.current_frame.saturating_sub(1);
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Playback");
        ui.horizontal(|ui| {
            if ui.button("\u{23EE}").clicked() {
                self.current_frame = 0;
            }
            if ui.button("\u{25C0}").clicked() {
                self.step_back();
            }
            let play_label = if self.playing { "\u{23F8}" } else { "\u{25B6}" };
            if ui.button(play_label).clicked() {
                self.playing = !self.playing;
            }
            if ui.button("\u{25B6}\u{25B6}").clicked() {
                self.step_forward();
            }
        });

        let total = self.total_frames();
        if total > 1 && !self.is_live() {
            let max = total - 1;
            ui.add(egui::Slider::new(&mut self.current_frame, 0..=max).text("Frame"));
        }
        if let Some(frame) = self.current() {
            ui.label(format!("Time: {:.2}s", frame.timestamp));
        }

        ui.separator();
        ui.heading("Display");
        ui.checkbox(&mut self.show_detections, "Detections");
        ui.checkbox(&mut self.show_ground_truth, "Ground truth");
        ui.checkbox(&mut self.show_trails, "Track trails");
        ui.checkbox(&mut self.show_associations, "Association lines");
        ui.checkbox(&mut self.show_ellipses, "Covariance ellipses (2σ)");
        ui.checkbox(&mut self.show_event_log, "Lifecycle event log");
        ui.add(egui::Slider::new(&mut self.trail_length, 1..=100).text("Trail length"));
        ui.add(
            egui::Slider::new(&mut self.playback_speed, 0.1..=5.0)
                .text("Speed")
                .logarithmic(true),
        );

        ui.separator();
        ui.label(format!("Press {:?} for help", self.keys.toggle_help));
    }

    fn render_metrics(&self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Metrics");
        if let Some(frame) = self.current() {
            ui.label(format!(
                "Tracks: {} confirmed, {} tentative",
                frame.confirmed_track_count(),
                frame.tentative_track_count(),
            ));
            ui.label(format!("Detections: {}", frame.detections.len()));
            ui.label(format!("Ground truth: {}", frame.ground_truth.len()));
        }

        if self.has_ground_truth() {
            if let Some(m) = &self.last_metrics {
                ui.label(format!("MOTA: {:.3}", m.mota));
                ui.label(format!("MOTP: {:.3}", m.motp));
                ui.label(format!("IDF1: {:.3}", m.idf1));
                ui.label(format!("ID switches: {}", m.id_switches));
            } else {
                ui.label("MOTA: —");
                ui.label("MOTP: —");
                ui.label("IDF1: —");
            }
        } else {
            ui.label("MOTA: n/a — no ground truth");
            ui.label("MOTP: n/a — no ground truth");
            ui.label("IDF1: n/a — no ground truth");
        }

        ui.separator();
        ui.heading("Connection");
        match &self.source {
            VizSource::None => {
                ui.label("No source");
            }
            VizSource::Recording(_) => {
                ui.label("Source: Recording");
            }
            VizSource::Live(live) => {
                let status = live.bridge.status();
                let (text, color) = match status {
                    ConnectionStatus::Connected => ("Connected", egui::Color32::GREEN),
                    ConnectionStatus::Lagging => ("Lagging", egui::Color32::YELLOW),
                    ConnectionStatus::Disconnected => ("Disconnected", egui::Color32::LIGHT_RED),
                };
                ui.colored_label(color, text);
                ui.label(format!("Buffered: {}", live.bridge.buffered_len()));
            }
        }
    }

    fn render_event_log(&self, ui: &mut egui::Ui) {
        if !self.show_event_log {
            return;
        }
        ui.separator();
        ui.heading("Events");
        egui::ScrollArea::vertical()
            .max_height(200.0)
            .show(ui, |ui| {
                if self.event_log.is_empty() {
                    ui.label("(no events yet)");
                } else {
                    for (ts, ev) in self.event_log.iter().rev() {
                        ui.label(format!("t={ts:.2}s  {}", format_event(ev)));
                    }
                }
            });
    }

    fn render_track_list(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Tracks");
        let tracks: Vec<_> = self
            .current()
            .map(|frame| {
                frame
                    .tracks
                    .iter()
                    .map(|t| (t.id, t.position, t.is_confirmed, t.class_label.clone()))
                    .collect()
            })
            .unwrap_or_default();

        egui::ScrollArea::vertical()
            .max_height(160.0)
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

    fn render_plot(&self, ui: &mut egui::Ui) {
        let total = self.total_frames();
        if total == 0 {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() / 3.0);
                if self.is_live() {
                    ui.label("Live source attached. Waiting for first snapshot…");
                } else {
                    ui.label("No source. Run with --recording <file.json> or --stream <addr>.");
                }
            });
            return;
        }

        let Some(frame) = self.current() else {
            ui.label("Frame index out of range.");
            return;
        };

        Plot::new("track_plot")
            .data_aspect(1.0)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                self.render_track_markers(plot_ui, frame);
                self.render_trail_lines(plot_ui, frame);
                if self.show_ellipses {
                    self.render_covariance_ellipses(plot_ui, frame);
                }
                if self.show_associations {
                    self.render_association_lines(plot_ui, frame);
                }
                if self.show_detections {
                    self.render_detection_markers(plot_ui, &frame.detections);
                }
                if self.show_ground_truth {
                    self.render_ground_truth_markers(plot_ui, &frame.ground_truth);
                }
            });
    }

    fn render_track_markers(&self, plot_ui: &mut egui_plot::PlotUi, frame: &VizFrame) {
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
    }

    fn render_trail_lines(&self, plot_ui: &mut egui_plot::PlotUi, frame: &VizFrame) {
        if !self.show_trails {
            return;
        }
        let frames = self.frames();
        let start = self.current_frame.saturating_sub(self.trail_length);
        let window = &frames[start..=self.current_frame.min(frames.len().saturating_sub(1))];
        for track in &frame.tracks {
            let trail: Vec<[f64; 2]> = window
                .iter()
                .filter_map(|f| f.tracks.iter().find(|t| t.id == track.id))
                .map(|t| [t.position[0], t.position[1]])
                .collect();
            if trail.len() >= 2 {
                plot_ui.line(
                    Line::new(PlotPoints::new(trail))
                        .color(id_to_color(track.id))
                        .width(1.5),
                );
            }
        }
    }

    fn render_covariance_ellipses(&self, plot_ui: &mut egui_plot::PlotUi, frame: &VizFrame) {
        for track in &frame.tracks {
            // covariance_diag layout: [σ²_x, σ²_vx, σ²_y, σ²_vy, σ²_z, σ²_vz]
            // 2D position covariance (no off-diagonal in diag-only data).
            let cov_xx = track.covariance_diag[0];
            let cov_yy = track.covariance_diag[2];
            let Some((maj, min, angle)) = ellipse_axes(cov_xx, 0.0, cov_yy) else {
                continue;
            };
            let center = [track.position[0], track.position[1]];
            let pts = ellipse_polyline(center, maj, min, angle, 48);
            plot_ui.line(
                Line::new(PlotPoints::new(pts))
                    .color(id_to_color(track.id).gamma_multiply(0.6))
                    .width(1.0),
            );
        }
    }

    fn render_association_lines(&self, plot_ui: &mut egui_plot::PlotUi, frame: &VizFrame) {
        for &(det_idx, track_id) in &frame.associations {
            let Some(det) = frame.detections.get(det_idx) else {
                continue;
            };
            let Some(track) = frame.tracks.iter().find(|t| t.id == track_id) else {
                continue;
            };
            plot_ui.line(
                Line::new(PlotPoints::new(vec![
                    [det.position[0], det.position[1]],
                    [track.position[0], track.position[1]],
                ]))
                .color(id_to_color(track.id).gamma_multiply(0.4))
                .width(1.0),
            );
        }
    }

    fn render_detection_markers(
        &self,
        plot_ui: &mut egui_plot::PlotUi,
        detections: &[VizDetection],
    ) {
        let pts: Vec<[f64; 2]> = detections
            .iter()
            .map(|d| [d.position[0], d.position[1]])
            .collect();
        if !pts.is_empty() {
            plot_ui.points(
                Points::new(pts)
                    .radius(4.0)
                    .color(egui::Color32::GRAY)
                    .shape(MarkerShape::Diamond)
                    .name("Detections"),
            );
        }
    }

    fn render_ground_truth_markers(
        &self,
        plot_ui: &mut egui_plot::PlotUi,
        ground_truth: &[VizGroundTruth],
    ) {
        let pts: Vec<[f64; 2]> = ground_truth
            .iter()
            .map(|g| [g.position[0], g.position[1]])
            .collect();
        if !pts.is_empty() {
            plot_ui.points(
                Points::new(pts)
                    .radius(5.0)
                    .color(egui::Color32::GREEN)
                    .shape(MarkerShape::Cross)
                    .name("Ground Truth"),
            );
        }
    }

    fn render_help_overlay(&self, ctx: &egui::Context) {
        if !self.show_help_overlay {
            return;
        }
        egui::Window::new("Keyboard Shortcuts")
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                egui::Grid::new("shortcuts_grid")
                    .num_columns(2)
                    .spacing([24.0, 6.0])
                    .show(ui, |ui| {
                        for (key, desc) in self.keys.catalog() {
                            ui.monospace(key);
                            ui.label(desc);
                            ui.end_row();
                        }
                    });
                ui.add_space(8.0);
                ui.label(format!(
                    "Press {:?} or Esc to dismiss",
                    self.keys.toggle_help
                ));
            });
    }

    fn render_toast(&self, ctx: &egui::Context) {
        let Some((msg, t)) = &self.toast else { return };
        if t.elapsed().as_secs_f64() > TOAST_DURATION_SECS {
            return;
        }
        egui::TopBottomPanel::bottom("toast")
            .resizable(false)
            .show(ctx, |ui| {
                ui.colored_label(egui::Color32::LIGHT_BLUE, msg);
            });
    }
}

impl eframe::App for ThreshVizApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Pull any newly-available live snapshots.
        self.ingest_live();

        // For recordings, ensure the metrics builder has been primed
        // once; recordings don't change so we recompute lazily on first
        // update.
        if self.metrics_frame_index.is_none()
            && matches!(self.source, VizSource::Recording(_))
            && self.has_ground_truth()
        {
            self.rebuild_metrics_for_recording();
        }

        self.handle_input(ctx);

        // Capture pending screenshot framebuffer if available.
        let pending = ctx.input(|i| {
            i.raw
                .events
                .iter()
                .filter_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
                .next()
        });
        if let Some(img) = pending {
            self.handle_captured_screenshot(&img);
        }

        egui::SidePanel::left("controls")
            .min_width(240.0)
            .show(ctx, |ui| {
                self.render_controls(ui);
                self.render_metrics(ui);
                self.render_event_log(ui);
                self.render_track_list(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_plot(ui);
        });

        self.render_help_overlay(ctx);
        self.render_toast(ctx);

        // Auto-advance when playing.
        if self.playing {
            ctx.request_repaint();
            let dt = ctx.input(|i| i.stable_dt) as f64;
            self.accumulated_dt += dt * self.playback_speed as f64;
            let interval = self.frame_interval();
            while self.accumulated_dt >= interval {
                self.accumulated_dt -= interval;
                self.step_forward();
            }
        }
    }
}

impl ThreshVizApp {
    fn frame_interval(&self) -> f64 {
        match &self.source {
            VizSource::Recording(r) if r.frames.len() >= 2 => {
                r.duration() / (r.frames.len() - 1) as f64
            }
            _ => 1.0 / 30.0,
        }
    }
}

/// Deterministic hash-based color for a track ID.
pub fn id_to_color(id: u64) -> egui::Color32 {
    let hue = ((id.wrapping_mul(2_654_435_761)) % 360) as f32;
    egui::ecolor::Hsva::new(hue / 360.0, 0.7, 0.9, 1.0).into()
}

/// Convert a `TrackSnapshot` to a `VizFrame`. If `prev` is provided,
/// derive lifecycle events by snapshot-diffing.
fn track_snapshot_to_viz_frame(
    snap: thresh_tracker::streaming::TrackSnapshot,
    prev: Option<&VizFrame>,
) -> VizFrame {
    use crate::recording::VizTrack;
    let tracks = snap
        .tracks
        .into_iter()
        .map(|t| VizTrack {
            id: t.id,
            position: t.position,
            velocity: t.velocity,
            covariance_diag: t.covariance_diag,
            is_confirmed: t.is_confirmed,
            class_label: None,
        })
        .collect();
    let mut frame = VizFrame {
        timestamp: snap.timestamp,
        tracks,
        detections: Vec::new(),
        ground_truth: Vec::new(),
        associations: Vec::new(),
        events: Vec::new(),
    };
    if let Some(prev_frame) = prev {
        frame.events = diff_snapshots(prev_frame, &frame);
    }
    frame
}

fn format_event(ev: &LifecycleEvent) -> String {
    match ev {
        LifecycleEvent::Born { id } => format!("Born  ID {id}"),
        LifecycleEvent::Died { id } => format!("Died  ID {id}"),
        LifecycleEvent::IdSwitched { from, to } => format!("Switch  {from} → {to}"),
        LifecycleEvent::Merged { from, into } => format!("Merge  {from} → {into}"),
    }
}

fn key_pressed(event: &egui::Event) -> Option<egui::Key> {
    match event {
        egui::Event::Key {
            key,
            pressed: true,
            repeat: false,
            ..
        } => Some(*key),
        _ => None,
    }
}

fn save_color_image_as_png(
    image: &egui::ColorImage,
    dir: &std::path::Path,
) -> std::io::Result<PathBuf> {
    use std::io::ErrorKind;

    std::fs::create_dir_all(dir)?;
    let filename = format!(
        "thresh-viz-screenshot-{}.png",
        chrono_compat_iso8601_utc_now()
    );
    let path = dir.join(filename);

    let [w, h] = [image.width() as u32, image.height() as u32];
    let mut rgba = Vec::with_capacity(image.pixels.len() * 4);
    for px in &image.pixels {
        rgba.push(px.r());
        rgba.push(px.g());
        rgba.push(px.b());
        rgba.push(px.a());
    }
    let buf: image::RgbaImage = image::RgbaImage::from_raw(w, h, rgba)
        .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "framebuffer size mismatch"))?;
    buf.save(&path)
        .map_err(|e| std::io::Error::other(format!("PNG encode failed: {e}")))?;
    Ok(path.canonicalize().unwrap_or(path))
}

/// ISO-8601 UTC timestamp without external `chrono` dependency.
fn chrono_compat_iso8601_utc_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let total_secs = dur.as_secs();
    let (year, month, day, hour, minute, second) = unix_to_ymdhms(total_secs);
    format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z")
}

/// Civil-from-days algorithm (Howard Hinnant). Avoids pulling in a date
/// crate solely for screenshot filenames.
fn unix_to_ymdhms(secs: u64) -> (i32, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
    let time_of_day = secs % 86_400;
    let hour = (time_of_day / 3600) as u32;
    let minute = ((time_of_day % 3600) / 60) as u32;
    let second = (time_of_day % 60) as u32;

    // Days since 1970-01-01 → civil date (Hinnant 2013).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if m <= 2 { y + 1 } else { y } as i32;
    (year, m, d, hour, minute, second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::{VizDetection, VizGroundTruth, VizTrack};

    fn make_frame(timestamp: f64, track_ids: &[u64]) -> VizFrame {
        VizFrame {
            timestamp,
            tracks: track_ids
                .iter()
                .map(|&id| VizTrack {
                    id,
                    position: [id as f64 * 10.0, 0.0, 0.0],
                    velocity: [0.0; 3],
                    covariance_diag: [1.0; 6],
                    is_confirmed: true,
                    class_label: None,
                })
                .collect(),
            detections: Vec::<VizDetection>::new(),
            ground_truth: Vec::<VizGroundTruth>::new(),
            associations: Vec::new(),
            events: Vec::new(),
        }
    }

    #[test]
    fn key_bindings_default_catalog_is_complete() {
        let kb = KeyBindings::default();
        let cat = kb.catalog();
        assert!(cat.iter().any(|(_, d)| d.contains("Play")));
        assert!(cat.iter().any(|(_, d)| d.contains("Screenshot")));
        assert!(cat.iter().any(|(_, d)| d.contains("ellipses")));
        assert!(cat.iter().any(|(_, d)| d.contains("association")));
    }

    #[test]
    fn unix_to_ymdhms_known_epochs() {
        // 1970-01-01T00:00:00Z
        assert_eq!(unix_to_ymdhms(0), (1970, 1, 1, 0, 0, 0));
        // 1970-01-02T00:00:00Z
        assert_eq!(unix_to_ymdhms(86_400), (1970, 1, 2, 0, 0, 0));
        // 2000-01-01T00:00:00Z
        assert_eq!(unix_to_ymdhms(946_684_800), (2000, 1, 1, 0, 0, 0));
        // 2025-04-25T22:00:00Z
        assert_eq!(unix_to_ymdhms(1_745_618_400), (2025, 4, 25, 22, 0, 0));
        // Sub-day arithmetic.
        assert_eq!(
            unix_to_ymdhms(86_400 + 3600 + 60 + 7),
            (1970, 1, 2, 1, 1, 7)
        );
    }

    #[test]
    fn iso8601_filename_is_well_formed() {
        let s = chrono_compat_iso8601_utc_now();
        // YYYYMMDDTHHMMSSZ → 16 chars
        assert_eq!(s.len(), 16);
        assert!(s.ends_with('Z'));
        assert!(s.chars().nth(8) == Some('T'));
    }

    #[test]
    fn from_source_recording_populates_frames() {
        let mut rec = Recording::new("test");
        rec.push_frame(make_frame(0.0, &[1, 2]));
        rec.push_frame(make_frame(1.0, &[1, 2, 3]));
        let app = ThreshVizApp::new(Some(rec));
        assert_eq!(app.total_frames(), 2);
        assert_eq!(app.current().unwrap().tracks.len(), 2);
    }

    #[test]
    fn snapshot_diff_emits_birth_event_in_live_ingest() {
        use thresh_tracker::streaming::{TrackSnapshot, TrackState};
        let snap1 = TrackSnapshot {
            timestamp: 0.0,
            tracks: vec![TrackState {
                id: 1,
                position: [0.0; 3],
                velocity: [0.0; 3],
                covariance_diag: [1.0; 6],
                is_confirmed: true,
            }],
            frames_dropped: 0,
        };
        let snap2 = TrackSnapshot {
            timestamp: 1.0,
            tracks: vec![
                TrackState {
                    id: 1,
                    position: [0.0; 3],
                    velocity: [0.0; 3],
                    covariance_diag: [1.0; 6],
                    is_confirmed: true,
                },
                TrackState {
                    id: 2,
                    position: [10.0, 0.0, 0.0],
                    velocity: [0.0; 3],
                    covariance_diag: [1.0; 6],
                    is_confirmed: true,
                },
            ],
            frames_dropped: 0,
        };
        let f1 = track_snapshot_to_viz_frame(snap1, None);
        let f2 = track_snapshot_to_viz_frame(snap2, Some(&f1));
        assert_eq!(f2.events, vec![LifecycleEvent::Born { id: 2 }]);
    }
}
