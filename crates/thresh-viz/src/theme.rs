//! Modern styling for the `thresh-viz` egui dashboard.
//!
//! Centralizes palette, typography, spacing, and corner radii so the
//! look-and-feel can be tweaked in one place. Apply via
//! [`apply_modern_theme`] on the `egui::Context` at startup
//! (typically from the `eframe::CreationContext` closure in `main`).

use eframe::egui::{self, Color32, CornerRadius, FontFamily, FontId, Margin, Shadow, Stroke};

/// Accent color used for highlights, headings, and primary indicators.
/// Teal-cyan reads well on the dark background and matches the existing
/// HSV-based per-track colors without clashing.
pub const ACCENT: Color32 = Color32::from_rgb(94, 234, 212);

/// Subtle accent for hover and secondary highlights.
pub const ACCENT_DIM: Color32 = Color32::from_rgb(45, 212, 191);

/// Colors used by the connection status indicator.
pub const STATUS_OK: Color32 = Color32::from_rgb(74, 222, 128);
pub const STATUS_WARN: Color32 = Color32::from_rgb(250, 204, 21);
pub const STATUS_BAD: Color32 = Color32::from_rgb(248, 113, 113);

// Slate palette — modern dark theme inspired by tailwindcss `slate` scale.
const SLATE_950: Color32 = Color32::from_rgb(7, 11, 20);
const SLATE_900: Color32 = Color32::from_rgb(15, 23, 42);
const SLATE_850: Color32 = Color32::from_rgb(22, 32, 56);
const SLATE_800: Color32 = Color32::from_rgb(30, 41, 59);
const SLATE_700: Color32 = Color32::from_rgb(51, 65, 85);
const SLATE_500: Color32 = Color32::from_rgb(100, 116, 139);
const SLATE_300: Color32 = Color32::from_rgb(203, 213, 225);
const SLATE_100: Color32 = Color32::from_rgb(241, 245, 249);

/// Apply the modern dark theme to the given context. Call once at
/// startup (e.g., from `eframe::CreationContext`).
pub fn apply_modern_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    apply_modern_visuals(&mut visuals);
    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    apply_modern_style(&mut style);
    ctx.set_style(style);
}

/// Override the [`egui::Visuals`] palette with the modern theme.
pub fn apply_modern_visuals(v: &mut egui::Visuals) {
    v.dark_mode = true;
    v.override_text_color = Some(SLATE_100);
    v.hyperlink_color = ACCENT;
    v.faint_bg_color = SLATE_900;
    v.extreme_bg_color = SLATE_950;
    v.code_bg_color = SLATE_850;
    v.warn_fg_color = STATUS_WARN;
    v.error_fg_color = STATUS_BAD;
    v.window_fill = SLATE_900;
    v.window_stroke = Stroke::new(1.0, SLATE_700);
    v.window_corner_radius = CornerRadius::same(10);
    v.window_shadow = soft_shadow();
    v.popup_shadow = soft_shadow();
    v.panel_fill = SLATE_900;
    v.menu_corner_radius = CornerRadius::same(6);

    let r6 = CornerRadius::same(6);
    let r8 = CornerRadius::same(8);

    // Inactive widgets (default state).
    v.widgets.noninteractive.bg_fill = SLATE_900;
    v.widgets.noninteractive.weak_bg_fill = SLATE_850;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, SLATE_700);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, SLATE_300);
    v.widgets.noninteractive.corner_radius = r6;

    v.widgets.inactive.bg_fill = SLATE_800;
    v.widgets.inactive.weak_bg_fill = SLATE_850;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, SLATE_700);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, SLATE_300);
    v.widgets.inactive.corner_radius = r6;

    v.widgets.hovered.bg_fill = SLATE_700;
    v.widgets.hovered.weak_bg_fill = SLATE_800;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT_DIM);
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, SLATE_100);
    v.widgets.hovered.corner_radius = r6;

    v.widgets.active.bg_fill = ACCENT_DIM;
    v.widgets.active.weak_bg_fill = SLATE_700;
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.0, SLATE_950);
    v.widgets.active.corner_radius = r6;

    v.widgets.open.bg_fill = SLATE_800;
    v.widgets.open.bg_stroke = Stroke::new(1.0, ACCENT_DIM);
    v.widgets.open.fg_stroke = Stroke::new(1.0, SLATE_100);
    v.widgets.open.corner_radius = r8;

    v.selection.bg_fill = ACCENT_DIM.linear_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0, ACCENT);

    v.striped = true;
}

/// Apply spacing, font, and text-style tweaks.
pub fn apply_modern_style(style: &mut egui::Style) {
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
    style.spacing.window_margin = Margin::same(12);
    style.spacing.menu_margin = Margin::same(8);
    style.spacing.indent = 14.0;
    style.spacing.icon_width = 16.0;
    style.spacing.icon_width_inner = 10.0;
    style.spacing.slider_width = 160.0;
    style.spacing.combo_width = 160.0;
    style.spacing.tooltip_width = 280.0;
    style.spacing.scroll = egui::style::ScrollStyle::solid();

    let mut text_styles = std::collections::BTreeMap::new();
    text_styles.insert(
        egui::TextStyle::Heading,
        FontId::new(20.0, FontFamily::Proportional),
    );
    text_styles.insert(
        egui::TextStyle::Name("H2".into()),
        FontId::new(16.0, FontFamily::Proportional),
    );
    text_styles.insert(
        egui::TextStyle::Body,
        FontId::new(13.5, FontFamily::Proportional),
    );
    text_styles.insert(
        egui::TextStyle::Monospace,
        FontId::new(12.5, FontFamily::Monospace),
    );
    text_styles.insert(
        egui::TextStyle::Button,
        FontId::new(13.5, FontFamily::Proportional),
    );
    text_styles.insert(
        egui::TextStyle::Small,
        FontId::new(11.5, FontFamily::Proportional),
    );
    style.text_styles = text_styles;

    style.visuals.collapsing_header_frame = false;
    style.spacing.combo_height = 220.0;
}

fn soft_shadow() -> Shadow {
    Shadow {
        offset: [0, 6],
        blur: 18,
        spread: 0,
        color: Color32::from_black_alpha(120),
    }
}

/// Build a section header label suitable for sidebar groups. Renders a
/// small uppercase muted label above content — more "modern dashboard"
/// than a full `egui::Heading`.
pub fn section_header(ui: &mut egui::Ui, text: &str) {
    let upper = text.to_uppercase();
    ui.add_space(2.0);
    ui.label(egui::RichText::new(upper).small().color(SLATE_500).strong());
    ui.add_space(2.0);
}

/// Render a simple "pill" indicator: a colored circle + label, used
/// for status indicators.
pub fn status_pill(ui: &mut egui::Ui, color: Color32, label: &str) -> egui::Response {
    ui.horizontal(|ui| {
        let dot_size = egui::vec2(10.0, 10.0);
        let (rect, response) = ui.allocate_exact_size(dot_size, egui::Sense::hover());
        if ui.is_rect_visible(rect) {
            ui.painter().circle_filled(rect.center(), 5.0, color);
            ui.painter().circle_stroke(
                rect.center(),
                5.0,
                Stroke::new(1.0, color.linear_multiply(0.6)),
            );
        }
        ui.label(egui::RichText::new(label).color(SLATE_100));
        response
    })
    .inner
}

/// Render a label/value row with label dimmed and value normal weight.
/// Used in the metric sidebar to keep alignment consistent.
pub fn metric_row(ui: &mut egui::Ui, label: &str, value: impl Into<egui::WidgetText>) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(SLATE_500).small());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(value);
        });
    });
}

/// Monospace "key chip" used in the keyboard help overlay.
pub fn key_chip(ui: &mut egui::Ui, text: &str) {
    egui::Frame::new()
        .fill(SLATE_800)
        .stroke(Stroke::new(1.0, SLATE_700))
        .corner_radius(CornerRadius::same(4))
        .inner_margin(Margin::symmetric(7, 3))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).monospace().color(SLATE_100));
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theme_apply_does_not_panic() {
        let ctx = egui::Context::default();
        apply_modern_theme(&ctx);
        // Verify the override text color was set.
        let v = ctx.style().visuals.clone();
        assert_eq!(v.override_text_color, Some(SLATE_100));
        assert!(v.dark_mode);
        assert_eq!(v.window_corner_radius, CornerRadius::same(10));
    }

    #[test]
    fn modern_style_has_h2_text_style() {
        let mut style = egui::Style::default();
        apply_modern_style(&mut style);
        assert!(
            style
                .text_styles
                .contains_key(&egui::TextStyle::Name("H2".into()))
        );
    }

    #[test]
    fn accent_colors_are_distinct() {
        assert_ne!(ACCENT, ACCENT_DIM);
        assert_ne!(STATUS_OK, STATUS_WARN);
        assert_ne!(STATUS_OK, STATUS_BAD);
    }
}
