#[cfg(feature = "gui")]
fn main() -> eframe::Result<()> {
    use eframe::egui;
    use thresh_viz::app::ThreshVizApp;
    use thresh_viz::recording::Recording;

    let args: Vec<String> = std::env::args().collect();
    let recording = if let Some(pos) = args.iter().position(|a| a == "--recording") {
        args.get(pos + 1)
            .map(|p| Recording::load_json(p).expect("failed to load recording"))
    } else {
        None
    };

    let app = ThreshVizApp::new(recording);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native("thresh-viz", options, Box::new(|_cc| Ok(Box::new(app))))
}

#[cfg(not(feature = "gui"))]
fn main() {
    eprintln!("thresh-viz: build with --features gui to enable the visualization window");
    eprintln!("Data layer available for recording/playback without GUI.");
}
