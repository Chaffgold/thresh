#[cfg(feature = "gui")]
fn main() -> eframe::Result<()> {
    use std::path::PathBuf;

    use eframe::egui;
    use thresh_viz::app::ThreshVizApp;
    use thresh_viz::recording::Recording;

    let args: Vec<String> = std::env::args().collect();

    let recording_path = arg_value(&args, "--recording");
    let stream_addr = arg_value(&args, "--stream");
    let max_buffered =
        arg_value(&args, "--max-buffered-snapshots").and_then(|v| v.parse::<usize>().ok());
    let screenshot_dir = arg_value(&args, "--screenshot-dir").map(PathBuf::from);

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }

    let mut app = if let Some(addr) = stream_addr {
        eprintln!(
            "thresh-viz: --stream is reserved for a future cross-process bridge (requested addr: {addr}). \
             Build a SnapshotBridge from your StreamingTracker in process and pass it to ThreshVizApp::live() instead."
        );
        let cap = max_buffered.unwrap_or(thresh_viz::streaming::DEFAULT_HIGH_WATER_MARK);
        let bridge = thresh_viz::streaming::SnapshotBridge::with_capacity(cap)
            .expect("failed to construct snapshot bridge");
        ThreshVizApp::live(bridge)
    } else if let Some(path) = recording_path {
        let rec = Recording::load_json(&path).expect("failed to load recording");
        ThreshVizApp::new(Some(rec))
    } else {
        ThreshVizApp::new(None)
    };

    if let Some(dir) = screenshot_dir {
        app = app.with_screenshot_dir(dir);
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native("thresh-viz", options, Box::new(|_cc| Ok(Box::new(app))))
}

#[cfg(feature = "gui")]
fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

#[cfg(feature = "gui")]
fn print_help() {
    println!("thresh-viz — interactive track visualization dashboard");
    println!();
    println!("USAGE:");
    println!("    thresh-viz [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --recording <FILE.json>          Load a recorded session for offline playback");
    println!("    --stream <ADDR>                  (Reserved for future cross-process bridge)");
    println!(
        "    --max-buffered-snapshots <N>     Max snapshots in the live ingest deque (default 64)"
    );
    println!(
        "    --screenshot-dir <PATH>          Directory to save PNG screenshots (default cwd)"
    );
    println!("    -h, --help                       Print this help message");
    println!();
    println!("HOTKEYS (also visible in-app via the help overlay):");
    println!("    Space        Play / pause");
    println!("    \u{2190} / \u{2192}        Step backward / forward");
    println!("    + / -        Zoom in / out (or scroll wheel)");
    println!("    Drag         Pan");
    println!("    S            Screenshot to PNG");
    println!("    A            Toggle association lines");
    println!("    E            Toggle covariance ellipses (2\u{03C3})");
    println!("    L            Toggle lifecycle event log");
    println!("    ?            Toggle help overlay");
}

#[cfg(not(feature = "gui"))]
fn main() {
    eprintln!("thresh-viz: build with --features gui to enable the visualization window");
    eprintln!("Data layer available for recording/playback without GUI.");
}
