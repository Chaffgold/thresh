fn main() {
    #[cfg(feature = "gui")]
    {
        // TODO: parse args (--recording <file>), launch eframe app
        eprintln!("thresh-viz: build with --features gui to enable the visualization window");
        eprintln!("Usage: thresh-viz [--recording <file.json>]");
    }
    #[cfg(not(feature = "gui"))]
    {
        eprintln!("thresh-viz: build with --features gui to enable the visualization window");
        eprintln!("Data layer available for recording/playback without GUI.");
    }
}
