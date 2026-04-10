//! Cache directory management for downloaded datasets.

use std::fs;
use std::path::PathBuf;

/// Return the user's home directory.
fn dirs_home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("HOME environment variable not set")
}

/// Return the cache directory for a given source and dataset, creating it if
/// necessary.
///
/// Path: `~/.thresh/data/<source>/<dataset>/`
pub fn cache_dir(source: &str, dataset: &str) -> PathBuf {
    let dir = dirs_home()
        .join(".thresh")
        .join("data")
        .join(source)
        .join(dataset);
    fs::create_dir_all(&dir).expect("failed to create cache directory");
    dir
}

/// Check whether a file has already been cached.
pub fn is_cached(source: &str, dataset: &str, filename: &str) -> bool {
    cache_path(source, dataset, filename).exists()
}

/// Return the full path to a cached file (without creating anything).
pub fn cache_path(source: &str, dataset: &str, filename: &str) -> PathBuf {
    dirs_home()
        .join(".thresh")
        .join("data")
        .join(source)
        .join(dataset)
        .join(filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_creates_directory() {
        let dir = cache_dir("test_source", "test_dataset");
        assert!(dir.exists());
        assert!(dir.is_dir());
        // Clean up.
        let _ = fs::remove_dir_all(dirs_home().join(".thresh").join("data").join("test_source"));
    }

    #[test]
    fn is_cached_returns_false_for_missing() {
        assert!(!is_cached(
            "nonexistent_src",
            "nonexistent_ds",
            "nofile.bin"
        ));
    }

    #[test]
    fn cache_path_builds_correct_path() {
        let p = cache_path("opensky", "flights", "data.csv");
        assert!(p.ends_with(".thresh/data/opensky/flights/data.csv"));
    }
}
