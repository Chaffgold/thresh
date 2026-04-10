//! Cache directory management for downloaded datasets.

use std::io;
use std::path::PathBuf;

/// Return the user's home directory (cross-platform).
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Return the cache directory for a given source and dataset, creating it if
/// necessary.
///
/// Path: `~/.thresh/data/<source>/<dataset>/`
///
/// Returns an error if the home directory cannot be determined or the
/// directory cannot be created.
pub fn cache_dir(source: &str, dataset: &str) -> io::Result<PathBuf> {
    let home = home_dir()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "home directory not found"))?;
    let dir = home.join(".thresh").join("data").join(source).join(dataset);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Check whether a file has already been cached.
pub fn is_cached(source: &str, dataset: &str, filename: &str) -> bool {
    cache_path(source, dataset, filename)
        .map(|p| p.exists())
        .unwrap_or(false)
}

/// Return the full path to a cached file (without creating anything).
pub fn cache_path(source: &str, dataset: &str, filename: &str) -> io::Result<PathBuf> {
    let home = home_dir()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "home directory not found"))?;
    Ok(home
        .join(".thresh")
        .join("data")
        .join(source)
        .join(dataset)
        .join(filename))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_creates_directory() {
        let dir = cache_dir("test_source", "test_dataset").unwrap();
        assert!(dir.exists());
        assert!(dir.is_dir());
        // Clean up.
        if let Some(home) = home_dir() {
            let _ = std::fs::remove_dir_all(home.join(".thresh").join("data").join("test_source"));
        }
    }

    #[test]
    fn is_cached_returns_false_for_missing() {
        assert!(!is_cached("nonexistent_src", "nonexistent_ds", "nofile.bin"));
    }

    #[test]
    fn cache_path_builds_correct_path() {
        use std::path::Path;
        let p = cache_path("opensky", "flights", "data.csv").unwrap();
        let expected = Path::new(".thresh")
            .join("data")
            .join("opensky")
            .join("flights")
            .join("data.csv");
        assert!(p.ends_with(&expected));
    }
}
