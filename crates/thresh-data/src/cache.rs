//! Cache directory management for downloaded datasets.
//!
//! Two complementary APIs live here:
//!
//! - [`cache_dir`] / [`cache_path`] / [`is_cached`] — resolve a human-readable
//!   cache location given a source, dataset, and filename. These are what
//!   callers use when the filename is already known (e.g. a raw response
//!   body cache keyed on a request tuple).
//! - [`content_hash_key`] — derive a deterministic, collision-resistant
//!   cache key from an arbitrary list of request-identifying parts (URL,
//!   query params, bounding box floats, etc). Callers that previously
//!   built cache keys by string-concatenating human-readable fields
//!   (e.g. `states_{time}_{lat_min}_{lat_max}...`) can switch to this
//!   helper to avoid collisions when two requests only differ in a field
//!   that was not part of the composed key. See the OpenSky client in
//!   `adsb.rs` for an example.

use std::hash::{DefaultHasher, Hash, Hasher};
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

/// Compute a deterministic, collision-resistant cache key from a list of
/// request-identifying parts.
///
/// The result is a 16-character lowercase hex string derived from a
/// `DefaultHasher` over `(namespace, parts)` — short enough to keep on disk
/// comfortably, long enough to make accidental collisions vanishingly rare
/// for any realistic dataset size. `namespace` lets callers bucket keys by
/// endpoint (e.g. `"opensky/states"` vs `"opensky/track"`) so the same
/// parts under different endpoints never collide.
///
/// # Example
///
/// ```
/// use thresh_data::cache::content_hash_key;
///
/// let a = content_hash_key("opensky/states", &["1234567", "40.6,40.8,-74.1,-73.7"]);
/// let b = content_hash_key("opensky/states", &["1234567", "40.6,40.8,-74.1,-73.7"]);
/// let c = content_hash_key("opensky/states", &["1234568", "40.6,40.8,-74.1,-73.7"]);
///
/// assert_eq!(a, b);                       // same parts → same key
/// assert_ne!(a, c);                       // one field changed → different key
/// assert_eq!(a.len(), 16);                // 64 bits as lowercase hex
/// assert!(a.chars().all(|ch| ch.is_ascii_hexdigit()));
/// ```
///
/// # When to use this
///
/// Prefer `content_hash_key` when any of the following apply:
///
/// - The set of request-identifying fields may grow over time (new query
///   parameters). With hand-composed keys, adding a parameter later
///   silently collides with old cache entries.
/// - Fields are floating-point. Formatted-float concatenation is
///   easy to get wrong (rounding, locale, `-0.0`). Hashing sidesteps
///   the whole class of formatting bugs.
/// - The human-readable key is too long for comfortable filesystem use.
///
/// Stick with the existing human-readable key when debuggability of the
/// cache directory matters more than collision resistance.
pub fn content_hash_key(namespace: &str, parts: &[&str]) -> String {
    let mut hasher = DefaultHasher::new();
    namespace.hash(&mut hasher);
    for part in parts {
        part.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
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
        assert!(!is_cached(
            "nonexistent_src",
            "nonexistent_ds",
            "nofile.bin"
        ));
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

    // ---- content-hash cache key (§3.8) ----

    #[test]
    fn content_hash_key_is_deterministic() {
        let a = content_hash_key(
            "opensky/states",
            &["time=1234567", "bbox=40.6,40.8,-74.1,-73.7"],
        );
        let b = content_hash_key(
            "opensky/states",
            &["time=1234567", "bbox=40.6,40.8,-74.1,-73.7"],
        );
        assert_eq!(a, b);
    }

    #[test]
    fn content_hash_key_distinguishes_parts() {
        let a = content_hash_key("opensky/states", &["time=1234567"]);
        let b = content_hash_key("opensky/states", &["time=1234568"]);
        assert_ne!(a, b);
    }

    #[test]
    fn content_hash_key_distinguishes_namespaces() {
        // Same payload under different namespaces must not collide.
        let a = content_hash_key("opensky/states", &["icao24=a1b2c3"]);
        let b = content_hash_key("opensky/tracks", &["icao24=a1b2c3"]);
        assert_ne!(a, b);
    }

    #[test]
    fn content_hash_key_is_hex_16_chars() {
        let key = content_hash_key("ns", &["p"]);
        assert_eq!(key.len(), 16);
        assert!(
            key.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn content_hash_key_part_order_matters() {
        // Swapping the order of parts gives a different key, so callers
        // can't accidentally unify cache entries whose identity depends
        // on field order (e.g. positional SGP4 query parameters).
        let a = content_hash_key("ns", &["foo", "bar"]);
        let b = content_hash_key("ns", &["bar", "foo"]);
        assert_ne!(a, b);
    }

    #[test]
    fn content_hash_key_empty_parts_valid() {
        // Zero-part call is still a valid namespace-only key.
        let key = content_hash_key("namespace-only", &[]);
        assert_eq!(key.len(), 16);
    }
}
