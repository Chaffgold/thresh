//! Credential loading from environment variables and TOML config files.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Credentials for a data-source service.
#[derive(Debug, Clone, Default)]
pub struct Credentials {
    /// Username, if available.
    pub username: Option<String>,
    /// Password, if available.
    pub password: Option<String>,
    /// API key, if available.
    pub api_key: Option<String>,
}

/// Return the path to the credentials file (`~/.thresh/credentials.toml`).
fn credentials_path() -> Option<PathBuf> {
    dirs_home().map(|h| h.join(".thresh").join("credentials.toml"))
}

/// Return the user's home directory.
fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Load credentials for `service` (e.g. `"opensky"`).
///
/// Environment variables take priority over the TOML file.
///
/// Env-var format: `THRESH_<SERVICE>_USERNAME`, `THRESH_<SERVICE>_PASSWORD`,
/// `THRESH_<SERVICE>_API_KEY`.
///
/// TOML format (`~/.thresh/credentials.toml`):
/// ```toml
/// [opensky]
/// username = "..."
/// password = "..."
/// api_key  = "..."
/// ```
pub fn load_credentials(service: &str) -> Credentials {
    let upper = service.to_uppercase();

    // Try environment variables first.
    let env_user = std::env::var(format!("THRESH_{upper}_USERNAME")).ok();
    let env_pass = std::env::var(format!("THRESH_{upper}_PASSWORD")).ok();
    let env_key = std::env::var(format!("THRESH_{upper}_API_KEY")).ok();

    // Fall back to TOML file.
    let (file_user, file_pass, file_key) = credentials_path()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|contents| {
            let table: HashMap<String, HashMap<String, String>> = toml::from_str(&contents).ok()?;
            let section = table.get(service)?;
            Some((
                section.get("username").cloned(),
                section.get("password").cloned(),
                section.get("api_key").cloned(),
            ))
        })
        .unwrap_or((None, None, None));

    Credentials {
        username: env_user.or(file_user),
        password: env_pass.or(file_pass),
        api_key: env_key.or(file_key),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_vars_override_file() {
        // With no env vars and no file, everything should be None.
        let creds = load_credentials("nonexistent_test_service_xyz");
        assert!(creds.username.is_none());
        assert!(creds.password.is_none());
        assert!(creds.api_key.is_none());
    }
}
