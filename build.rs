//! Records the locked splintr version for the `.tokens.json` sidecar.
//!
//! The version comes from `Cargo.lock` beside the manifest. Without a lock
//! file, or without a splintr entry in it, the recorded version is `unknown`.

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let lock = Path::new(&manifest).join("Cargo.lock");
    if lock.is_file() {
        println!("cargo:rerun-if-changed={}", lock.display());
    }
    let version = fs::read_to_string(&lock)
        .ok()
        .and_then(|text| locked_version(&text, "splintr"))
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=OXIDIZR_SPLINTR_VERSION={version}");
}

/// The `version` line after the first `name = "<name>"` line of a lock file.
fn locked_version(lock: &str, name: &str) -> Option<String> {
    let wanted = format!("name = \"{name}\"");
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line.trim() == wanted {
            let version = lines.next()?.trim();
            return version
                .strip_prefix("version = \"")?
                .strip_suffix('"')
                .map(str::to_owned);
        }
    }
    None
}
