//! Records the version of `earthsci-astdiff` this crate is built against, for
//! `plan.report.json` (spec/plan.md §0.2). `earthsci-ast` exports its own
//! version (`earthsci_ast::LIBRARY_VERSION`); `earthsci-astdiff` does not, and
//! Cargo gives a build script no dependency versions, so this one reads them
//! from the `Cargo.lock` Cargo resolved the build with. A `[patch]` shows
//! there as the patched version, so a build against an unreleased checkout
//! reports that checkout's version, not the requirement in `Cargo.toml`.
//!
//! The lock file is looked for above `OUT_DIR` (the target directory, inside
//! the workspace being built in the usual layout) and then above this crate's
//! manifest. A lock file inside a packaged copy of this crate (from crates.io)
//! is skipped: Cargo ignores it when the crate is a dependency. A lock file
//! counts only if it lists this crate at this version. When none does, the
//! version is reported as `unknown` rather than guessed.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const DEP: &str = "earthsci-astdiff";

fn main() {
    let me = env::var("CARGO_PKG_NAME").unwrap();
    let my_version = env::var("CARGO_PKG_VERSION").unwrap();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // A crate unpacked from a `.crate` file carries `.cargo_vcs_info.json`
    // or `Cargo.toml.orig`; its own lock file is not the one in use.
    let packaged = manifest_dir.join("Cargo.toml.orig").exists()
        || manifest_dir.join(".cargo_vcs_info.json").exists();
    let candidates = out_dir
        .ancestors()
        .chain(
            manifest_dir
                .ancestors()
                .filter(|d| !(packaged && *d == manifest_dir.as_path())),
        )
        .map(|d| d.join("Cargo.lock"));
    let mut found = None;
    let mut seen = Vec::new();
    for lock in candidates {
        if seen.contains(&lock) || !lock.is_file() {
            continue;
        }
        seen.push(lock.clone());
        if let Some(v) = dep_version(&lock, &me, &my_version) {
            found = Some((lock, v));
            break;
        }
    }
    let version = match found {
        Some((lock, v)) => {
            println!("cargo:rerun-if-changed={}", lock.display());
            v
        }
        None => {
            println!("cargo:rerun-if-changed=build.rs");
            "unknown".to_string()
        }
    };
    println!("cargo:rustc-env=ESI_EARTHSCI_ASTDIFF_VERSION={version}");
}

/// One `[[package]]` entry of a lock file: its name, version and
/// dependency strings (`name`, `name version` or `name version (source)`).
struct Package {
    name: String,
    version: String,
    dependencies: Vec<String>,
}

/// The version of [`DEP`] that `me` at `my_version` depends on in `lock`.
fn dep_version(lock: &Path, me: &str, my_version: &str) -> Option<String> {
    let text = fs::read_to_string(lock).ok()?;
    let packages = parse(&text);
    let mine = packages
        .iter()
        .find(|p| p.name == me && p.version == my_version)?;
    let dep = mine
        .dependencies
        .iter()
        .find(|d| d.split_whitespace().next() == Some(DEP))?;
    // A version is written only when the lock holds several of the name.
    if let Some(v) = dep.split_whitespace().nth(1) {
        return Some(v.to_string());
    }
    let mut versions = packages
        .iter()
        .filter(|p| p.name == DEP)
        .map(|p| &p.version);
    match (versions.next(), versions.next()) {
        (Some(v), None) => Some(v.clone()),
        _ => None,
    }
}

/// The `[[package]]` entries of a lock file (format versions 1 to 4), read
/// line by line: only `name`, `version` and `dependencies` are needed.
fn parse(text: &str) -> Vec<Package> {
    let mut out: Vec<Package> = Vec::new();
    let mut in_package = false;
    let mut in_deps = false;
    for line in text.lines() {
        let line = line.trim();
        if !in_deps && line.starts_with('[') {
            // A table header: a package entry, or another table
            // (`[metadata]`, `[[patch.unused]]`), whose keys are skipped.
            in_package = line == "[[package]]";
            if in_package {
                out.push(Package {
                    name: String::new(),
                    version: String::new(),
                    dependencies: Vec::new(),
                });
            }
            continue;
        }
        if !in_package {
            continue;
        }
        let Some(p) = out.last_mut() else { continue };
        if in_deps {
            if line.starts_with(']') {
                in_deps = false;
            } else if let Some(s) = quoted(line) {
                p.dependencies.push(s);
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("name = ") {
            p.name = quoted(rest).unwrap_or_default();
        } else if let Some(rest) = line.strip_prefix("version = ") {
            p.version = quoted(rest).unwrap_or_default();
        } else if let Some(rest) = line.strip_prefix("dependencies = [") {
            // Entries follow on their own lines, or inline before `]`.
            for item in rest.trim_end_matches(']').split(',') {
                if let Some(s) = quoted(item.trim()) {
                    p.dependencies.push(s);
                }
            }
            in_deps = !rest.trim_end().ends_with(']');
        }
    }
    out
}

/// The contents of the first double-quoted string in `s`.
fn quoted(s: &str) -> Option<String> {
    let start = s.find('"')? + 1;
    let end = start + s[start..].find('"')?;
    Some(s[start..end].to_string())
}
