//! Which version of a dependency a build resolved, read from the `Cargo.lock`
//! files a build script can reach. Shared by `build.rs` and the crate's tests
//! (`tests/lockfile.rs`), so both read lock files the same way.
//!
//! Cargo gives a build script no dependency versions and no path to the lock
//! file of the workspace being built, so the lock is looked for in two
//! places: above `OUT_DIR` (the target directory, which in the usual layout
//! sits inside the workspace being built) and above the crate's manifest. A
//! lock file inside a packaged copy of the crate (one unpacked from a
//! `.crate` file) is skipped, because Cargo ignores it when the crate is a
//! dependency. Either place can hold the wrong workspace's lock: a target
//! directory shared between checkouts, or relocated into another workspace,
//! puts a different checkout's lock above `OUT_DIR`; a path or git dependency
//! has its own repository's lock above its manifest. So the answer is given
//! only when it cannot be the wrong one:
//!
//! 1. a lock file *lists* the crate when it has a `[[package]]` entry with
//!    the crate's name and version; every reachable lock that lists it is
//!    read, not only the nearest;
//! 2. at least one lock that lists it must be above `OUT_DIR`, since a lock
//!    reached only from the manifest belongs to the crate's own repository,
//!    which need not be the workspace being built;
//! 3. every lock that lists it must name the same version of the dependency.
//!    A lock that lists the crate but does not determine the version (the
//!    dependency entry is missing, or names no version while the lock holds
//!    several versions of the dependency or none) makes the answer unknown;
//!    the search does not go on to a farther lock.
//!
//! Otherwise the version is unknown. A target directory outside every
//! workspace that lists the crate (`CARGO_TARGET_DIR` set elsewhere) therefore
//! gives unknown, as does a crate built from two checkouts whose locks
//! disagree.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// What one lock file says about the dependency's version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Listing {
    /// The lock has no entry for the crate at its version.
    NotListed,
    /// The lock lists the crate but does not determine the version.
    Indeterminate,
    /// The lock lists the crate, depending on this version.
    Version(String),
}

/// The crate whose build asks, and the dependency it asks about.
pub struct Query<'a> {
    pub me: &'a str,
    pub my_version: &'a str,
    pub dep: &'a str,
}

/// What [`resolve`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    /// The version, when rules 1–3 determine it.
    pub version: Option<String>,
    /// Every lock file the search read, for `cargo:rerun-if-changed`.
    pub read: Vec<PathBuf>,
}

/// What the lock file text `text` says about `q.dep`'s version.
pub fn listing(text: &str, q: &Query) -> Listing {
    let Ok(doc) = text.parse::<toml::Table>() else {
        return Listing::NotListed;
    };
    let Some(packages) = doc.get("package").and_then(toml::Value::as_array) else {
        return Listing::NotListed;
    };
    let field =
        |p: &toml::Value, k: &str| p.get(k).and_then(toml::Value::as_str).map(str::to_string);
    let dep_versions: BTreeSet<String> = packages
        .iter()
        .filter(|p| field(p, "name").as_deref() == Some(q.dep))
        .filter_map(|p| field(p, "version"))
        .collect();
    let mine: Vec<&toml::Value> = packages
        .iter()
        .filter(|p| {
            field(p, "name").as_deref() == Some(q.me)
                && field(p, "version").as_deref() == Some(q.my_version)
        })
        .collect();
    if mine.is_empty() {
        return Listing::NotListed;
    }
    // The same name and version can appear from two sources; each entry
    // must say the same.
    let mut found: Option<String> = None;
    for p in mine {
        // Entries are `name`, `name version` or `name version (source)`; a
        // version is written only when the lock holds several of the name.
        let entry = p
            .get("dependencies")
            .and_then(toml::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(toml::Value::as_str)
            .find(|d| d.split_whitespace().next() == Some(q.dep));
        let v = match entry.map(|d| d.split_whitespace().nth(1)) {
            None => return Listing::Indeterminate,
            Some(Some(v)) => v.to_string(),
            Some(None) if dep_versions.len() == 1 => dep_versions.first().unwrap().clone(),
            Some(None) => return Listing::Indeterminate,
        };
        match &found {
            Some(f) if *f != v => return Listing::Indeterminate,
            _ => found = Some(v),
        }
    }
    Listing::Version(found.unwrap())
}

/// The dependency's version by rules 1–3, searching above `out_dir` and above
/// `manifest_dir` (`packaged` when the crate was unpacked from a `.crate`
/// file, whose own lock file is not the one in use).
pub fn resolve(out_dir: &Path, manifest_dir: &Path, packaged: bool, q: &Query) -> Resolution {
    let above_out: Vec<PathBuf> = out_dir.ancestors().map(|d| d.join("Cargo.lock")).collect();
    let above_manifest = manifest_dir
        .ancestors()
        .filter(|d| !(packaged && *d == manifest_dir))
        .map(|d| d.join("Cargo.lock"));
    let mut read: Vec<PathBuf> = Vec::new();
    let mut versions: BTreeSet<Option<String>> = BTreeSet::new();
    let mut listed_above_out = false;
    for lock in above_out.iter().cloned().chain(above_manifest) {
        if read.contains(&lock) {
            continue;
        }
        let Ok(text) = fs::read_to_string(&lock) else {
            continue;
        };
        read.push(lock.clone());
        match listing(&text, q) {
            Listing::NotListed => continue,
            Listing::Indeterminate => {
                versions.insert(None);
            }
            Listing::Version(v) => {
                versions.insert(Some(v));
            }
        }
        listed_above_out |= above_out.contains(&lock);
    }
    let version = match (listed_above_out, versions.len()) {
        (true, 1) => versions.into_iter().next().unwrap(),
        _ => None,
    };
    Resolution { version, read }
}
