//! Records the version of `earthsci-astdiff` this crate is built against, for
//! `plan.report.json` (spec/plan.md §0.2). `earthsci-ast` exports its own
//! version (`earthsci_ast::LIBRARY_VERSION`); `earthsci-astdiff` does not, and
//! Cargo gives a build script no dependency versions, so this one reads them
//! from the `Cargo.lock` Cargo resolved the build with. A `[patch]` shows
//! there as the patched version, so a build against an unreleased checkout
//! reports that checkout's version, not the requirement in `Cargo.toml`.
//! Which lock files count, and when the version is reported as `unknown`
//! rather than guessed, is in `build/lockfile.rs`.
//!
//! Every lock file the search read is watched, so editing one, or a
//! `[patch]` that changes it, reruns this script. A lock file created later
//! where none was read is not noticed (Cargo has no cheap way to watch a path
//! that does not exist yet); `cargo clean` picks it up.

#[path = "build/lockfile.rs"]
mod lockfile;

use std::env;
use std::path::PathBuf;

fn main() {
    let me = env::var("CARGO_PKG_NAME").unwrap();
    let my_version = env::var("CARGO_PKG_VERSION").unwrap();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // A crate unpacked from a `.crate` file carries `.cargo_vcs_info.json`
    // or `Cargo.toml.orig`; its own lock file is not the one in use.
    let packaged = manifest_dir.join("Cargo.toml.orig").exists()
        || manifest_dir.join(".cargo_vcs_info.json").exists();
    let q = lockfile::Query {
        me: &me,
        my_version: &my_version,
        dep: "earthsci-astdiff",
    };
    let r = lockfile::resolve(&out_dir, &manifest_dir, packaged, &q);
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build/lockfile.rs");
    for lock in &r.read {
        println!("cargo:rerun-if-changed={}", lock.display());
    }
    let version = r.version.unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=ESI_EARTHSCI_ASTDIFF_VERSION={version}");
}
