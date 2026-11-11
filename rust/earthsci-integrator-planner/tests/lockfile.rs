//! The lock-file rules of `build/lockfile.rs`, which build.rs uses to report
//! the earthsci-astdiff version in `plan.report.json` (spec/plan.md §0.2), on
//! lock files written into Cargo's test directory.

#[path = "../build/lockfile.rs"]
mod lockfile;

use lockfile::{listing, resolve, Listing, Query};
use std::fs;
use std::path::{Path, PathBuf};

const Q: Query = Query {
    me: "planner",
    my_version: "0.0.1",
    dep: "astdiff",
};

/// A lock file with the planner at `mine` (None: not listed) depending on
/// `dep_entry`, and the dependency at each of `dep_versions`.
fn lock(mine: Option<&str>, dep_entry: &str, dep_versions: &[&str]) -> String {
    let mut s = String::from("version = 4\n\n");
    for v in dep_versions {
        s += &format!(
            "[[package]]\nname = \"astdiff\"\nversion = \"{v}\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n\n"
        );
    }
    if let Some(v) = mine {
        s += &format!(
            "[[package]]\nname = \"planner\"\nversion = \"{v}\"\ndependencies = [\n \"{dep_entry}\",\n \"serde\",\n]\n\n"
        );
    }
    s
}

fn root(case: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("lockfile")
        .join(case);
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// The lock files read under a test's root (the search also reaches the
/// repository's own lock above Cargo's test directory).
fn under(read: &[PathBuf], root: &Path) -> Vec<PathBuf> {
    read.iter()
        .filter(|l| l.starts_with(root))
        .cloned()
        .collect()
}

fn put(dir: &Path, text: &str) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("Cargo.lock"), text).unwrap();
}

#[test]
fn listings() {
    assert_eq!(
        listing(&lock(Some("0.0.1"), "astdiff", &["0.5.1"]), &Q),
        Listing::Version("0.5.1".into())
    );
    // Several versions in the lock: the entry names its own.
    assert_eq!(
        listing(
            &lock(Some("0.0.1"), "astdiff 0.5.0", &["0.4.0", "0.5.0"]),
            &Q
        ),
        Listing::Version("0.5.0".into())
    );
    assert_eq!(
        listing(
            &lock(
                Some("0.0.1"),
                "astdiff 0.5.0 (registry+https://github.com/rust-lang/crates.io-index)",
                &["0.4.0", "0.5.0"]
            ),
            &Q
        ),
        Listing::Version("0.5.0".into())
    );
    // Listed, but the version is not determined.
    assert_eq!(
        listing(&lock(Some("0.0.1"), "astdiff", &["0.4.0", "0.5.0"]), &Q),
        Listing::Indeterminate
    );
    assert_eq!(
        listing(&lock(Some("0.0.1"), "other", &["0.5.0"]), &Q),
        Listing::Indeterminate
    );
    // Not listed: absent, or another version of the crate.
    assert_eq!(listing(&lock(None, "", &["0.5.0"]), &Q), Listing::NotListed);
    assert_eq!(
        listing(&lock(Some("0.0.2"), "astdiff", &["0.5.0"]), &Q),
        Listing::NotListed
    );
    assert_eq!(listing("not toml [", &Q), Listing::NotListed);
}

#[test]
fn the_workspace_being_built() {
    // The usual layout: target inside the workspace.
    let r = root("usual");
    put(&r, &lock(Some("0.0.1"), "astdiff", &["0.5.1"]));
    let out = r.join("target/debug/build/planner-1/out");
    let res = resolve(&out, &r.join("planner"), false, &Q);
    assert_eq!(res.version.as_deref(), Some("0.5.1"));
    assert_eq!(under(&res.read, &r), vec![r.join("Cargo.lock")]);
}

#[test]
fn a_shared_target_directory() {
    // A worktree built into the main checkout's target directory, the two
    // locks disagreeing: unknown, not the main checkout's version.
    let r = root("shared");
    put(&r.join("main"), &lock(Some("0.0.1"), "astdiff", &["0.5.0"]));
    put(&r.join("wt"), &lock(Some("0.0.1"), "astdiff", &["0.5.1"]));
    let out = r.join("main/target/debug/build/planner-1/out");
    let res = resolve(&out, &r.join("wt/planner"), false, &Q);
    assert_eq!(res.version, None);
    assert_eq!(
        under(&res.read, &r),
        vec![r.join("main/Cargo.lock"), r.join("wt/Cargo.lock")]
    );
    // Agreeing, the version is known.
    put(&r.join("wt"), &lock(Some("0.0.1"), "astdiff", &["0.5.0"]));
    let res = resolve(&out, &r.join("wt/planner"), false, &Q);
    assert_eq!(res.version.as_deref(), Some("0.5.0"));
}

#[test]
fn a_path_dependency_of_another_workspace() {
    let r = root("consumer");
    put(&r.join("esi"), &lock(Some("0.0.1"), "astdiff", &["0.5.0"]));
    put(&r.join("app"), &lock(Some("0.0.1"), "astdiff", &["0.5.1"]));
    let manifest = r.join("esi/planner");
    // The consumer's target directory is outside it: only the crate's own
    // repository's lock is reachable, and it is not used.
    let res = resolve(
        &r.join("elsewhere/debug/build/planner-1/out"),
        &manifest,
        false,
        &Q,
    );
    assert_eq!(res.version, None);
    // Inside it, the two locks disagree: unknown.
    let res = resolve(
        &r.join("app/target/debug/build/planner-1/out"),
        &manifest,
        false,
        &Q,
    );
    assert_eq!(res.version, None);
    // A packaged copy's own lock is skipped.
    put(&r.join("pkg"), &lock(Some("0.0.1"), "astdiff", &["0.4.0"]));
    let res = resolve(
        &r.join("app/target/debug/build/planner-1/out"),
        &r.join("pkg"),
        true,
        &Q,
    );
    assert_eq!(res.version.as_deref(), Some("0.5.1"));
    assert!(!res.read.contains(&r.join("pkg/Cargo.lock")));
}

#[test]
fn an_indeterminate_lock_stops_the_search() {
    // The nearest lock lists the crate without determining the version; a
    // farther lock that would determine one is not used.
    let r = root("indeterminate");
    put(&r, &lock(Some("0.0.1"), "astdiff", &["0.5.0"]));
    put(
        &r.join("ws"),
        &lock(Some("0.0.1"), "astdiff", &["0.4.0", "0.5.1"]),
    );
    let res = resolve(
        &r.join("ws/target/debug/build/planner-1/out"),
        &r.join("ws/planner"),
        false,
        &Q,
    );
    assert_eq!(res.version, None);
    // A lock that does not list the crate is skipped.
    put(&r.join("ws"), &lock(None, "", &["0.5.1"]));
    let res = resolve(
        &r.join("ws/target/debug/build/planner-1/out"),
        &r.join("ws/planner"),
        false,
        &Q,
    );
    assert_eq!(res.version.as_deref(), Some("0.5.0"));
}
