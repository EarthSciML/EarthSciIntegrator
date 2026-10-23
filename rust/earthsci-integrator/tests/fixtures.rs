use std::fs;
use std::path::Path;

#[test]
fn run_config_fixtures_target_this_spec_version() {
    let dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/conformance/run-configs/valid");
    let mut n = 0;
    for entry in fs::read_dir(&dir).expect("fixture directory") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let cfg: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            cfg["esm_integrator"],
            earthsci_integrator::SPEC_VERSION,
            "{}",
            path.display()
        );
        n += 1;
    }
    assert!(n > 0, "no fixtures in {}", dir.display());
}
