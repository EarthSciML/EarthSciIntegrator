#!/usr/bin/env bash
# Copies the spec data the packages read into each package's own `data/`.
#
# spec/ is the one place to edit these files. But a registered Julia package
# ships only julia/EarthSciIntegrator.jl/, and a published crate only its own
# directory, so neither can reach spec/ once installed. Each keeps a copy
# instead, and CI (ci.yml, `schemas` job) fails when a copy differs from spec/.
#
# Run after changing anything under spec/tableaus or the run-config schema:
#   tools/sync-package-data.sh
set -euo pipefail

cd "$(dirname "$0")/.."

sync_dir() { # <source dir> <destination dir>: an exact copy of the *.json files
  rm -rf "$2"
  mkdir -p "$2"
  cp "$1"/*.json "$2"/
}
sync_file() { # <source file> <destination file>
  mkdir -p "$(dirname "$2")"
  cp "$1" "$2"
}

sync_dir  spec/tableaus                         julia/EarthSciIntegrator.jl/data/tableaus
sync_file spec/schemas/run-config.schema.json   julia/EarthSciIntegrator.jl/data/schemas/run-config.schema.json
sync_dir  spec/tableaus                         rust/earthsci-integrator-solvers/data/tableaus
sync_file spec/schemas/run-config.schema.json   rust/earthsci-integrator-planner/data/schemas/run-config.schema.json
