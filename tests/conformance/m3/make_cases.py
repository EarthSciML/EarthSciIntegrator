#!/usr/bin/env python3
"""Generate cases.json, the M3 runtime cases (spec/toy-problems.md §5;
spec/partitioning.md §9).

Each case runs a reseact-shaped toy end to end (planner, then runtime) with
one dt_scope for its pointwise leaf. The checks are `invariance` (bitwise
equality across shard counts, lane capacities and thread counts, each
implementation on its own), `scope_agreement` (cell or shard against global,
to tolerance) and `cross_language`.

Usage: python tests/conformance/m3/make_cases.py   (rewrites cases.json beside this file)
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MINI = "../toys/reseact_mini.esm"
MINI_AXES = {"horizontal": ["lon", "lat"], "vertical": "lev"}
MINI_OPS = {
    "transport": {"select": {"locality": ["stencil", "column"]}},
    "chemistry": {"select": {"locality": "pointwise"}},
}
MINI_TSPAN = [79200.0, 90000.0]
MINI_SAVEAT = [79200.0 + 1800.0 * k for k in range(1, 7)]
MINI_LAYOUT = ["Mini.NO2", "Mini.NO", "Mini.O3"]

PHOTO = "../toys/photostationary_cells.esm"
PHOTO_OPS = {"photochemistry": {"select": {"components": "Photo"}}}
PHOTO_LAYOUT = ["Photo.NO2", "Photo.NO", "Photo.O3"]

CROSS = {"check": "cross_language", "rtol": 1e-9, "rtol_steps_differ": 1e-3}
# Shards, lane capacities and threads together; 120 shards is one cell each.
CELL_VARIANTS = [
    {"shards": 1, "lanes": 1, "threads": 1},
    {"shards": 1, "lanes": 8, "threads": 1},
    {"shards": 4, "lanes": 4, "threads": 2},
    {"shards": 7, "lanes": 5, "threads": 3},
    {"shards": 120, "lanes": 2, "threads": 4},
]
# Under shard scope the composition is part of the numerics: shards stay fixed.
SHARD_VARIANTS = [
    {"lanes": 1, "threads": 1},
    {"lanes": 4, "threads": 2},
    {"lanes": 8, "threads": 4},
]


def chem_leaf(alg, scope, partition):
    leaf = {
        "id": "chem",
        "leaf": "chemistry",
        "alg": alg,
        "jacobian": "symbolic",
        "control": {"reltol": 1e-4, "abstol": 1e-9, "dt0": 0.5, "dt_scope": scope},
        "domain": {"nonnegative": "all", "on_violation": "reject"},
        "partition": partition,
    }
    if alg.startswith("Ros"):
        leaf["linsolve"] = {"block_lu": {"pivot": False}}
    return leaf


def mini(case_id, alg, scope, partition, checks):
    return {
        "id": case_id,
        "run_config": {
            "esm_integrator": "0.1.0",
            "model": MINI,
            "axes": MINI_AXES,
            "operators": MINI_OPS,
            "schedule": {
                "sequence": [
                    {
                        "leaf": "transport",
                        "alg": "SSPRK43",
                        "control": {"reltol": 1e-4, "abstol": 1e-6, "dt0": 15},
                    },
                    chem_leaf(alg, scope, partition),
                ]
            },
            "step": {"dt": 300, "hit": ["forcing_refresh", "output"]},
            "output": {"saveat": 1800},
        },
        "tspan": MINI_TSPAN,
        "saveat": MINI_SAVEAT,
        "layout": MINI_LAYOUT,
        "checks": checks,
    }


def photo(case_id, scope, checks):
    return {
        "id": case_id,
        "run_config": {
            "esm_integrator": "0.1.0",
            "model": PHOTO,
            "operators": PHOTO_OPS,
            "schedule": {
                "leaf": "photochemistry",
                "alg": "TRBDF2",
                "control": {"reltol": 1e-6, "abstol": 1e-9, "dt_scope": scope},
                "partition": {"shards": 3},
            },
            "step": {"dt": 600, "hit": ["output"]},
            "output": {"saveat": 600},
        },
        "tspan": [0.0, 3600.0],
        "saveat": [600.0 * k for k in range(1, 7)],
        "layout": PHOTO_LAYOUT,
        "checks": checks,
    }


def cases():
    inv = {"check": "invariance", "variants": CELL_VARIANTS}
    return [
        mini("m3.reseact_mini.global", "Rosenbrock23", "global", {"shards": 4}, [inv, CROSS]),
        mini(
            "m3.reseact_mini.cell",
            "Rosenbrock23",
            "cell",
            {"shards": 4, "lanes": 8},
            [
                inv,
                {"check": "scope_agreement", "against": "m3.reseact_mini.global", "rtol": 1e-3},
                CROSS,
            ],
        ),
        mini("m3.reseact_mini.global_trbdf2", "TRBDF2", "global", {"shards": 4}, [inv, CROSS]),
        mini(
            "m3.reseact_mini.cell_trbdf2",
            "TRBDF2",
            "cell",
            {"shards": 4, "lanes": 8},
            [
                inv,
                {
                    "check": "scope_agreement",
                    "against": "m3.reseact_mini.global_trbdf2",
                    "rtol": 1e-3,
                },
                CROSS,
            ],
        ),
        mini(
            "m3.reseact_mini.shard",
            "TRBDF2",
            "shard",
            {"shards": 4, "reshard": {"every": 2}},
            [
                {"check": "invariance", "variants": SHARD_VARIANTS},
                {
                    "check": "scope_agreement",
                    "against": "m3.reseact_mini.global_trbdf2",
                    "rtol": 1e-3,
                },
                CROSS,
            ],
        ),
        photo(
            "m3.photostationary_cells.global",
            "global",
            [{"check": "invariance", "variants": CELL_VARIANTS[:4]}, CROSS],
        ),
        photo(
            "m3.photostationary_cells.cell",
            "cell",
            [
                {"check": "invariance", "variants": CELL_VARIANTS[:4]},
                {
                    "check": "scope_agreement",
                    "against": "m3.photostationary_cells.global",
                    "rtol": 1e-5,
                },
                CROSS,
            ],
        ),
    ]


def main():
    doc = {
        "esm_integrator": "0.1.0",
        "generated_by": "tests/conformance/m3/make_cases.py",
        "cases": cases(),
    }
    (HERE / "cases.json").write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
