#!/usr/bin/env python3
"""Generate cases.json, the M2 end-to-end cases (spec/toy-problems.md §4).

Each M2 case is an M1 case (../m1/cases.json) run on the .esm version of its
toy problem (../toys/): the planner plans it, the runtime executes the plan,
and the result is compared with the same implementation's result for the M1
case on the hand-written problem (check `toy_equivalence`, D80).

Usage: python tests/conformance/m2/make_cases.py   (rewrites cases.json beside this file)
"""

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# toy model -> (.esm file relative to this directory, operators, layout, depends on t)
PROBLEMS = {
    "toy:forced_decay": (
        "../toys/forced_decay.esm",
        {"decay": {"select": {"components": "Decay"}}},
        ["Decay.u"],
        True,
    ),
    "toy:van_der_pol/nonstiff": (
        "../toys/van_der_pol_nonstiff.esm",
        {"oscillator": {"select": {"components": "Oscillator"}}},
        ["Oscillator.y", "Oscillator.z"],
        False,
    ),
    "toy:van_der_pol/stiff": (
        "../toys/van_der_pol_stiff.esm",
        {"oscillator": {"select": {"components": "Oscillator"}}},
        ["Oscillator.y", "Oscillator.z"],
        False,
    ),
    "toy:robertson": (
        "../toys/robertson.esm",
        {"kinetics": {"select": {"components": "Kinetics"}}},
        ["Kinetics.y1", "Kinetics.y2", "Kinetics.y3"],
        False,
    ),
    "toy:photostationary_cells": (
        "../toys/photostationary_cells.esm",
        {"photochemistry": {"select": {"components": "Photo"}}},
        ["Photo.NO2", "Photo.NO", "Photo.O3"],
        True,
    ),
    "toy:advection_reaction": (
        "../toys/advection_reaction.esm",
        {
            "transport": {"select": {"components": "Transport"}},
            "reaction": {"select": {"components": "Reaction"}},
        },
        ["Reaction.A", "Reaction.B"],
        False,
    ),
}
# A Rosenbrock leaf needs f_t on a time-dependent problem, which planned
# operators do not provide at M2 (spec/operator-interface.md §8.3).
ROSENBROCK = {
    t["name"]
    for t in (
        json.loads(f.read_text()) for f in (HERE.parents[2] / "spec" / "tableaus").glob("*.json")
    )
    if t["family"] == "rosenbrock"
}


def leaves(node):
    if isinstance(node, dict):
        if "leaf" in node or "coupled" in node:
            yield node
        for v in node.values():
            yield from leaves(v)
    elif isinstance(node, list):
        for v in node:
            yield from leaves(v)


def m2_case(m1):
    model = m1["run_config"]["model"]
    if model not in PROBLEMS:
        return None  # source_decay_map: map operators need U6 (M4b)
    path, operators, layout, depends_on_t = PROBLEMS[model]
    rc = copy.deepcopy(m1["run_config"])
    if depends_on_t and any(n.get("alg") in ROSENBROCK for n in leaves(rc["schedule"])):
        return None
    rc["model"] = path
    rc["operators"] = operators
    rc.pop("adjoint", None)
    for node in leaves(rc["schedule"]):
        # M1 toy leaves all run with dt_scope global (spec/toy-problems.md);
        # the planner would otherwise resolve pointwise leaves to cell (D17).
        node.setdefault("control", {}).setdefault("dt_scope", "global")
    xl = next((c for c in m1["checks"] if c["check"] == "cross_language"), None)
    if xl is None:
        raise SystemExit(f"M1 case {m1['id']} has no cross_language check to take tolerances from")
    check = {"check": "toy_equivalence", "against": m1["id"], "rtol": xl["rtol"]}
    if "rtol_steps_differ" in xl:
        check["rtol_steps_differ"] = xl["rtol_steps_differ"]
    case = {"id": "m2." + m1["id"], "run_config": rc, "tspan": m1["tspan"]}
    if "saveat" in m1:
        case["saveat"] = m1["saveat"]
    case["layout"] = layout
    case["checks"] = [check]
    return case


def main():
    m1 = json.loads((HERE.parent / "m1" / "cases.json").read_text())
    cases = [c for c in (m2_case(c) for c in m1["cases"]) if c is not None]
    doc = {
        "esm_integrator": m1["esm_integrator"],
        "generated_by": "tests/conformance/m2/make_cases.py",
        "cases": cases,
    }
    (HERE / "cases.json").write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
