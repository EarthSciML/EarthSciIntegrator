#!/usr/bin/env python3
"""The `cross_language` check (spec/toy-problems.md §2.1, conformance tier 3).

Compares the Julia and Rust result files (spec/toy-problems.md §3) of every
case in cases.json that has a `cross_language` check, and their tapes
(spec/tape.md §5) by path and count.

Usage: python tests/conformance/m1/compare_results.py JULIA_DIR RUST_DIR [--require-both]

A case for which neither implementation wrote a result is skipped (its methods
are not implemented yet). A case only one of them wrote is reported and, with
--require-both, fails. Exits 1 if any comparison fails.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def load_tape(path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except FileNotFoundError:
        return None


def compare_states(jl, rs, rtol):
    """Output states per component, relative to the component's largest
    magnitude over the run in either result, as the `reference` check."""
    errs = []
    if len(jl["t"]) != len(rs["t"]) or any(a != b for a, b in zip(jl["t"], rs["t"])):
        return [f"output times differ: julia {jl['t']}, rust {rs['t']}"], float("nan")
    ncomp = len(jl["u"][0])
    if any(len(x) != ncomp for x in jl["u"] + rs["u"]):
        return ["state lengths differ"], float("nan")
    worst = (0.0, None)
    for j in range(ncomp):
        scale = max(max(abs(x[j]) for x in jl["u"]), max(abs(x[j]) for x in rs["u"]))
        for k, (x, y) in enumerate(zip(jl["u"], rs["u"])):
            d = abs(x[j] - y[j]) / scale if scale > 0 else abs(x[j] - y[j])
            if d > worst[0]:
                worst = (d, (k, j))
    if worst[0] > rtol:
        k, j = worst[1]
        errs.append(f"state component {j} at t = {jl['t'][k]}: relative difference "
                    f"{worst[0]:.3g} > rtol {rtol:g}")
    return errs, worst[0]


def compare_gradient(jl, rs, rtol):
    """The functional relative to its magnitude; each gradient vector relative
    to its largest-magnitude entry in either result."""
    errs, worst = [], 0.0
    if ("functional" in jl) != ("functional" in rs):
        return ["only one result has a functional and gradient"], float("nan")
    if "functional" not in jl:
        return errs, worst
    a, b = jl["functional"], rs["functional"]
    scale = max(abs(a), abs(b))
    d = abs(a - b) / scale if scale > 0 else abs(a - b)
    worst = max(worst, d)
    if d > rtol:
        errs.append(f"functional: julia {a!r}, rust {b!r}, relative difference {d:.3g} > rtol {rtol:g}")
    for key in sorted(set(jl["gradient"]) | set(rs["gradient"])):
        ga, gb = jl["gradient"].get(key), rs["gradient"].get(key)
        if ga is None or gb is None or len(ga) != len(gb):
            errs.append(f"gradient.{key}: present or sized differently")
            continue
        scale = max((abs(x) for x in ga + gb), default=0.0)
        for i, (x, y) in enumerate(zip(ga, gb)):
            d = abs(x - y) / scale if scale > 0 else abs(x - y)
            worst = max(worst, d)
            if d > rtol:
                errs.append(f"gradient.{key}[{i}]: julia {x!r}, rust {y!r}, relative difference "
                            f"{d:.3g} > rtol {rtol:g}")
    return errs, worst


def compare_tapes(jt, rt):
    """By path and count, not by value (spec/tape.md §5)."""
    if jt is None or rt is None:
        return [f"tape missing from {'julia' if jt is None else 'rust'}"]
    if len(jt) != len(rt):
        return [f"tapes have {len(jt)} (julia) and {len(rt)} (rust) records"]
    for a, b in zip(jt, rt):
        if (a["path"], a["node"]) != (b["path"], b["node"]):
            return [f"tape record differs: julia {a['path']} {a['node']}, rust {b['path']} {b['node']}"]
        for key in ("accepted", "retcode"):
            if a.get(key) != b.get(key):
                return [f"tape record {a['path']}: {key} julia {a.get(key)!r}, rust {b.get(key)!r}"]
    return []


def compare_case(case_id, chk, jdir, rdir):
    jl, rs = load(jdir / f"{case_id}.json"), load(rdir / f"{case_id}.json")
    if jl is None and rs is None:
        return "skipped", "neither implementation ran it"
    if jl is None or rs is None:
        return "one-sided", f"only {'rust' if jl is None else 'julia'} ran it"
    rtol = chk["rtol"]
    errs = []
    same_steps = jl["stats"]["accepted"] == rs["stats"]["accepted"]
    note = f"accepted {jl['stats']['accepted']}"
    if not same_steps:
        note = f"accepted julia {jl['stats']['accepted']}, rust {rs['stats']['accepted']}"
        if "rtol_steps_differ" in chk:
            rtol = chk["rtol_steps_differ"]
        else:
            errs.append(f"accepted step counts differ ({note})")
    if same_steps:
        errs += compare_tapes(load_tape(jdir / f"{case_id}.tape.jsonl"),
                              load_tape(rdir / f"{case_id}.tape.jsonl"))
    s_errs, worst = compare_states(jl, rs, rtol)
    errs += s_errs
    g_errs, g_worst = compare_gradient(jl, rs, rtol)
    errs += g_errs
    note += f"; states {worst:.2g}"
    if "functional" in jl:
        note += f", gradient {g_worst:.2g}"
    note += f" (rtol {rtol:g})"
    return ("fail", "; ".join(errs) + f" [{note}]") if errs else ("pass", note)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("julia_dir", type=Path)
    ap.add_argument("rust_dir", type=Path)
    ap.add_argument("--require-both", action="store_true",
                    help="fail a case only one implementation ran")
    args = ap.parse_args()
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    counts = {"pass": 0, "fail": 0, "skipped": 0, "one-sided": 0}
    failed = False
    for case in cases:
        chk = next((c for c in case["checks"] if c["check"] == "cross_language"), None)
        if chk is None:
            continue
        status, msg = compare_case(case["id"], chk, args.julia_dir, args.rust_dir)
        counts[status] += 1
        if status == "fail" or (status == "one-sided" and args.require_both):
            failed = True
        print(f"{status.upper():9} {case['id']}: {msg}")
    print(f"\ncross_language: {counts['pass']} passed, {counts['fail']} failed, "
          f"{counts['one-sided']} run by one implementation only, {counts['skipped']} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
