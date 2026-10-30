#!/usr/bin/env python3
"""The `cross_language` check (spec/toy-problems.md §2.1, conformance tier 3).

Compares the Julia and Rust result files (spec/toy-problems.md §3) of every
case in cases.json that has a `cross_language` check, and their tapes
(spec/tape.md §5) by path and count.

Usage: python tests/conformance/m1/compare_results.py JULIA_DIR RUST_DIR [--require-both]

A case for which neither implementation wrote a result is skipped (its methods
are not implemented yet), and a case only one of them wrote is reported; with
--require-both, either fails. A result with a missing field or a value that is not a
finite number fails. Exits 1 if any comparison fails, or if either directory
is absent.
"""

import argparse
import json
import math
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


def malformed(res):
    """What makes a result unusable for comparison: a missing field, or a
    value that is not a finite number. Either would otherwise slip through,
    since a comparison with NaN is never greater than rtol."""
    keys = ["t", "u", "stats"] + (["gradient"] if "functional" in res else [])
    missing = [k for k in keys if k not in res]
    if missing:
        return f"missing {', '.join(missing)}"
    values = [("t", res["t"]), ("u", [x for row in res["u"] for x in row])]
    if "functional" in res:
        values += [("functional", [res["functional"]])]
        values += [(f"gradient.{k}", v) for k, v in res["gradient"].items()]
    for name, xs in values:
        bad = [
            x
            for x in xs
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        ]
        if bad:
            return f"{name} has a value that is not a finite number: {bad[0]!r}"
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
        errs.append(
            f"state component {j} at t = {jl['t'][k]}: relative difference "
            f"{worst[0]:.3g} > rtol {rtol:g}"
        )
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
        errs.append(
            f"functional: julia {a!r}, rust {b!r}, relative difference {d:.3g} > rtol {rtol:g}"
        )
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
                errs.append(
                    f"gradient.{key}[{i}]: julia {x!r}, rust {y!r}, relative difference "
                    f"{d:.3g} > rtol {rtol:g}"
                )
    return errs, worst


def compare_tapes(jt, rt):
    """By path and count, not by value (spec/tape.md §5)."""
    if jt is None or rt is None:
        return [f"tape missing from {'julia' if jt is None else 'rust'}"]
    if len(jt) != len(rt):
        return [f"tapes have {len(jt)} (julia) and {len(rt)} (rust) records"]
    for a, b in zip(jt, rt):
        if (a["path"], a["node"]) != (b["path"], b["node"]):
            return [
                f"tape record differs: julia {a['path']} {a['node']}, rust {b['path']} {b['node']}"
            ]
        for key in ("accepted", "retcode"):
            if a.get(key) != b.get(key):
                return [f"tape record {a['path']}: {key} julia {a.get(key)!r}, rust {b.get(key)!r}"]
    return []


def steps_of(tape):
    """Every leaf record's accepted (t, h), in order."""
    return [r["steps"] for r in tape or [] if "steps" in r]


def compare_case(case_id, chk, jdir, rdir):
    jl, rs = load(jdir / f"{case_id}.json"), load(rdir / f"{case_id}.json")
    if jl is None and rs is None:
        return "skipped", "neither implementation ran it"
    if jl is None or rs is None:
        return "one-sided", f"only {'rust' if jl is None else 'julia'} ran it"
    bad = [
        f"{who} result: {m}" for who, res in (("julia", jl), ("rust", rs)) if (m := malformed(res))
    ]
    if bad:
        return "fail", "; ".join(bad)
    rtol = chk["rtol"]
    errs = []
    jt = load_tape(jdir / f"{case_id}.tape.jsonl")
    rt = load_tape(rdir / f"{case_id}.tape.jsonl")
    same_count = jl["stats"]["accepted"] == rs["stats"]["accepted"]
    if same_count:
        errs += compare_tapes(jt, rt)
    note = (
        f"accepted {jl['stats']['accepted']}"
        if same_count
        else f"accepted julia {jl['stats']['accepted']}, rust {rs['stats']['accepted']}"
    )
    # The step sequences differ when the counts do, or when any step's
    # (t, h) does: libm functions may differ in the last bit (spec/adjoint.md
    # §7.5), and an adaptive controller can turn that into different sizes.
    differ = not same_count or (not errs and steps_of(jt) != steps_of(rt))
    if differ:
        note += ", step sequences differ"
        if "rtol_steps_differ" in chk:
            rtol = chk["rtol_steps_differ"]
        else:
            errs.append("step sequences differ and the case has no rtol_steps_differ")
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
    ap.add_argument(
        "--require-both",
        action="store_true",
        help="fail a case that either implementation did not run",
    )
    args = ap.parse_args()
    # A suite that wrote no result files uploads no artifact, so CI's
    # download leaves the directory absent; say so rather than skip every case.
    absent = [
        f"{who} ({d})"
        for who, d in (("julia", args.julia_dir), ("rust", args.rust_dir))
        if not d.is_dir()
    ]
    if absent:
        print(f"no result files from {' or '.join(absent)}: did that test job run?")
        return 1
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    counts = {"pass": 0, "fail": 0, "skipped": 0, "one-sided": 0}
    failed = False
    for case in cases:
        chk = next((c for c in case["checks"] if c["check"] == "cross_language"), None)
        if chk is None:
            continue
        status, msg = compare_case(case["id"], chk, args.julia_dir, args.rust_dir)
        counts[status] += 1
        if status == "fail" or (status != "pass" and args.require_both):
            failed = True
        print(f"{status.upper():9} {case['id']}: {msg}")
    print(
        f"\ncross_language: {counts['pass']} passed, {counts['fail']} failed, "
        f"{counts['one-sided']} run by one implementation only, {counts['skipped']} skipped"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
