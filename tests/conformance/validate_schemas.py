#!/usr/bin/env python3
"""Conformance tier 0: the JSON Schemas under spec/schemas are valid, every
fixture under */valid validates, and every fixture under */invalid fails for
the reason its entry in that directory's expected-errors.json names.

An invalid fixture's entry gives the JSON Schema keyword that must fail and
the instance locations ("at") of its one defect. The fixture passes when some
error with that keyword is reported at one of those locations, and every other error lies at or inside
one of them. The one exception is the errors of the branches of a oneOf or
anyOf, which may also lie on the path to one of them, because a branch for
another node kind fails at the node itself. An error anywhere else means the
fixture has a second defect.

Usage: python tests/conformance/validate_schemas.py
Needs jsonschema >= 4.18 (for the `referencing` registry).
"""

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "spec" / "schemas"
FIXTURES = ROOT / "tests" / "conformance"
EXPECTED = "expected-errors.json"

# Fixture directory -> schema file that governs it.
SUITES = {
    "run-configs": "run-config.schema.json",
    "plans": "plan.schema.json",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def leaf_errors(errors, in_branch=False):
    """The innermost errors, each with whether it came from a oneOf/anyOf branch."""
    for e in errors:
        if e.context:
            yield from leaf_errors(e.context, True)
        else:
            yield e, in_branch


def within(inner, outer):
    """True when JSON path `inner` is `outer` or inside it."""
    return inner == outer or (inner.startswith(outer) and inner[len(outer)] in ".[")


def check_invalid(errors, expected):
    """None if `errors` show exactly the expected defect, else what went wrong."""
    if not errors:
        return "should fail validation, but passed"
    leaves = list(leaf_errors(errors))
    at = expected["at"]
    if not any(e.validator == expected["keyword"] and e.json_path in at for e, _ in leaves):
        got = sorted({f"{e.json_path} ({e.validator})" for e, _ in leaves})
        return f"should fail with '{expected['keyword']}' at {', '.join(at)}, but failed with {'; '.join(got)}"

    def explained(e, in_branch):
        return any(within(e.json_path, a) or (in_branch and within(a, e.json_path)) for a in at)

    stray = sorted({f"{e.json_path} ({e.validator}): {e.message}"
                    for e, in_branch in leaves if not explained(e, in_branch)})
    if stray:
        return "fails for a second reason: " + "; ".join(stray)
    return None


def main():
    schemas = {p.name: load(p) for p in sorted(SCHEMAS.glob("*.schema.json"))}
    registry = Registry().with_resources(
        (s["$id"], Resource.from_contents(s)) for s in schemas.values()
    )
    failures = []
    checked = 0

    for name in SUITES.values():
        if name not in schemas:
            failures.append(f"spec/schemas/{name}: missing")
    for name, schema in schemas.items():
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as e:  # noqa: BLE001 - report every schema error
            failures.append(f"{name}: not a valid 2020-12 schema: {e}")

    for suite, schema_name in SUITES.items():
        if schema_name not in schemas:
            continue
        validator = Draft202012Validator(schemas[schema_name], registry=registry)
        for expect_valid in (True, False):
            folder = FIXTURES / suite / ("valid" if expect_valid else "invalid")
            paths = sorted(p for p in folder.glob("*.json") if p.name != EXPECTED)
            if not paths:
                failures.append(f"{folder.relative_to(ROOT)}: missing or has no fixtures")
                continue
            expected = {} if expect_valid else load(folder / EXPECTED)
            if not expect_valid:
                for name in sorted(set(expected) - {p.name for p in paths}):
                    failures.append(f"{(folder / EXPECTED).relative_to(ROOT)}: entry '{name}' has no fixture")
            for path in paths:
                checked += 1
                errors = list(validator.iter_errors(load(path)))
                rel = path.relative_to(ROOT)
                if expect_valid:
                    if errors:
                        best = min(errors, key=lambda e: len(list(e.absolute_path)))
                        failures.append(f"{rel}: should validate, but {best.json_path}: {best.message}")
                elif path.name not in expected:
                    failures.append(f"{rel}: no entry in {EXPECTED}")
                else:
                    problem = check_invalid(errors, expected[path.name])
                    if problem:
                        failures.append(f"{rel}: {problem}")

    for f in failures:
        print("FAIL", f)
    print(f"{checked} fixtures against {len(schemas)} schemas: {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
