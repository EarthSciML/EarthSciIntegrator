#!/usr/bin/env python3
"""Conformance tier 0: the JSON Schemas under spec/schemas are valid, every
fixture under */valid validates, and every fixture under */invalid fails.

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

# Fixture directory -> schema file that governs it.
SUITES = {
    "run-configs": "run-config.schema.json",
    "plans": "plan.schema.json",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    schemas = {p.name: load(p) for p in sorted(SCHEMAS.glob("*.schema.json"))}
    registry = Registry().with_resources(
        (s["$id"], Resource.from_contents(s)) for s in schemas.values()
    )
    failures = []
    checked = 0

    for name, schema in schemas.items():
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as e:  # noqa: BLE001 - report every schema error
            failures.append(f"{name}: not a valid 2020-12 schema: {e}")

    for suite, schema_name in SUITES.items():
        validator = Draft202012Validator(schemas[schema_name], registry=registry)
        for expect_valid in (True, False):
            folder = FIXTURES / suite / ("valid" if expect_valid else "invalid")
            for path in sorted(folder.glob("*.json")):
                checked += 1
                errors = list(validator.iter_errors(load(path)))
                rel = path.relative_to(ROOT)
                if expect_valid and errors:
                    best = min(errors, key=lambda e: len(list(e.absolute_path)))
                    failures.append(f"{rel}: should validate, but {best.json_path}: {best.message}")
                elif not expect_valid and not errors:
                    failures.append(f"{rel}: should fail validation, but passed")

    for f in failures:
        print("FAIL", f)
    print(f"{checked} fixtures against {len(schemas)} schemas: {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
