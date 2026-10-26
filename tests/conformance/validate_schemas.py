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

It also checks the M2 planner fixtures' structure (spec/plan.md §9,
tests/conformance/README.md): every case run config validates and names a
model that exists, every case has a section in its fixture's expected.md,
every refusal fixture's code is listed in spec/planner-errors.json, hand-written
expected plans are canonical JSON, and every .esm document under
tests/conformance loads, validates and flattens with EarthSciAST 0.4.0
(the earthsci-ast Python package; with --require-esm a missing package fails).
The spec targets EarthSciAST 0.5.0 (spec/plan.md §0.2); ESM_VERSION and the CI
pin move to 0.5.0 once it is on PyPI.

Usage: python tests/conformance/validate_schemas.py [--require-esm]
Needs jsonschema >= 4.18 (for the `referencing` registry).
"""

import json
from decimal import Decimal
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

# Spec and conformance data files that must validate: glob under the repository
# root -> schema file. A glob that matches nothing is a failure only when
# marked required.
DOCUMENTS = [
    ("tests/conformance/m1/cases.json", "cases.schema.json", True),
    ("tests/conformance/m2/cases.json", "cases.schema.json", True),
    ("spec/tableaus/*.json", "tableau.schema.json", False),
    ("tests/conformance/m1/reference/*.json", "result.schema.json", False),
    ("spec/planner-errors.json", "planner-errors.schema.json", True),
    ("tests/conformance/plans/fixtures/*/*.run.json", "run-config.schema.json", True),
    ("tests/conformance/plans/refusals/*/run.json", "run-config.schema.json", True),
    ("tests/conformance/plans/fixtures/*/*.expected-plan.json", "plan.schema.json", False),
    ("tests/conformance/plans/fixtures/*/*.golden/plan.json", "plan.schema.json", False),
]
PLAN_FIXTURES = FIXTURES / "plans" / "fixtures"
REFUSALS = FIXTURES / "plans" / "refusals"
ESM_VERSION = "0.4.0"  # EarthSciAST, esm 1.3.0 (spec/plan.md §0.2)


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


def jcs(value):
    """RFC 8785 canonical JSON (spec/plan.md §2.1)."""
    if isinstance(value, dict):
        keys = sorted(value, key=lambda k: k.encode("utf-16-be"))
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + jcs(value[k]) for k in keys) + "}"
    if isinstance(value, list):
        return "[" + ",".join(jcs(v) for v in value) + "]"
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    return jcs_number(value)


def jcs_number(x):
    """ECMAScript Number.prototype.toString for a finite float."""
    if x == 0:
        return "0"
    if x == int(x) and abs(x) < 1e21:
        return str(int(x))
    d = Decimal(repr(x))
    digits = "".join(map(str, d.as_tuple().digits)).rstrip("0") or "0"
    k, n = len(digits), d.adjusted() + 1
    sign = "-" if x < 0 else ""
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    e = n - 1
    exp = ("+" if e > 0 else "-") + str(abs(e))
    return sign + digits[0] + ("." + digits[1:] if k > 1 else "") + "e" + exp


def pointer_parent_resolves(doc, pointer):
    """True when every token of the JSON pointer but the last resolves in doc."""
    if pointer == "":
        return True
    if not pointer.startswith("/"):
        return False
    node = doc
    for token in pointer[1:].split("/")[:-1]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        elif isinstance(node, dict) and token in node:
            node = node[token]
        else:
            return False
    return True


def check_planner_fixtures(failures):
    """Structure of the M2 planner fixtures (spec/plan.md §9). Returns files checked."""
    checked = 0
    codes = [c["code"] for c in load(ROOT / "spec" / "planner-errors.json")["codes"]]
    if len(codes) != len(set(codes)):
        failures.append("spec/planner-errors.json: duplicate codes")
    for folder in sorted(p for p in PLAN_FIXTURES.iterdir() if p.is_dir()):
        rel = folder.relative_to(ROOT)
        expected = folder / "expected.md"
        text = expected.read_text(encoding="utf-8") if expected.exists() else None
        if text is None:
            failures.append(f"{rel}: no expected.md")
        cases = sorted(folder.glob("*.run.json"))
        if not cases:
            failures.append(f"{rel}: no <case>.run.json")
        for case in cases:
            checked += 1
            name = case.name[: -len(".run.json")]
            if text is not None and f"\n## {name}\n" not in text:
                failures.append(f"{rel}/expected.md: no section '## {name}'")
            model = load(case)["model"]
            if not (case.parent / model).is_file():
                failures.append(f"{case.relative_to(ROOT)}: model '{model}' does not exist")
        for plan in sorted(folder.glob("*.expected-plan.json")):
            checked += 1
            raw = plan.read_text(encoding="utf-8")
            if jcs(json.loads(raw)) != raw:
                failures.append(f"{plan.relative_to(ROOT)}: not RFC 8785 canonical JSON")
    expected_codes = load(REFUSALS / "expected-codes.json")
    dirs = {p.name for p in REFUSALS.iterdir() if p.is_dir()}
    for name in sorted(dirs - set(expected_codes)):
        failures.append(f"{(REFUSALS / name).relative_to(ROOT)}: no entry in expected-codes.json")
    for name, entry in sorted(expected_codes.items()):
        checked += 1
        run = REFUSALS / name / "run.json"
        if name not in dirs or not run.is_file():
            failures.append(f"tests/conformance/plans/refusals/expected-codes.json: '{name}' has no run.json")
            continue
        if entry["code"] not in codes:
            failures.append(f"{run.relative_to(ROOT)}: code '{entry['code']}' is not in spec/planner-errors.json")
        if not pointer_parent_resolves(load(run), entry["at"]):
            failures.append(f"{run.relative_to(ROOT)}: pointer '{entry['at']}' does not lead into the run config")
        model = load(run)["model"]
        if not model.startswith("toy:") and not (run.parent / model).is_file():
            failures.append(f"{run.relative_to(ROOT)}: model '{model}' does not exist")
    return checked


def check_esm_documents(failures, require):
    """Every .esm under tests/conformance loads, validates and flattens with EarthSciAST."""
    paths = sorted(FIXTURES.rglob("*.esm"))
    try:
        import earthsci_ast
    except ImportError:
        if require:
            failures.append(f"earthsci_ast is not installed (pip install earthsci-ast=={ESM_VERSION})")
        else:
            print(f"SKIP {len(paths)} .esm documents: earthsci_ast is not installed")
        return 0
    version = getattr(earthsci_ast, "__version__", "?")
    if version != ESM_VERSION:
        message = f"earthsci_ast {version} is installed; the fixtures are pinned to {ESM_VERSION}"
        if require:
            failures.append(message)
        else:
            print(f"SKIP {len(paths)} .esm documents: {message}")
        return 0
    for path in paths:
        rel = path.relative_to(ROOT)
        try:
            result = earthsci_ast.validate_path(str(path))
            if not result.is_valid:
                failures.append(f"{rel}: invalid: {result.schema_errors or result.structural_errors}")
                continue
            earthsci_ast.flatten(earthsci_ast.load_path(str(path)))
        except Exception as e:  # noqa: BLE001 - report every load failure
            failures.append(f"{rel}: {type(e).__name__}: {str(e)[:300]}")
    return len(paths)


def main():
    require_esm = "--require-esm" in sys.argv[1:]
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

    for pattern, schema_name, required in DOCUMENTS:
        paths = sorted(ROOT.glob(pattern))
        if required and not paths:
            failures.append(f"{pattern}: missing")
        if not paths:
            continue
        if schema_name not in schemas:
            failures.append(f"spec/schemas/{schema_name}: missing (needed by {pattern})")
            continue
        validator = Draft202012Validator(schemas[schema_name], registry=registry)
        for path in paths:
            checked += 1
            errors = list(validator.iter_errors(load(path)))
            if errors:
                best = min(errors, key=lambda e: len(list(e.absolute_path)))
                failures.append(f"{path.relative_to(ROOT)}: should validate, but {best.json_path}: {best.message}")

    checked += check_planner_fixtures(failures)
    checked += check_esm_documents(failures, require_esm)

    for f in failures:
        print("FAIL", f)
    print(f"{checked} files against {len(schemas)} schemas: {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
