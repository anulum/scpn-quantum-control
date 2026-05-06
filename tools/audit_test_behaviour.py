# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — behavioural test audit helper
"""Inventory behavioural strength signals across pytest modules."""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

BEHAVIOURAL_CALLS = {
    "approx",
    "raises",
    "warns",
}

ASSERTION_CALL_PREFIXES = (
    "assert",
    "assert_",
)


@dataclass(frozen=True)
class TestFunctionAudit:
    """Behavioural signals for one test function."""

    name: str
    assertions: int
    raises_contracts: int
    parametrised: bool
    behavioural_calls: int

    @property
    def has_behavioural_contract(self) -> bool:
        """Return True when the test constrains behaviour beyond execution."""
        return self.assertions > 0 or self.raises_contracts > 0 or self.behavioural_calls > 0


@dataclass(frozen=True)
class TestModuleAudit:
    """Behavioural-test audit summary for one module."""

    path: str
    test_functions: tuple[TestFunctionAudit, ...]

    @property
    def test_count(self) -> int:
        """Number of test functions in the module."""
        return len(self.test_functions)

    @property
    def assertion_count(self) -> int:
        """Total explicit assertions in the module."""
        return sum(item.assertions for item in self.test_functions)

    @property
    def raises_contract_count(self) -> int:
        """Total pytest.raises contracts in the module."""
        return sum(item.raises_contracts for item in self.test_functions)

    @property
    def parametrised_count(self) -> int:
        """Number of parametrised tests in the module."""
        return sum(1 for item in self.test_functions if item.parametrised)

    @property
    def smoke_only_tests(self) -> tuple[str, ...]:
        """Tests that execute code but expose no local behavioural contract."""
        return tuple(
            item.name for item in self.test_functions if not item.has_behavioural_contract
        )


def _call_name(node: ast.AST) -> str:
    """Return the final dotted name component for a call expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_assertion_call(name: str) -> bool:
    """Return True when a call name is an assertion helper."""
    return any(name.startswith(prefix) for prefix in ASSERTION_CALL_PREFIXES)


def _is_pytest_mark_parametrize(decorator: ast.AST) -> bool:
    """Return True when a decorator is pytest.mark.parametrize."""
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return (
        isinstance(target, ast.Attribute)
        and target.attr == "parametrize"
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "mark"
        and isinstance(target.value.value, ast.Name)
        and target.value.value.id == "pytest"
    )


def _function_audit(node: ast.FunctionDef | ast.AsyncFunctionDef) -> TestFunctionAudit:
    """Audit one pytest function node."""
    assertions = 0
    raises_contracts = 0
    behavioural_calls = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            assertions += 1
        elif isinstance(child, ast.Call):
            name = _call_name(child.func)
            if _is_assertion_call(name):
                assertions += 1
            elif name == "raises":
                raises_contracts += 1
            if name in BEHAVIOURAL_CALLS:
                behavioural_calls += 1
    return TestFunctionAudit(
        name=node.name,
        assertions=assertions,
        raises_contracts=raises_contracts,
        parametrised=any(_is_pytest_mark_parametrize(item) for item in node.decorator_list),
        behavioural_calls=behavioural_calls,
    )


def audit_test_module(path: Path) -> TestModuleAudit:
    """Audit one Python test module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tests = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
            "test_"
        ):
            tests.append(_function_audit(node))
    return TestModuleAudit(path=str(path), test_functions=tuple(tests))


def audit_test_tree(root: Path) -> tuple[TestModuleAudit, ...]:
    """Audit all top-level pytest modules under a test root."""
    return tuple(
        audit_test_module(path) for path in sorted(root.glob("test_*.py")) if path.is_file()
    )


def _module_to_dict(module: TestModuleAudit) -> dict[str, object]:
    """Convert a module audit to JSON-compatible data."""
    return {
        "path": module.path,
        "test_count": module.test_count,
        "assertion_count": module.assertion_count,
        "raises_contract_count": module.raises_contract_count,
        "parametrised_count": module.parametrised_count,
        "smoke_only_tests": list(module.smoke_only_tests),
    }


def audits_to_json(audits: Sequence[TestModuleAudit]) -> str:
    """Serialise behavioural-test audits as deterministic JSON."""
    return json.dumps([_module_to_dict(item) for item in audits], indent=2, sort_keys=True)


def format_audits(audits: Iterable[TestModuleAudit]) -> str:
    """Render a compact behavioural-test audit summary."""
    modules = tuple(audits)
    total_tests = sum(item.test_count for item in modules)
    total_assertions = sum(item.assertion_count for item in modules)
    smoke_modules = [item for item in modules if item.smoke_only_tests]
    lines = [
        "Behavioural test audit summary:",
        f"- modules: {len(modules)}",
        f"- tests: {total_tests}",
        f"- assertions: {total_assertions}",
        f"- modules_with_smoke_only_tests: {len(smoke_modules)}",
    ]
    for module in smoke_modules[:20]:
        lines.append(f"- {module.path}: {', '.join(module.smoke_only_tests)}")
    if len(smoke_modules) > 20:
        lines.append(f"- additional_modules_with_smoke_only_tests: {len(smoke_modules) - 20}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests",
        help="Directory containing pytest modules.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--fail-on-smoke-only",
        action="store_true",
        help="Return non-zero when any test has no local behavioural contract.",
    )
    args = parser.parse_args(argv)

    audits = audit_test_tree(args.tests_root)
    print(audits_to_json(audits) if args.json else format_audits(audits))
    has_smoke_only = any(item.smoke_only_tests for item in audits)
    return 1 if args.fail_on_smoke_only and has_smoke_only else 0


if __name__ == "__main__":
    raise SystemExit(main())
