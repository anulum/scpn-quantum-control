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
    "_assert",
    "_assert_",
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


@dataclass(frozen=True)
class BehaviourQualityGate:
    """Aggregate behavioural-quality gate for a test audit run."""

    total_modules: int
    total_tests: int
    total_assertions: int
    total_raises_contracts: int
    smoke_only_tests: int
    assertion_density: float
    raises_contract_density: float
    min_assertion_density: float
    min_raises_contract_density: float
    valid: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialise the quality gate."""
        return {
            "total_modules": self.total_modules,
            "total_tests": self.total_tests,
            "total_assertions": self.total_assertions,
            "total_raises_contracts": self.total_raises_contracts,
            "smoke_only_tests": self.smoke_only_tests,
            "assertion_density": self.assertion_density,
            "raises_contract_density": self.raises_contract_density,
            "min_assertion_density": self.min_assertion_density,
            "min_raises_contract_density": self.min_raises_contract_density,
            "valid": self.valid,
            "blockers": list(self.blockers),
        }


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
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for member in node.body:
                if isinstance(
                    member, ast.FunctionDef | ast.AsyncFunctionDef
                ) and member.name.startswith("test_"):
                    tests.append(_function_audit(member))
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


def evaluate_quality_gate(
    audits: Sequence[TestModuleAudit],
    *,
    min_assertion_density: float = 0.0,
    min_raises_contract_density: float = 0.0,
) -> BehaviourQualityGate:
    """Evaluate aggregate behavioural quality thresholds."""
    if min_assertion_density < 0.0:
        raise ValueError("min_assertion_density must be non-negative.")
    if min_raises_contract_density < 0.0:
        raise ValueError("min_raises_contract_density must be non-negative.")

    total_tests = sum(item.test_count for item in audits)
    total_assertions = sum(item.assertion_count for item in audits)
    total_raises = sum(item.raises_contract_count for item in audits)
    smoke_only = sum(len(item.smoke_only_tests) for item in audits)
    assertion_density = total_assertions / total_tests if total_tests else 0.0
    raises_density = total_raises / total_tests if total_tests else 0.0
    blockers: list[str] = []
    if smoke_only:
        blockers.append(f"{smoke_only} tests lack a local behavioural contract")
    if assertion_density < min_assertion_density:
        blockers.append(
            f"assertion density {assertion_density:.6f} below minimum {min_assertion_density:.6f}"
        )
    if raises_density < min_raises_contract_density:
        blockers.append(
            f"raises-contract density {raises_density:.6f} below minimum "
            f"{min_raises_contract_density:.6f}"
        )
    return BehaviourQualityGate(
        total_modules=len(audits),
        total_tests=total_tests,
        total_assertions=total_assertions,
        total_raises_contracts=total_raises,
        smoke_only_tests=smoke_only,
        assertion_density=assertion_density,
        raises_contract_density=raises_density,
        min_assertion_density=min_assertion_density,
        min_raises_contract_density=min_raises_contract_density,
        valid=not blockers,
        blockers=tuple(blockers),
    )


def format_audits(
    audits: Iterable[TestModuleAudit],
    *,
    quality_gate: BehaviourQualityGate | None = None,
) -> str:
    """Render a compact behavioural-test audit summary."""
    modules = tuple(audits)
    total_tests = sum(item.test_count for item in modules)
    total_assertions = sum(item.assertion_count for item in modules)
    total_raises = sum(item.raises_contract_count for item in modules)
    smoke_modules = [item for item in modules if item.smoke_only_tests]
    lines = [
        "Behavioural test audit summary:",
        f"- modules: {len(modules)}",
        f"- tests: {total_tests}",
        f"- assertions: {total_assertions}",
        f"- raises_contracts: {total_raises}",
        f"- modules_with_smoke_only_tests: {len(smoke_modules)}",
    ]
    for module in smoke_modules[:20]:
        lines.append(f"- {module.path}: {', '.join(module.smoke_only_tests)}")
    if len(smoke_modules) > 20:
        lines.append(f"- additional_modules_with_smoke_only_tests: {len(smoke_modules) - 20}")
    if quality_gate is not None:
        lines.extend(
            [
                f"- quality_gate_valid: {quality_gate.valid}",
                f"- assertion_density: {quality_gate.assertion_density:.6f}",
                f"- raises_contract_density: {quality_gate.raises_contract_density:.6f}",
            ]
        )
        for blocker in quality_gate.blockers:
            lines.append(f"- quality_gate_blocker: {blocker}")
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
    parser.add_argument(
        "--min-assertion-density",
        type=float,
        default=0.0,
        help="Minimum explicit assertion-helper density per discovered test.",
    )
    parser.add_argument(
        "--min-raises-contract-density",
        type=float,
        default=0.0,
        help="Minimum pytest.raises contract density per discovered test.",
    )
    parser.add_argument(
        "--fail-on-quality-gate",
        action="store_true",
        help="Return non-zero when aggregate behavioural-quality thresholds fail.",
    )
    args = parser.parse_args(argv)

    audits = audit_test_tree(args.tests_root)
    quality_gate = evaluate_quality_gate(
        audits,
        min_assertion_density=args.min_assertion_density,
        min_raises_contract_density=args.min_raises_contract_density,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "modules": [_module_to_dict(item) for item in audits],
                    "quality_gate": quality_gate.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(format_audits(audits, quality_gate=quality_gate))
    has_smoke_only = any(item.smoke_only_tests for item in audits)
    if args.fail_on_smoke_only and has_smoke_only:
        return 1
    return 1 if args.fail_on_quality_gate and not quality_gate.valid else 0


if __name__ == "__main__":
    raise SystemExit(main())
