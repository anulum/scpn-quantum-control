# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — mock and stub audit helper
"""Inventory mock, fake, stub, and monkeypatch usage in pytest modules."""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

MOCK_CALL_NAMES = {
    "AsyncMock",
    "MagicMock",
    "Mock",
    "PropertyMock",
    "mock_open",
    "patch",
}

STUB_NAME_PREFIXES = (
    "dummy",
    "fake",
    "mock",
    "stub",
)

BOUNDARY_TERMS = (
    "backend",
    "executor",
    "github",
    "ibm",
    "mitiq",
    "qiskit",
    "requests",
    "subprocess",
    "sys.modules",
)

SCIENTIFIC_RESULT_TERMS = (
    "counts",
    "energy",
    "expectation",
    "fidelity",
    "job_id",
    "raw",
    "result",
    "shots",
)


@dataclass(frozen=True)
class MockStubFinding:
    """One mock/stub usage finding."""

    path: str
    line: int
    kind: str
    name: str
    context: str

    @property
    def appears_third_party_boundary(self) -> bool:
        """Return True when the finding appears to mock an external boundary."""
        lowered = self.context.lower()
        return any(term in lowered for term in BOUNDARY_TERMS)

    @property
    def touches_scientific_result_terms(self) -> bool:
        """Return True when the finding mentions result-like scientific data."""
        lowered = self.context.lower()
        return any(term in lowered for term in SCIENTIFIC_RESULT_TERMS)


def _call_name(node: ast.AST) -> str:
    """Return the final dotted name component for a call expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _full_attr_name(node: ast.AST) -> str:
    """Return a dotted attribute/name expression when statically available."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _full_attr_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _is_stub_name(name: str) -> bool:
    """Return True when a symbol name indicates a fake/mock/stub helper."""
    lowered = name.lower()
    return any(lowered.startswith(prefix) for prefix in STUB_NAME_PREFIXES)


def _source_segment(source: str, node: ast.AST) -> str:
    """Return compact source text for a node."""
    segment = ast.get_source_segment(source, node) or ""
    return " ".join(segment.split())


def audit_mock_stub_file(path: Path) -> tuple[MockStubFinding, ...]:
    """Audit one test module for mock/stub usage."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    findings: list[MockStubFinding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            full_name = _full_attr_name(node.func)
            if name in MOCK_CALL_NAMES or full_name.startswith("monkeypatch."):
                findings.append(
                    MockStubFinding(
                        path=str(path),
                        line=node.lineno,
                        kind="call",
                        name=full_name or name,
                        context=_source_segment(source, node),
                    )
                )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if _is_stub_name(node.name):
                findings.append(
                    MockStubFinding(
                        path=str(path),
                        line=node.lineno,
                        kind=type(node).__name__,
                        name=node.name,
                        context=_source_segment(source, node).split(":")[0],
                    )
                )
        elif isinstance(node, ast.arg) and _is_stub_name(node.arg):
            findings.append(
                MockStubFinding(
                    path=str(path),
                    line=node.lineno,
                    kind="arg",
                    name=node.arg,
                    context=node.arg,
                )
            )
    return tuple(sorted(findings, key=lambda item: (item.path, item.line, item.name)))


def audit_mock_stub_tree(root: Path) -> tuple[MockStubFinding, ...]:
    """Audit all top-level pytest modules under a test root."""
    findings: list[MockStubFinding] = []
    for path in sorted(root.glob("test_*.py")):
        if path.is_file():
            findings.extend(audit_mock_stub_file(path))
    return tuple(findings)


def _finding_to_dict(finding: MockStubFinding) -> dict[str, object]:
    """Convert one finding to JSON-compatible data."""
    return {
        "path": finding.path,
        "line": finding.line,
        "kind": finding.kind,
        "name": finding.name,
        "appears_third_party_boundary": finding.appears_third_party_boundary,
        "touches_scientific_result_terms": finding.touches_scientific_result_terms,
        "context": finding.context,
    }


def findings_to_json(findings: Sequence[MockStubFinding]) -> str:
    """Serialise findings as deterministic JSON."""
    return json.dumps(
        [_finding_to_dict(item) for item in findings],
        indent=2,
        sort_keys=True,
    )


def format_findings(findings: Iterable[MockStubFinding]) -> str:
    """Render a compact mock/stub audit summary."""
    items = tuple(findings)
    boundary_count = sum(1 for item in items if item.appears_third_party_boundary)
    result_term_count = sum(1 for item in items if item.touches_scientific_result_terms)
    lines = [
        "Mock/stub audit summary:",
        f"- findings: {len(items)}",
        f"- apparent_third_party_boundaries: {boundary_count}",
        f"- result_term_mentions_requiring_review: {result_term_count}",
    ]
    for item in items[:30]:
        flags = []
        if item.appears_third_party_boundary:
            flags.append("boundary")
        if item.touches_scientific_result_terms:
            flags.append("result-term")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        lines.append(f"- {item.path}:{item.line} {item.name}{suffix}")
    if len(items) > 30:
        lines.append(f"- additional_findings: {len(items) - 30}")
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
        "--fail-on-result-terms",
        action="store_true",
        help="Return non-zero when any mock/stub context mentions result-like terms.",
    )
    args = parser.parse_args(argv)

    findings = audit_mock_stub_tree(args.tests_root)
    print(findings_to_json(findings) if args.json else format_findings(findings))
    has_result_terms = any(item.touches_scientific_result_terms for item in findings)
    return 1 if args.fail_on_result_terms and has_result_terms else 0


if __name__ == "__main__":
    raise SystemExit(main())
