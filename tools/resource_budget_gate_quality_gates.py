# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — resource-budget quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
RESOURCE_BUDGET_GATE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/compile_budget.py",
    "src/scpn_quantum_control/resource_budget_gate.py",
    "tests/test_compile_budget.py",
    "tests/test_resource_budget_gate.py",
    "tools/resource_budget_gate_quality_gates.py",
    "tests/test_resource_budget_gate_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
RESOURCE_BUDGET_GATE_COVERAGE_COHORT = [
    "tests/test_compile_budget.py",
    "tests/test_resource_budget_gate.py",
]
"""Tests that own exact resource-budget gate coverage."""
RESOURCE_BUDGET_GATE_COVERAGE_DATA_FILE = "/tmp/scpn-qc-resource-budget-gate-quality.coverage"  # nosec B108
"""Isolated coverage database for the resource-budget gate owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-resource-budget-gate-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *RESOURCE_BUDGET_GATE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D resource-budget-gate quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "D,D413,D417,D420",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *RESOURCE_BUDGET_GATE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "resource-budget-gate focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={RESOURCE_BUDGET_GATE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *RESOURCE_BUDGET_GATE_COVERAGE_COHORT,
            ],
        ),
        (
            "resource-budget-gate exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={RESOURCE_BUDGET_GATE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/compile_budget.py,*/resource_budget_gate.py",
            ],
        ),
    ]


__all__ = [
    "RESOURCE_BUDGET_GATE_COVERAGE_COHORT",
    "RESOURCE_BUDGET_GATE_COVERAGE_DATA_FILE",
    "RESOURCE_BUDGET_GATE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
