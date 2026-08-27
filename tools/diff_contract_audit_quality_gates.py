# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable-circuit audit quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

DIFF_CONTRACT_AUDIT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/diff_contract_audit.py",
    "tests/test_diff_namespace.py",
    "tools/diff_contract_audit_quality_gates.py",
    "tests/test_diff_contract_audit_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

DIFF_CONTRACT_AUDIT_COVERAGE_COHORT = [
    "tests/test_diff_namespace.py",
]
"""Tests that own exact differentiable-circuit audit coverage."""

DIFF_CONTRACT_AUDIT_COVERAGE_DATA_FILE = "/tmp/scpn-qc-diff-contract-audit-quality.coverage"  # nosec B108
"""Isolated coverage database for the differentiable-circuit audit owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-diff-contract-audit-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DIFF_CONTRACT_AUDIT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D diff-contract-audit quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--select",
                "D,D413",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *DIFF_CONTRACT_AUDIT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "diff-contract-audit focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DIFF_CONTRACT_AUDIT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DIFF_CONTRACT_AUDIT_COVERAGE_COHORT,
            ],
        ),
        (
            "diff-contract-audit exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DIFF_CONTRACT_AUDIT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/diff_contract_audit.py",
            ],
        ),
    ]


__all__ = [
    "DIFF_CONTRACT_AUDIT_COVERAGE_COHORT",
    "DIFF_CONTRACT_AUDIT_COVERAGE_DATA_FILE",
    "DIFF_CONTRACT_AUDIT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
