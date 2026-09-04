# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable audit-contract quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
DIFFERENTIABLE_AUDIT_CONTRACTS_SOURCE = (
    "src/scpn_quantum_control/phase/differentiable_audit_contracts.py"
)
DIFFERENTIABLE_AUDIT_SOURCE = "src/scpn_quantum_control/phase/differentiable_audit.py"
DIFFERENTIABLE_AUDIT_SOURCES = [
    DIFFERENTIABLE_AUDIT_CONTRACTS_SOURCE,
    DIFFERENTIABLE_AUDIT_SOURCE,
]
DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_COHORT = [
    "tests/test_phase_differentiable_audit.py",
    "tests/test_phase_differentiable_audit_contracts.py",
    "tests/test_phase_differentiable_audit_edges.py",
]
DIFFERENTIABLE_AUDIT_CONTRACTS_TYPING_RATCHET = [
    *DIFFERENTIABLE_AUDIT_SOURCES,
    "tests/test_phase_differentiable_audit_edges.py",
    "tools/differentiable_audit_contracts_quality_gates.py",
    "tests/test_differentiable_audit_contracts_quality_gate.py",
]
DIFFERENTIABLE_AUDIT_CONTRACTS_DOCSTRING_RATCHET = [
    *DIFFERENTIABLE_AUDIT_SOURCES,
    *DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_COHORT,
    "tools/differentiable_audit_contracts_quality_gates.py",
    "tests/test_differentiable_audit_contracts_quality_gate.py",
]
DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-differentiable-audit-contracts-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-differentiable-audit-contracts-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DIFFERENTIABLE_AUDIT_CONTRACTS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D differentiable-audit-contracts quality ratchet",
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
                *DIFFERENTIABLE_AUDIT_CONTRACTS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build connected execution and exact source coverage gates."""
    return [
        (
            "differentiable-audit-contracts focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_COHORT,
            ],
        ),
        (
            "differentiable-audit-contracts exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/differentiable_audit_contracts.py,"
                "*/phase/differentiable_audit.py",
            ],
        ),
    ]


__all__ = [
    "DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_COHORT",
    "DIFFERENTIABLE_AUDIT_CONTRACTS_COVERAGE_DATA_FILE",
    "DIFFERENTIABLE_AUDIT_CONTRACTS_DOCSTRING_RATCHET",
    "DIFFERENTIABLE_AUDIT_CONTRACTS_SOURCE",
    "DIFFERENTIABLE_AUDIT_CONTRACTS_TYPING_RATCHET",
    "DIFFERENTIABLE_AUDIT_SOURCE",
    "DIFFERENTIABLE_AUDIT_SOURCES",
    "build_coverage_gates",
    "build_static_quality_gates",
]
