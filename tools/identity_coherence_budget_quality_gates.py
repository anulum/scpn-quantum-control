# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — identity coherence-budget quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

IDENTITY_COHERENCE_BUDGET_QUALITY_RATCHET = [
    "src/scpn_quantum_control/identity/coherence_budget.py",
    "tests/test_identity_coherence_budget.py",
    "tools/identity_coherence_budget_quality_gates.py",
    "tests/test_identity_coherence_budget_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

IDENTITY_COHERENCE_BUDGET_COVERAGE_COHORT = [
    "tests/test_identity_coherence_budget.py",
]
"""Tests that own exact identity coherence-budget coverage."""

IDENTITY_COHERENCE_BUDGET_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-identity-coherence-budget-quality.coverage"  # nosec B108
)
"""Isolated coverage database for the identity coherence-budget owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-identity-coherence-budget-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *IDENTITY_COHERENCE_BUDGET_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D identity-coherence-budget quality ratchet",
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
                *IDENTITY_COHERENCE_BUDGET_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "identity-coherence-budget focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={IDENTITY_COHERENCE_BUDGET_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *IDENTITY_COHERENCE_BUDGET_COVERAGE_COHORT,
            ],
        ),
        (
            "identity-coherence-budget exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={IDENTITY_COHERENCE_BUDGET_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/identity/coherence_budget.py",
            ],
        ),
    ]


__all__ = [
    "IDENTITY_COHERENCE_BUDGET_COVERAGE_COHORT",
    "IDENTITY_COHERENCE_BUDGET_COVERAGE_DATA_FILE",
    "IDENTITY_COHERENCE_BUDGET_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
