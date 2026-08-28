# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — whole-program frontend-contract quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

WHOLE_PROGRAM_FRONTEND_CONTRACTS_SOURCE = (
    "src/scpn_quantum_control/whole_program_frontend_contracts.py"
)
"""Production source owned by the whole-program frontend contracts."""

WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_COHORT = [
    "tests/test_whole_program_frontend.py",
    "tests/test_whole_program_frontend_contracts.py",
]
"""Tests that own exact whole-program frontend-contract coverage."""

WHOLE_PROGRAM_FRONTEND_CONTRACTS_TYPING_RATCHET = [
    WHOLE_PROGRAM_FRONTEND_CONTRACTS_SOURCE,
    "tools/whole_program_frontend_contracts_quality_gates.py",
    "tests/test_whole_program_frontend_contracts_quality_gate.py",
]
"""Strict-typing cohort for production and gate contracts."""

WHOLE_PROGRAM_FRONTEND_CONTRACTS_DOCSTRING_RATCHET = [
    WHOLE_PROGRAM_FRONTEND_CONTRACTS_SOURCE,
    *WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_COHORT,
    "tools/whole_program_frontend_contracts_quality_gates.py",
    "tests/test_whole_program_frontend_contracts_quality_gate.py",
]
"""Complete production, owner-test, and gate-contract docstring cohort."""

WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-whole-program-frontend-contracts-quality.coverage"  # nosec B108
)
"""Isolated coverage database for the frontend-contract owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-whole-program-frontend-contracts-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *WHOLE_PROGRAM_FRONTEND_CONTRACTS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D whole-program-frontend-contracts quality ratchet",
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
                *WHOLE_PROGRAM_FRONTEND_CONTRACTS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    return [
        (
            "whole-program-frontend-contracts focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_COHORT,
            ],
        ),
        (
            "whole-program-frontend-contracts exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/whole_program_frontend_contracts.py",
            ],
        ),
    ]


__all__ = [
    "WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_COHORT",
    "WHOLE_PROGRAM_FRONTEND_CONTRACTS_COVERAGE_DATA_FILE",
    "WHOLE_PROGRAM_FRONTEND_CONTRACTS_DOCSTRING_RATCHET",
    "WHOLE_PROGRAM_FRONTEND_CONTRACTS_SOURCE",
    "WHOLE_PROGRAM_FRONTEND_CONTRACTS_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
