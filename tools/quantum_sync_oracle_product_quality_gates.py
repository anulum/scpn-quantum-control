# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-sync oracle product quality-gate specification
"""Build strict documentation, typing, and coverage gates for BL-19."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

QUANTUM_SYNC_ORACLE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/quantum_sync_challenge_oracle_product.py",
    "tests/test_quantum_sync_challenge_oracle_product.py",
    "tools/quantum_sync_oracle_product_quality_gates.py",
    "tests/test_quantum_sync_oracle_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

QUANTUM_SYNC_ORACLE_COVERAGE_COHORT = ["tests/test_quantum_sync_challenge_oracle_product.py"]
"""Tests that own exact oracle-product statement and branch coverage."""

QUANTUM_SYNC_ORACLE_COVERAGE_DATA_FILE = ".coverage.quantum-sync-oracle-quality"
"""Isolated coverage database for the challenge-oracle source owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Ordered static gates for the product owner cohort.

    """
    return [
        (
            "mypy-strict-quantum-sync-oracle-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *QUANTUM_SYNC_ORACLE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D quantum-sync-oracle quality ratchet",
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
                *QUANTUM_SYNC_ORACLE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Focused owner execution followed by the exact report.

    """
    return [
        (
            "quantum-sync-oracle focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={QUANTUM_SYNC_ORACLE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *QUANTUM_SYNC_ORACLE_COVERAGE_COHORT,
            ],
        ),
        (
            "quantum-sync-oracle exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={QUANTUM_SYNC_ORACLE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/quantum_sync_challenge_oracle_product.py",
            ],
        ),
    ]


__all__ = [
    "QUANTUM_SYNC_ORACLE_COVERAGE_COHORT",
    "QUANTUM_SYNC_ORACLE_COVERAGE_DATA_FILE",
    "QUANTUM_SYNC_ORACLE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
