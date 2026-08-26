# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase trainability quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PHASE_TRAINABILITY_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/trainability.py",
    "tests/test_phase_trainability.py",
    "tools/phase_trainability_quality_gates.py",
    "tests/test_phase_trainability_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
PHASE_TRAINABILITY_COVERAGE_COHORT = ["tests/test_phase_trainability.py"]
"""Tests that own exact phase-trainability coverage."""
PHASE_TRAINABILITY_COVERAGE_DATA_FILE = ".coverage.phase-trainability-quality"
"""Isolated coverage database for phase-trainability diagnostics."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-phase-trainability-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PHASE_TRAINABILITY_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D phase-trainability quality ratchet",
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
                *PHASE_TRAINABILITY_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    data = PHASE_TRAINABILITY_COVERAGE_DATA_FILE
    return [
        (
            "phase-trainability focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PHASE_TRAINABILITY_COVERAGE_COHORT,
            ],
        ),
        (
            "phase-trainability exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/trainability.py",
            ],
        ),
    ]


__all__ = [
    "PHASE_TRAINABILITY_COVERAGE_COHORT",
    "PHASE_TRAINABILITY_COVERAGE_DATA_FILE",
    "PHASE_TRAINABILITY_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
