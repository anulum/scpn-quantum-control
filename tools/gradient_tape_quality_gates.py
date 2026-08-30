# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase tape quality gates
"""Build strict documentation, typing, and exact gradient/QNode tape coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
GRADIENT_TAPE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/gradient_tape.py",
    "src/scpn_quantum_control/phase/qnode_tape.py",
    "tests/test_phase_gradient_tape.py",
    "tests/test_phase_qnode_tape.py",
    "tools/gradient_tape_quality_gates.py",
    "tests/test_gradient_tape_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
GRADIENT_TAPE_COVERAGE_COHORT = [
    "tests/test_phase_gradient_tape.py",
    "tests/test_phase_qnode_tape.py",
]
"""Tests that own exact phase gradient and QNode tape coverage."""
GRADIENT_TAPE_COVERAGE_DATA_FILE = "/tmp/scpn-qc-gradient-tape.coverage"  # nosec B108
"""Isolated coverage database for phase gradient and QNode tapes."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-gradient-tape-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *GRADIENT_TAPE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D gradient-tape quality ratchet",
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
                *GRADIENT_TAPE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "gradient-tape focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={GRADIENT_TAPE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *GRADIENT_TAPE_COVERAGE_COHORT,
            ],
        ),
        (
            "gradient-tape exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={GRADIENT_TAPE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/gradient_tape.py,*/phase/qnode_tape.py",
            ],
        ),
    ]


__all__ = [
    "GRADIENT_TAPE_COVERAGE_COHORT",
    "GRADIENT_TAPE_COVERAGE_DATA_FILE",
    "GRADIENT_TAPE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
