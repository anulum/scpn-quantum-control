# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — finite-size scaling quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
FINITE_SIZE_SCALING_SOURCE = "src/scpn_quantum_control/analysis/finite_size_scaling.py"
FINITE_SIZE_SCALING_COVERAGE_COHORT = ["tests/test_finite_size_scaling.py"]
FINITE_SIZE_SCALING_TYPING_RATCHET = [
    FINITE_SIZE_SCALING_SOURCE,
    *FINITE_SIZE_SCALING_COVERAGE_COHORT,
    "tools/finite_size_scaling_quality_gates.py",
    "tests/test_finite_size_scaling_quality_gate.py",
]
FINITE_SIZE_SCALING_DOCSTRING_RATCHET = list(FINITE_SIZE_SCALING_TYPING_RATCHET)
FINITE_SIZE_SCALING_COVERAGE_DATA_FILE = "/tmp/scpn-qc-finite-size-scaling-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-finite-size-scaling-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *FINITE_SIZE_SCALING_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D finite-size-scaling quality ratchet",
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
                *FINITE_SIZE_SCALING_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline finite-size execution and exact source coverage gates."""
    return [
        (
            "finite-size-scaling focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={FINITE_SIZE_SCALING_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *FINITE_SIZE_SCALING_COVERAGE_COHORT,
            ],
        ),
        (
            "finite-size-scaling exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={FINITE_SIZE_SCALING_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/analysis/finite_size_scaling.py",
            ],
        ),
    ]


__all__ = [
    "FINITE_SIZE_SCALING_COVERAGE_COHORT",
    "FINITE_SIZE_SCALING_COVERAGE_DATA_FILE",
    "FINITE_SIZE_SCALING_DOCSTRING_RATCHET",
    "FINITE_SIZE_SCALING_SOURCE",
    "FINITE_SIZE_SCALING_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
