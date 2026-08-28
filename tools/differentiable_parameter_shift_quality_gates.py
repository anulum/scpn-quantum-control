# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable parameter-shift quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
DIFFERENTIABLE_PARAMETER_SHIFT_SOURCE = (
    "src/scpn_quantum_control/differentiable_parameter_shift.py"
)
DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_COHORT = ["tests/test_differentiable_parameter_shift.py"]
DIFFERENTIABLE_PARAMETER_SHIFT_TYPING_RATCHET = [
    DIFFERENTIABLE_PARAMETER_SHIFT_SOURCE,
    "tools/differentiable_parameter_shift_quality_gates.py",
    "tests/test_differentiable_parameter_shift_quality_gate.py",
]
DIFFERENTIABLE_PARAMETER_SHIFT_DOCSTRING_RATCHET = [
    DIFFERENTIABLE_PARAMETER_SHIFT_SOURCE,
    *DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_COHORT,
    "tools/differentiable_parameter_shift_quality_gates.py",
    "tests/test_differentiable_parameter_shift_quality_gate.py",
]
DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-differentiable-parameter-shift-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-differentiable-parameter-shift-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DIFFERENTIABLE_PARAMETER_SHIFT_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D differentiable-parameter-shift quality ratchet",
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
                *DIFFERENTIABLE_PARAMETER_SHIFT_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline transform execution and exact source coverage gates."""
    return [
        (
            "differentiable-parameter-shift focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_COHORT,
            ],
        ),
        (
            "differentiable-parameter-shift exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/differentiable_parameter_shift.py",
            ],
        ),
    ]


__all__ = [
    "DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_COHORT",
    "DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_DATA_FILE",
    "DIFFERENTIABLE_PARAMETER_SHIFT_DOCSTRING_RATCHET",
    "DIFFERENTIABLE_PARAMETER_SHIFT_SOURCE",
    "DIFFERENTIABLE_PARAMETER_SHIFT_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
