# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — generalised parameter-shift quality-gate specification
"""Build strict documentation and exact coverage gates for finite-spectrum shifts."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

GENERALISED_PARAMETER_SHIFT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/generalised_parameter_shift.py",
    "tests/test_phase_generalised_parameter_shift.py",
    "tools/generalised_parameter_shift_quality_gates.py",
    "tests/test_generalised_parameter_shift_quality_gate.py",
]
"""Complete strict-typing and preview-documentation owner."""

GENERALISED_PARAMETER_SHIFT_COVERAGE_COHORT = ["tests/test_phase_generalised_parameter_shift.py"]
"""Real finite-spectrum parameter-shift suite used for exact coverage."""

GENERALISED_PARAMETER_SHIFT_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-generalised-parameter-shift-quality.coverage"
)
"""Isolated coverage database for the finite-spectrum shift owner."""

GENERALISED_PARAMETER_SHIFT_COVERAGE_INCLUDE = "*/phase/generalised_parameter_shift.py"
"""Production source enforced at exact branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and preview NumPy-documentation gates."""
    return [
        (
            "mypy-strict-generalised-parameter-shift-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *GENERALISED_PARAMETER_SHIFT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D generalised-parameter-shift quality ratchet",
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
                "lint.explicit-preview-rules = true",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *GENERALISED_PARAMETER_SHIFT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real execution and exact source-coverage gates."""
    return [
        (
            "generalised-parameter-shift focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={GENERALISED_PARAMETER_SHIFT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *GENERALISED_PARAMETER_SHIFT_COVERAGE_COHORT,
            ],
        ),
        (
            "generalised-parameter-shift exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={GENERALISED_PARAMETER_SHIFT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={GENERALISED_PARAMETER_SHIFT_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "GENERALISED_PARAMETER_SHIFT_COVERAGE_COHORT",
    "GENERALISED_PARAMETER_SHIFT_COVERAGE_DATA_FILE",
    "GENERALISED_PARAMETER_SHIFT_COVERAGE_INCLUDE",
    "GENERALISED_PARAMETER_SHIFT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
