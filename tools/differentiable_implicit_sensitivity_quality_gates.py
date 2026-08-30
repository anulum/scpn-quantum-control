# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable implicit-sensitivity quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
DIFFERENTIABLE_IMPLICIT_SENSITIVITY_SOURCE = (
    "src/scpn_quantum_control/differentiable_implicit_sensitivity.py"
)
DIFFERENTIABLE_IMPLICIT_SENSITIVITY_PRIMARY_TEST = (
    "tests/test_differentiable_implicit_sensitivity.py"
)
DIFFERENTIABLE_IMPLICIT_SENSITIVITY_COVERAGE_COHORT = [
    DIFFERENTIABLE_IMPLICIT_SENSITIVITY_PRIMARY_TEST
]
DIFFERENTIABLE_IMPLICIT_SENSITIVITY_TYPING_RATCHET = [
    DIFFERENTIABLE_IMPLICIT_SENSITIVITY_SOURCE,
    DIFFERENTIABLE_IMPLICIT_SENSITIVITY_PRIMARY_TEST,
    "tools/differentiable_implicit_sensitivity_quality_gates.py",
    "tests/test_differentiable_implicit_sensitivity_quality_gate.py",
]
DIFFERENTIABLE_IMPLICIT_SENSITIVITY_DOCSTRING_RATCHET = [
    *DIFFERENTIABLE_IMPLICIT_SENSITIVITY_TYPING_RATCHET
]
DIFFERENTIABLE_IMPLICIT_SENSITIVITY_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-differentiable-implicit-sensitivity-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-differentiable-implicit-sensitivity-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DIFFERENTIABLE_IMPLICIT_SENSITIVITY_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D differentiable implicit-sensitivity quality ratchet",
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
                'lint.pydocstyle.convention = "numpy"',
                *DIFFERENTIABLE_IMPLICIT_SENSITIVITY_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build local implicit-sensitivity execution and coverage gates."""
    return [
        (
            "differentiable implicit-sensitivity focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_IMPLICIT_SENSITIVITY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DIFFERENTIABLE_IMPLICIT_SENSITIVITY_COVERAGE_COHORT,
            ],
        ),
        (
            "differentiable implicit-sensitivity exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_IMPLICIT_SENSITIVITY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/differentiable_implicit_sensitivity.py",
            ],
        ),
    ]


__all__ = [
    "DIFFERENTIABLE_IMPLICIT_SENSITIVITY_COVERAGE_COHORT",
    "DIFFERENTIABLE_IMPLICIT_SENSITIVITY_COVERAGE_DATA_FILE",
    "DIFFERENTIABLE_IMPLICIT_SENSITIVITY_DOCSTRING_RATCHET",
    "DIFFERENTIABLE_IMPLICIT_SENSITIVITY_PRIMARY_TEST",
    "DIFFERENTIABLE_IMPLICIT_SENSITIVITY_SOURCE",
    "DIFFERENTIABLE_IMPLICIT_SENSITIVITY_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
