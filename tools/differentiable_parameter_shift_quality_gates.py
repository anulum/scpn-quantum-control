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
PHASE_PARAMETER_SHIFT_SOURCE = "src/scpn_quantum_control/phase/param_shift.py"
DIFFERENTIABLE_STOCHASTIC_POLICY_SOURCE = (
    "src/scpn_quantum_control/differentiable_stochastic_policy.py"
)
DIFFERENTIABLE_PARAMETER_SHIFT_SOURCES = [
    DIFFERENTIABLE_PARAMETER_SHIFT_SOURCE,
    PHASE_PARAMETER_SHIFT_SOURCE,
    DIFFERENTIABLE_STOCHASTIC_POLICY_SOURCE,
]
DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_COHORT = [
    "tests/test_differentiable_parameter_shift.py",
    "tests/test_param_shift.py",
    "tests/test_param_shift_contracts.py",
    "tests/test_phase_gradient_backend.py",
    "tests/test_phase_gradient_training.py",
    "tests/test_phase_param_shift.py",
    "tests/test_stochastic_gradient_failure_policy.py",
]
DIFFERENTIABLE_PARAMETER_SHIFT_TYPING_RATCHET = [
    *DIFFERENTIABLE_PARAMETER_SHIFT_SOURCES,
    "tests/test_phase_param_shift.py",
    "tests/test_stochastic_gradient_failure_policy.py",
    "tools/differentiable_parameter_shift_quality_gates.py",
    "tests/test_differentiable_parameter_shift_quality_gate.py",
]
DIFFERENTIABLE_PARAMETER_SHIFT_DOCSTRING_RATCHET = [
    *DIFFERENTIABLE_PARAMETER_SHIFT_SOURCES,
    "tests/test_differentiable_parameter_shift.py",
    "tests/test_phase_param_shift.py",
    "tests/test_stochastic_gradient_failure_policy.py",
    "tools/differentiable_parameter_shift_quality_gates.py",
    "tests/test_differentiable_parameter_shift_quality_gate.py",
]
DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-differentiable-parameter-shift-quality.coverage"  # nosec B108
)
DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_INCLUDE = (
    "--include=*/differentiable_parameter_shift.py,*/phase/param_shift.py,"
    "*/differentiable_stochastic_policy.py"
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
                "--preview",
                "--select",
                "D,D413,D417,D420",
                "--config",
                "lint.explicit-preview-rules = true",
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
                DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_INCLUDE,
            ],
        ),
    ]


__all__ = [
    "DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_COHORT",
    "DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_DATA_FILE",
    "DIFFERENTIABLE_PARAMETER_SHIFT_COVERAGE_INCLUDE",
    "DIFFERENTIABLE_PARAMETER_SHIFT_DOCSTRING_RATCHET",
    "DIFFERENTIABLE_PARAMETER_SHIFT_SOURCE",
    "DIFFERENTIABLE_PARAMETER_SHIFT_SOURCES",
    "DIFFERENTIABLE_PARAMETER_SHIFT_TYPING_RATCHET",
    "DIFFERENTIABLE_STOCHASTIC_POLICY_SOURCE",
    "PHASE_PARAMETER_SHIFT_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
