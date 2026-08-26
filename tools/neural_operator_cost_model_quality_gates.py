# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — neural-operator cost-model quality-gate specification
"""Build strict documentation, typing, and coverage gates for BL-19."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
NEURAL_OPERATOR_COST_MODEL_QUALITY_RATCHET = [
    "src/scpn_quantum_control/forecasting/neural_operator_cost_model.py",
    "tests/test_neural_operator_cost_model.py",
    "tools/neural_operator_cost_model_quality_gates.py",
    "tests/test_neural_operator_cost_model_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
NEURAL_OPERATOR_COST_MODEL_COVERAGE_COHORT = ["tests/test_neural_operator_cost_model.py"]
"""Tests that own exact neural-operator cost-model coverage."""
NEURAL_OPERATOR_COST_MODEL_COVERAGE_DATA_FILE = ".coverage.neural-operator-cost-model-quality"
"""Isolated coverage database for the neural-operator cost model."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-neural-operator-cost-model-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *NEURAL_OPERATOR_COST_MODEL_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D neural-operator-cost-model quality ratchet",
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
                *NEURAL_OPERATOR_COST_MODEL_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "neural-operator-cost-model focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={NEURAL_OPERATOR_COST_MODEL_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *NEURAL_OPERATOR_COST_MODEL_COVERAGE_COHORT,
            ],
        ),
        (
            "neural-operator-cost-model exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={NEURAL_OPERATOR_COST_MODEL_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/neural_operator_cost_model.py",
            ],
        ),
    ]


__all__ = [
    "NEURAL_OPERATOR_COST_MODEL_COVERAGE_COHORT",
    "NEURAL_OPERATOR_COST_MODEL_COVERAGE_DATA_FILE",
    "NEURAL_OPERATOR_COST_MODEL_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
