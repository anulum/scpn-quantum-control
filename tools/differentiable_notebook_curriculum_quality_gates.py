# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable-notebook-curriculum quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
DIFFERENTIABLE_NOTEBOOK_CURRICULUM_QUALITY_RATCHET = [
    "src/scpn_quantum_control/differentiable_notebook_curriculum.py",
    "tests/test_differentiable_notebook_curriculum.py",
    "tools/differentiable_notebook_curriculum_quality_gates.py",
    "tests/test_differentiable_notebook_curriculum_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
DIFFERENTIABLE_NOTEBOOK_CURRICULUM_COVERAGE_COHORT = [
    "tests/test_differentiable_notebook_curriculum.py"
]
"""Tests that own exact differentiable-curriculum coverage."""
DIFFERENTIABLE_NOTEBOOK_CURRICULUM_COVERAGE_DATA_FILE = (
    ".coverage.differentiable-notebook-curriculum"
)
"""Isolated coverage database for the differentiable curriculum owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-differentiable-notebook-curriculum",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DIFFERENTIABLE_NOTEBOOK_CURRICULUM_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D differentiable-notebook-curriculum ratchet",
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
                *DIFFERENTIABLE_NOTEBOOK_CURRICULUM_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "differentiable-notebook-curriculum focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_NOTEBOOK_CURRICULUM_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DIFFERENTIABLE_NOTEBOOK_CURRICULUM_COVERAGE_COHORT,
            ],
        ),
        (
            "differentiable-notebook-curriculum exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_NOTEBOOK_CURRICULUM_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/differentiable_notebook_curriculum.py",
            ],
        ),
    ]


__all__ = [
    "DIFFERENTIABLE_NOTEBOOK_CURRICULUM_COVERAGE_COHORT",
    "DIFFERENTIABLE_NOTEBOOK_CURRICULUM_COVERAGE_DATA_FILE",
    "DIFFERENTIABLE_NOTEBOOK_CURRICULUM_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
