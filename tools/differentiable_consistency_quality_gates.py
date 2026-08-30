# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable-consistency quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
DIFFERENTIABLE_CONSISTENCY_SOURCE = "src/scpn_quantum_control/differentiable_consistency.py"
DIFFERENTIABLE_CONSISTENCY_PRIMARY_TEST = "tests/test_differentiable_consistency.py"
DIFFERENTIABLE_CONSISTENCY_COVERAGE_COHORT = [DIFFERENTIABLE_CONSISTENCY_PRIMARY_TEST]
DIFFERENTIABLE_CONSISTENCY_TYPING_RATCHET = [
    DIFFERENTIABLE_CONSISTENCY_SOURCE,
    DIFFERENTIABLE_CONSISTENCY_PRIMARY_TEST,
    "tools/differentiable_consistency_quality_gates.py",
    "tests/test_differentiable_consistency_quality_gate.py",
]
DIFFERENTIABLE_CONSISTENCY_DOCSTRING_RATCHET = [*DIFFERENTIABLE_CONSISTENCY_TYPING_RATCHET]
DIFFERENTIABLE_CONSISTENCY_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-differentiable-consistency-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-differentiable-consistency-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DIFFERENTIABLE_CONSISTENCY_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D differentiable-consistency quality ratchet",
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
                *DIFFERENTIABLE_CONSISTENCY_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build local consistency-diagnostic execution and coverage gates."""
    return [
        (
            "differentiable-consistency focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_CONSISTENCY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DIFFERENTIABLE_CONSISTENCY_COVERAGE_COHORT,
            ],
        ),
        (
            "differentiable-consistency exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_CONSISTENCY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/differentiable_consistency.py",
            ],
        ),
    ]


__all__ = [
    "DIFFERENTIABLE_CONSISTENCY_COVERAGE_COHORT",
    "DIFFERENTIABLE_CONSISTENCY_COVERAGE_DATA_FILE",
    "DIFFERENTIABLE_CONSISTENCY_DOCSTRING_RATCHET",
    "DIFFERENTIABLE_CONSISTENCY_PRIMARY_TEST",
    "DIFFERENTIABLE_CONSISTENCY_SOURCE",
    "DIFFERENTIABLE_CONSISTENCY_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
