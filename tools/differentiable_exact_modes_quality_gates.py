# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable exact-mode quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
DIFFERENTIABLE_EXACT_MODES_SOURCE = "src/scpn_quantum_control/differentiable_exact_modes.py"
DIFFERENTIABLE_EXACT_MODES_COVERAGE_COHORT = ["tests/test_differentiable_exact_modes.py"]
DIFFERENTIABLE_EXACT_MODES_TYPING_RATCHET = [
    DIFFERENTIABLE_EXACT_MODES_SOURCE,
    "tests/test_differentiable_exact_modes.py",
    "tools/differentiable_exact_modes_quality_gates.py",
    "tests/test_differentiable_exact_modes_quality_gate.py",
]
DIFFERENTIABLE_EXACT_MODES_DOCSTRING_RATCHET = [
    *DIFFERENTIABLE_EXACT_MODES_TYPING_RATCHET,
]
DIFFERENTIABLE_EXACT_MODES_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-differentiable-exact-modes-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-differentiable-exact-modes-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DIFFERENTIABLE_EXACT_MODES_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D differentiable-exact-modes quality ratchet",
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
                *DIFFERENTIABLE_EXACT_MODES_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build exact execution and source-coverage gates."""
    return [
        (
            "differentiable-exact-modes focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_EXACT_MODES_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DIFFERENTIABLE_EXACT_MODES_COVERAGE_COHORT,
            ],
        ),
        (
            "differentiable-exact-modes exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_EXACT_MODES_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/differentiable_exact_modes.py",
            ],
        ),
    ]


__all__ = [
    "DIFFERENTIABLE_EXACT_MODES_COVERAGE_COHORT",
    "DIFFERENTIABLE_EXACT_MODES_COVERAGE_DATA_FILE",
    "DIFFERENTIABLE_EXACT_MODES_DOCSTRING_RATCHET",
    "DIFFERENTIABLE_EXACT_MODES_SOURCE",
    "DIFFERENTIABLE_EXACT_MODES_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
