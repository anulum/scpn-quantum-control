# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hamiltonian-learning quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
HAMILTONIAN_LEARNING_SOURCE = "src/scpn_quantum_control/analysis/hamiltonian_learning.py"
HAMILTONIAN_LEARNING_COVERAGE_COHORT = ["tests/test_hamiltonian_learning.py"]
HAMILTONIAN_LEARNING_TYPING_RATCHET = [
    HAMILTONIAN_LEARNING_SOURCE,
    *HAMILTONIAN_LEARNING_COVERAGE_COHORT,
    "tools/hamiltonian_learning_quality_gates.py",
    "tests/test_hamiltonian_learning_quality_gate.py",
]
HAMILTONIAN_LEARNING_DOCSTRING_RATCHET = [*HAMILTONIAN_LEARNING_TYPING_RATCHET]
HAMILTONIAN_LEARNING_COVERAGE_DATA_FILE = "/tmp/scpn-qc-hamiltonian-learning-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-hamiltonian-learning-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *HAMILTONIAN_LEARNING_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D Hamiltonian-learning quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "D,D107,D413,D417,D420",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *HAMILTONIAN_LEARNING_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real inverse-fit execution and exact source-coverage gates."""
    return [
        (
            "Hamiltonian-learning focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={HAMILTONIAN_LEARNING_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *HAMILTONIAN_LEARNING_COVERAGE_COHORT,
            ],
        ),
        (
            "Hamiltonian-learning exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={HAMILTONIAN_LEARNING_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/analysis/hamiltonian_learning.py",
            ],
        ),
    ]


__all__ = [
    "HAMILTONIAN_LEARNING_COVERAGE_COHORT",
    "HAMILTONIAN_LEARNING_COVERAGE_DATA_FILE",
    "HAMILTONIAN_LEARNING_DOCSTRING_RATCHET",
    "HAMILTONIAN_LEARNING_SOURCE",
    "HAMILTONIAN_LEARNING_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
