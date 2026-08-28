# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — sparse-Hamiltonian quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
SPARSE_HAMILTONIAN_SOURCE = "src/scpn_quantum_control/bridge/sparse_hamiltonian.py"
SPARSE_HAMILTONIAN_COVERAGE_COHORT = [
    "tests/test_sparse_hamiltonian.py",
    "tests/test_sparse_hamiltonian_branches.py",
]
SPARSE_HAMILTONIAN_TYPING_RATCHET = [
    SPARSE_HAMILTONIAN_SOURCE,
    *SPARSE_HAMILTONIAN_COVERAGE_COHORT,
    "tools/sparse_hamiltonian_quality_gates.py",
    "tests/test_sparse_hamiltonian_quality_gate.py",
]
SPARSE_HAMILTONIAN_DOCSTRING_RATCHET = [*SPARSE_HAMILTONIAN_TYPING_RATCHET]
SPARSE_HAMILTONIAN_COVERAGE_DATA_FILE = (  # nosec B108
    "/tmp/scpn-qc-sparse-hamiltonian-quality.coverage"
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-sparse-hamiltonian-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *SPARSE_HAMILTONIAN_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D sparse-Hamiltonian quality ratchet",
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
                *SPARSE_HAMILTONIAN_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real sparse-Hamiltonian execution and exact coverage gates."""
    return [
        (
            "sparse-Hamiltonian focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={SPARSE_HAMILTONIAN_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *SPARSE_HAMILTONIAN_COVERAGE_COHORT,
            ],
        ),
        (
            "sparse-Hamiltonian exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={SPARSE_HAMILTONIAN_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/bridge/sparse_hamiltonian.py",
            ],
        ),
    ]


__all__ = [
    "SPARSE_HAMILTONIAN_COVERAGE_COHORT",
    "SPARSE_HAMILTONIAN_COVERAGE_DATA_FILE",
    "SPARSE_HAMILTONIAN_DOCSTRING_RATCHET",
    "SPARSE_HAMILTONIAN_SOURCE",
    "SPARSE_HAMILTONIAN_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
