# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QPU compute-type quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
QPU_COMPUTE_TYPES_SOURCE = "src/scpn_quantum_control/qpu_compute_types.py"
QPU_COMPUTE_TYPES_COVERAGE_COHORT = [
    "tests/test_qpu_compute.py",
    "tests/test_qpu_compute_runtime_branches.py",
    "tests/test_qpu_compute_types_guards.py",
    "tests/test_qpu_compute_product.py",
]
QPU_COMPUTE_TYPES_TYPING_RATCHET = [
    QPU_COMPUTE_TYPES_SOURCE,
    "tools/qpu_compute_types_quality_gates.py",
    "tests/test_qpu_compute_types_quality_gate.py",
]
QPU_COMPUTE_TYPES_DOCSTRING_RATCHET = [
    QPU_COMPUTE_TYPES_SOURCE,
    *QPU_COMPUTE_TYPES_COVERAGE_COHORT,
    "tools/qpu_compute_types_quality_gates.py",
    "tests/test_qpu_compute_types_quality_gate.py",
]
QPU_COMPUTE_TYPES_COVERAGE_DATA_FILE = "/tmp/scpn-qc-qpu-compute-types-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-qpu-compute-types-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *QPU_COMPUTE_TYPES_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D QPU compute-types quality ratchet",
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
                *QPU_COMPUTE_TYPES_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline QPU compute-type execution and exact coverage gates."""
    return [
        (
            "QPU compute-types focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={QPU_COMPUTE_TYPES_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *QPU_COMPUTE_TYPES_COVERAGE_COHORT,
            ],
        ),
        (
            "QPU compute-types exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={QPU_COMPUTE_TYPES_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/qpu_compute_types.py",
            ],
        ),
    ]


__all__ = [
    "QPU_COMPUTE_TYPES_COVERAGE_COHORT",
    "QPU_COMPUTE_TYPES_COVERAGE_DATA_FILE",
    "QPU_COMPUTE_TYPES_DOCSTRING_RATCHET",
    "QPU_COMPUTE_TYPES_SOURCE",
    "QPU_COMPUTE_TYPES_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
