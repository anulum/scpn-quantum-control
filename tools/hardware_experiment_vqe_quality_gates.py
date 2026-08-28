# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware experiment VQE quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
HARDWARE_EXPERIMENT_VQE_SOURCE = "src/scpn_quantum_control/hardware/experiment_vqe.py"
HARDWARE_EXPERIMENT_VQE_COVERAGE_COHORT = [
    "tests/test_hardware_runner.py",
    "tests/test_hardware_experiments_contracts.py",
    "tests/test_experiments_edge_cases.py",
]
HARDWARE_EXPERIMENT_VQE_TYPING_RATCHET = [
    HARDWARE_EXPERIMENT_VQE_SOURCE,
    "tools/hardware_experiment_vqe_quality_gates.py",
    "tests/test_hardware_experiment_vqe_quality_gate.py",
]
HARDWARE_EXPERIMENT_VQE_DOCSTRING_RATCHET = [
    HARDWARE_EXPERIMENT_VQE_SOURCE,
    *HARDWARE_EXPERIMENT_VQE_COVERAGE_COHORT,
    "tools/hardware_experiment_vqe_quality_gates.py",
    "tests/test_hardware_experiment_vqe_quality_gate.py",
]
HARDWARE_EXPERIMENT_VQE_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-hardware-experiment-vqe-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-hardware-experiment-vqe-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *HARDWARE_EXPERIMENT_VQE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D hardware-experiment-vqe quality ratchet",
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
                *HARDWARE_EXPERIMENT_VQE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline execution and exact source coverage gates."""
    return [
        (
            "hardware-experiment-vqe focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={HARDWARE_EXPERIMENT_VQE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *HARDWARE_EXPERIMENT_VQE_COVERAGE_COHORT,
            ],
        ),
        (
            "hardware-experiment-vqe exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={HARDWARE_EXPERIMENT_VQE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hardware/experiment_vqe.py",
            ],
        ),
    ]


__all__ = [
    "HARDWARE_EXPERIMENT_VQE_COVERAGE_COHORT",
    "HARDWARE_EXPERIMENT_VQE_COVERAGE_DATA_FILE",
    "HARDWARE_EXPERIMENT_VQE_DOCSTRING_RATCHET",
    "HARDWARE_EXPERIMENT_VQE_SOURCE",
    "HARDWARE_EXPERIMENT_VQE_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
