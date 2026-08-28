# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware experiment-control quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
HARDWARE_EXPERIMENT_CONTROL_SOURCE = "src/scpn_quantum_control/hardware/experiment_control.py"
HARDWARE_EXPERIMENT_CONTROL_COVERAGE_COHORT = [
    "tests/test_hardware_runner.py",
    "tests/test_hardware_experiments_contracts.py",
]
HARDWARE_EXPERIMENT_CONTROL_TYPING_RATCHET = [
    HARDWARE_EXPERIMENT_CONTROL_SOURCE,
    "tools/hardware_experiment_control_quality_gates.py",
    "tests/test_hardware_experiment_control_quality_gate.py",
]
HARDWARE_EXPERIMENT_CONTROL_DOCSTRING_RATCHET = [
    HARDWARE_EXPERIMENT_CONTROL_SOURCE,
    *HARDWARE_EXPERIMENT_CONTROL_COVERAGE_COHORT,
    "tools/hardware_experiment_control_quality_gates.py",
    "tests/test_hardware_experiment_control_quality_gate.py",
]
HARDWARE_EXPERIMENT_CONTROL_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-hardware-experiment-control-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-hardware-experiment-control-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *HARDWARE_EXPERIMENT_CONTROL_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D hardware-experiment-control quality ratchet",
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
                *HARDWARE_EXPERIMENT_CONTROL_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline experiment execution and exact source coverage gates."""
    return [
        (
            "hardware-experiment-control focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={HARDWARE_EXPERIMENT_CONTROL_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *HARDWARE_EXPERIMENT_CONTROL_COVERAGE_COHORT,
            ],
        ),
        (
            "hardware-experiment-control exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={HARDWARE_EXPERIMENT_CONTROL_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hardware/experiment_control.py",
            ],
        ),
    ]


__all__ = [
    "HARDWARE_EXPERIMENT_CONTROL_COVERAGE_COHORT",
    "HARDWARE_EXPERIMENT_CONTROL_COVERAGE_DATA_FILE",
    "HARDWARE_EXPERIMENT_CONTROL_DOCSTRING_RATCHET",
    "HARDWARE_EXPERIMENT_CONTROL_SOURCE",
    "HARDWARE_EXPERIMENT_CONTROL_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
