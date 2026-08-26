# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — experiment mitigation quality gates
"""Build strict documentation and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
EXPERIMENT_MITIGATION_QUALITY_RATCHET = [
    "src/scpn_quantum_control/hardware/experiment_mitigation.py",
    "tests/test_experiment_mitigation.py",
    "tools/experiment_mitigation_quality_gates.py",
    "tests/test_experiment_mitigation_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
EXPERIMENT_MITIGATION_COVERAGE_COHORT = ["tests/test_experiment_mitigation.py"]
"""Tests that own exact experiment-mitigation coverage."""
EXPERIMENT_MITIGATION_COVERAGE_DATA_FILE = ".coverage.experiment-mitigation-quality"
"""Isolated coverage database for experiment-mitigation diagnostics."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-experiment-mitigation-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *EXPERIMENT_MITIGATION_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D experiment-mitigation quality ratchet",
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
                *EXPERIMENT_MITIGATION_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    data = EXPERIMENT_MITIGATION_COVERAGE_DATA_FILE
    return [
        (
            "experiment-mitigation focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *EXPERIMENT_MITIGATION_COVERAGE_COHORT,
            ],
        ),
        (
            "experiment-mitigation exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hardware/experiment_mitigation.py",
            ],
        ),
    ]


__all__ = [
    "EXPERIMENT_MITIGATION_COVERAGE_COHORT",
    "EXPERIMENT_MITIGATION_COVERAGE_DATA_FILE",
    "EXPERIMENT_MITIGATION_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
