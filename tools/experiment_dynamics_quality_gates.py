# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — experiment-dynamics quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
EXPERIMENT_DYNAMICS_SOURCE = "src/scpn_quantum_control/hardware/experiment_dynamics.py"
EXPERIMENT_DYNAMICS_COVERAGE_COHORT = ["tests/test_experiment_dynamics.py"]
EXPERIMENT_DYNAMICS_TYPING_RATCHET = [
    EXPERIMENT_DYNAMICS_SOURCE,
    "tools/experiment_dynamics_quality_gates.py",
    "tests/test_experiment_dynamics_quality_gate.py",
]
EXPERIMENT_DYNAMICS_DOCSTRING_RATCHET = [
    EXPERIMENT_DYNAMICS_SOURCE,
    "tests/test_experiment_dynamics.py",
    "tools/experiment_dynamics_quality_gates.py",
    "tests/test_experiment_dynamics_quality_gate.py",
]
EXPERIMENT_DYNAMICS_COVERAGE_DATA_FILE = "/tmp/scpn-qc-experiment-dynamics-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-experiment-dynamics-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *EXPERIMENT_DYNAMICS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D experiment-dynamics quality ratchet",
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
                *EXPERIMENT_DYNAMICS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build bounded experiment execution and exact coverage gates."""
    return [
        (
            "experiment-dynamics focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={EXPERIMENT_DYNAMICS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *EXPERIMENT_DYNAMICS_COVERAGE_COHORT,
            ],
        ),
        (
            "experiment-dynamics exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={EXPERIMENT_DYNAMICS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hardware/experiment_dynamics.py",
            ],
        ),
    ]


__all__ = [
    "EXPERIMENT_DYNAMICS_COVERAGE_COHORT",
    "EXPERIMENT_DYNAMICS_COVERAGE_DATA_FILE",
    "EXPERIMENT_DYNAMICS_DOCSTRING_RATCHET",
    "EXPERIMENT_DYNAMICS_SOURCE",
    "EXPERIMENT_DYNAMICS_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
