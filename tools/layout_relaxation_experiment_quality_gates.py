# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — layout-relaxation-experiment quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
LAYOUT_RELAXATION_EXPERIMENT_SOURCE = (
    "src/scpn_quantum_control/benchmarks/layout_relaxation_experiment.py"
)
LAYOUT_RELAXATION_EXPERIMENT_COVERAGE_COHORT = ["tests/test_layout_relaxation_experiment.py"]
LAYOUT_RELAXATION_EXPERIMENT_TYPING_RATCHET = [
    LAYOUT_RELAXATION_EXPERIMENT_SOURCE,
    *LAYOUT_RELAXATION_EXPERIMENT_COVERAGE_COHORT,
    "tests/test_run_layout_relaxation_experiment.py",
    "tools/layout_relaxation_experiment_quality_gates.py",
    "tests/test_layout_relaxation_experiment_quality_gate.py",
]
LAYOUT_RELAXATION_EXPERIMENT_DOCSTRING_RATCHET = [*LAYOUT_RELAXATION_EXPERIMENT_TYPING_RATCHET]
LAYOUT_RELAXATION_EXPERIMENT_COVERAGE_DATA_FILE = (  # nosec B108
    "/tmp/scpn-qc-layout-relaxation-experiment-quality.coverage"
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-layout-relaxation-experiment-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *LAYOUT_RELAXATION_EXPERIMENT_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D layout-relaxation-experiment quality ratchet",
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
                *LAYOUT_RELAXATION_EXPERIMENT_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build stubbed experiment execution and exact source-coverage gates."""
    return [
        (
            "layout-relaxation-experiment focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={LAYOUT_RELAXATION_EXPERIMENT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *LAYOUT_RELAXATION_EXPERIMENT_COVERAGE_COHORT,
            ],
        ),
        (
            "layout-relaxation-experiment exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={LAYOUT_RELAXATION_EXPERIMENT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/benchmarks/layout_relaxation_experiment.py",
            ],
        ),
    ]


__all__ = [
    "LAYOUT_RELAXATION_EXPERIMENT_COVERAGE_COHORT",
    "LAYOUT_RELAXATION_EXPERIMENT_COVERAGE_DATA_FILE",
    "LAYOUT_RELAXATION_EXPERIMENT_DOCSTRING_RATCHET",
    "LAYOUT_RELAXATION_EXPERIMENT_SOURCE",
    "LAYOUT_RELAXATION_EXPERIMENT_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
