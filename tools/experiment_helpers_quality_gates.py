# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Experiment-helper quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
EXPERIMENT_HELPERS_SOURCE = "src/scpn_quantum_control/hardware/_experiment_helpers.py"
EXPERIMENT_HELPERS_PRIMARY_TEST = "tests/test_experiment_helpers_branches.py"
EXPERIMENT_HELPERS_COVERAGE_COHORT = [EXPERIMENT_HELPERS_PRIMARY_TEST]
EXPERIMENT_HELPERS_TYPING_RATCHET = [
    EXPERIMENT_HELPERS_SOURCE,
    EXPERIMENT_HELPERS_PRIMARY_TEST,
    "tools/experiment_helpers_quality_gates.py",
    "tests/test_experiment_helpers_quality_gate.py",
]
EXPERIMENT_HELPERS_DOCSTRING_RATCHET = [*EXPERIMENT_HELPERS_TYPING_RATCHET]
EXPERIMENT_HELPERS_COVERAGE_DATA_FILE = "/tmp/scpn-qc-experiment-helpers-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-experiment-helpers-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *EXPERIMENT_HELPERS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D experiment-helpers quality ratchet",
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
                *EXPERIMENT_HELPERS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline circuit/reduction execution and exact coverage gates."""
    return [
        (
            "experiment-helpers focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={EXPERIMENT_HELPERS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *EXPERIMENT_HELPERS_COVERAGE_COHORT,
            ],
        ),
        (
            "experiment-helpers exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={EXPERIMENT_HELPERS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hardware/_experiment_helpers.py",
            ],
        ),
    ]


__all__ = [
    "EXPERIMENT_HELPERS_COVERAGE_COHORT",
    "EXPERIMENT_HELPERS_COVERAGE_DATA_FILE",
    "EXPERIMENT_HELPERS_DOCSTRING_RATCHET",
    "EXPERIMENT_HELPERS_PRIMARY_TEST",
    "EXPERIMENT_HELPERS_SOURCE",
    "EXPERIMENT_HELPERS_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
