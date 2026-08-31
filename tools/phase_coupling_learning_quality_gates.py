# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase coupling-learning quality-gate specification
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PHASE_COUPLING_LEARNING_SOURCE = "src/scpn_quantum_control/phase/coupling_learning.py"
PHASE_COUPLING_LEARNING_TEST = "tests/test_phase_coupling_learning.py"
PHASE_COUPLING_LEARNING_TYPING_RATCHET = [
    PHASE_COUPLING_LEARNING_SOURCE,
    PHASE_COUPLING_LEARNING_TEST,
    "tools/phase_coupling_learning_quality_gates.py",
    "tests/test_phase_coupling_learning_quality_gate.py",
    "tools/preflight.py",
    "tests/test_preflight_tool.py",
]
"""Production, tests, and gate surfaces held to strict MyPy."""
PHASE_COUPLING_LEARNING_DOCSTRING_RATCHET = [
    PHASE_COUPLING_LEARNING_SOURCE,
    PHASE_COUPLING_LEARNING_TEST,
    "tools/phase_coupling_learning_quality_gates.py",
    "tests/test_phase_coupling_learning_quality_gate.py",
    "tests/test_preflight_tool.py",
]
"""Whole owner cohort held to complete NumPy docstrings."""
PHASE_COUPLING_LEARNING_COVERAGE_COHORT = [PHASE_COUPLING_LEARNING_TEST]
"""Public simulator suite that owns source branch coverage."""
PHASE_COUPLING_LEARNING_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-phase-coupling-learning-quality.coverage"  # nosec B108
)
PHASE_COUPLING_LEARNING_COVERAGE_INCLUDE = "*/phase/coupling_learning.py"


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-phase-coupling-learning-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PHASE_COUPLING_LEARNING_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D phase-coupling-learning quality ratchet",
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
                "lint.explicit-preview-rules = true",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *PHASE_COUPLING_LEARNING_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real simulator execution and exact source-coverage gates."""
    return [
        (
            "phase-coupling-learning focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_COUPLING_LEARNING_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PHASE_COUPLING_LEARNING_COVERAGE_COHORT,
            ],
        ),
        (
            "phase-coupling-learning exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_COUPLING_LEARNING_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={PHASE_COUPLING_LEARNING_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "PHASE_COUPLING_LEARNING_COVERAGE_COHORT",
    "PHASE_COUPLING_LEARNING_COVERAGE_DATA_FILE",
    "PHASE_COUPLING_LEARNING_COVERAGE_INCLUDE",
    "PHASE_COUPLING_LEARNING_DOCSTRING_RATCHET",
    "PHASE_COUPLING_LEARNING_SOURCE",
    "PHASE_COUPLING_LEARNING_TEST",
    "PHASE_COUPLING_LEARNING_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
