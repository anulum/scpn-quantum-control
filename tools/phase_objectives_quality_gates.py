# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase-objectives quality-gate specification
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PHASE_OBJECTIVES_SOURCE = "src/scpn_quantum_control/phase/objectives.py"
PHASE_OBJECTIVES_TEST = "tests/test_phase_objectives.py"
PHASE_OBJECTIVES_TYPING_RATCHET = [
    PHASE_OBJECTIVES_SOURCE,
    PHASE_OBJECTIVES_TEST,
    "tools/phase_objectives_quality_gates.py",
    "tests/test_phase_objectives_quality_gate.py",
    "tools/preflight.py",
    "tests/test_preflight_tool.py",
]
"""Production, tests, and gate surfaces held to strict MyPy."""
PHASE_OBJECTIVES_DOCSTRING_RATCHET = [
    PHASE_OBJECTIVES_SOURCE,
    PHASE_OBJECTIVES_TEST,
    "tools/phase_objectives_quality_gates.py",
    "tests/test_phase_objectives_quality_gate.py",
    "tests/test_preflight_tool.py",
]
"""Whole owner cohort held to complete NumPy docstrings."""
PHASE_OBJECTIVES_COVERAGE_COHORT = [PHASE_OBJECTIVES_TEST]
"""Public phase-objective suite that owns source branch coverage."""
PHASE_OBJECTIVES_COVERAGE_DATA_FILE = "/tmp/scpn-qc-phase-objectives-quality.coverage"  # nosec B108
PHASE_OBJECTIVES_COVERAGE_INCLUDE = "*/phase/objectives.py"


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-phase-objectives-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PHASE_OBJECTIVES_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D phase-objectives quality ratchet",
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
                *PHASE_OBJECTIVES_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real objective execution and exact source-coverage gates."""
    return [
        (
            "phase-objectives focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_OBJECTIVES_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PHASE_OBJECTIVES_COVERAGE_COHORT,
            ],
        ),
        (
            "phase-objectives exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_OBJECTIVES_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={PHASE_OBJECTIVES_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "PHASE_OBJECTIVES_COVERAGE_COHORT",
    "PHASE_OBJECTIVES_COVERAGE_DATA_FILE",
    "PHASE_OBJECTIVES_COVERAGE_INCLUDE",
    "PHASE_OBJECTIVES_DOCSTRING_RATCHET",
    "PHASE_OBJECTIVES_SOURCE",
    "PHASE_OBJECTIVES_TEST",
    "PHASE_OBJECTIVES_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
