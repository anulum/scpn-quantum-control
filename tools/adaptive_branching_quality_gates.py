# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — adaptive-branching quality-gate specification
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
ADAPTIVE_BRANCHING_SOURCE = "src/scpn_quantum_control/control/adaptive_branching.py"
"""No-submit S8 branch-planning and readiness surface."""
ADAPTIVE_BRANCHING_EXPORTER = "scripts/export_s8_adaptive_branching_readiness.py"
"""Executable JSON and Markdown readiness exporter."""
ADAPTIVE_BRANCHING_TYPING_RATCHET = [
    ADAPTIVE_BRANCHING_SOURCE,
    "tests/test_adaptive_branching.py",
    "tests/test_adaptive_branching_branches.py",
    ADAPTIVE_BRANCHING_EXPORTER,
    "tests/test_export_s8_adaptive_branching_readiness.py",
    "tools/adaptive_branching_quality_gates.py",
    "tests/test_adaptive_branching_quality_gate.py",
    "tools/preflight.py",
    "tests/test_preflight_tool.py",
]
"""Production, export, tests, and gate surfaces held to strict MyPy."""
ADAPTIVE_BRANCHING_DOCSTRING_RATCHET = [
    ADAPTIVE_BRANCHING_SOURCE,
    "tests/test_adaptive_branching.py",
    "tests/test_adaptive_branching_branches.py",
    ADAPTIVE_BRANCHING_EXPORTER,
    "tests/test_export_s8_adaptive_branching_readiness.py",
    "tools/adaptive_branching_quality_gates.py",
    "tests/test_adaptive_branching_quality_gate.py",
    "tests/test_preflight_tool.py",
]
"""Whole owner cohort held to complete NumPy docstrings."""
ADAPTIVE_BRANCHING_COVERAGE_COHORT = [
    "tests/test_adaptive_branching.py",
    "tests/test_adaptive_branching_branches.py",
    "tests/test_export_s8_adaptive_branching_readiness.py",
]
"""Public and executable-export tests that own source branch coverage."""
ADAPTIVE_BRANCHING_COVERAGE_DATA_FILE = "/tmp/scpn-qc-adaptive-branching-quality.coverage"  # nosec B108
"""Isolated coverage database for the adaptive-branching owner."""
ADAPTIVE_BRANCHING_COVERAGE_INCLUDE = "*/control/adaptive_branching.py"
"""Exact production source include for the coverage report."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-adaptive-branching-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ADAPTIVE_BRANCHING_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D adaptive-branching quality ratchet",
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
                *ADAPTIVE_BRANCHING_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real owner execution and exact source-coverage gates."""
    return [
        (
            "adaptive-branching focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ADAPTIVE_BRANCHING_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ADAPTIVE_BRANCHING_COVERAGE_COHORT,
            ],
        ),
        (
            "adaptive-branching exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ADAPTIVE_BRANCHING_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={ADAPTIVE_BRANCHING_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "ADAPTIVE_BRANCHING_COVERAGE_COHORT",
    "ADAPTIVE_BRANCHING_COVERAGE_DATA_FILE",
    "ADAPTIVE_BRANCHING_COVERAGE_INCLUDE",
    "ADAPTIVE_BRANCHING_DOCSTRING_RATCHET",
    "ADAPTIVE_BRANCHING_EXPORTER",
    "ADAPTIVE_BRANCHING_SOURCE",
    "ADAPTIVE_BRANCHING_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
