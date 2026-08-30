# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — competitive-baseline-watch quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
COMPETITIVE_BASELINE_WATCH_QUALITY_RATCHET = [
    "src/scpn_quantum_control/competitive_baseline_watch.py",
    "src/scpn_quantum_control/benchmarks/reproducible_comparison.py",
    "src/scpn_quantum_control/benchmarks/kuramoto_competitive_types.py",
    "tests/test_competitive_baseline_watch.py",
    "tests/test_reproducible_comparison.py",
    "tests/test_kuramoto_competitive_types.py",
    "tools/competitive_baseline_watch_quality_gates.py",
    "tests/test_competitive_baseline_watch_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
COMPETITIVE_BASELINE_WATCH_COVERAGE_COHORT = [
    "tests/test_competitive_baseline_watch.py",
    "tests/test_reproducible_comparison.py",
    "tests/test_kuramoto_competitive_types.py",
]
"""Tests that own exact competitive-baseline-watch coverage."""
COMPETITIVE_BASELINE_WATCH_COVERAGE_DATA_FILE = ".coverage.competitive-baseline-watch-quality"
"""Isolated coverage database for the competitive-baseline-watch owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-competitive-baseline-watch-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *COMPETITIVE_BASELINE_WATCH_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D competitive-baseline-watch quality ratchet",
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
                *COMPETITIVE_BASELINE_WATCH_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "competitive-baseline-watch focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={COMPETITIVE_BASELINE_WATCH_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *COMPETITIVE_BASELINE_WATCH_COVERAGE_COHORT,
            ],
        ),
        (
            "competitive-baseline-watch exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={COMPETITIVE_BASELINE_WATCH_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/competitive_baseline_watch.py,*/benchmarks/reproducible_comparison.py,*/benchmarks/kuramoto_competitive_types.py",
            ],
        ),
    ]


__all__ = [
    "COMPETITIVE_BASELINE_WATCH_COVERAGE_COHORT",
    "COMPETITIVE_BASELINE_WATCH_COVERAGE_DATA_FILE",
    "COMPETITIVE_BASELINE_WATCH_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
