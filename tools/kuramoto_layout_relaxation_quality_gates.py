# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto layout-relaxation quality gates
"""Build strict documentation and exact coverage gates for layout relaxation."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
KURAMOTO_LAYOUT_RELAXATION_QUALITY_RATCHET = [
    "src/scpn_quantum_control/hardware/kuramoto_layout_relaxation.py",
    "tests/test_kuramoto_layout_relaxation.py",
    "tools/kuramoto_layout_relaxation_quality_gates.py",
    "tests/test_kuramoto_layout_relaxation_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
KURAMOTO_LAYOUT_RELAXATION_COVERAGE_COHORT = ["tests/test_kuramoto_layout_relaxation.py"]
"""Tests that own exact layout-relaxation coverage."""
KURAMOTO_LAYOUT_RELAXATION_COVERAGE_DATA_FILE = ".coverage.kuramoto-layout-relaxation-quality"
"""Isolated coverage database for the layout-relaxation module."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-kuramoto-layout-relaxation-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *KURAMOTO_LAYOUT_RELAXATION_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D kuramoto-layout-relaxation quality ratchet",
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
                *KURAMOTO_LAYOUT_RELAXATION_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    data = KURAMOTO_LAYOUT_RELAXATION_COVERAGE_DATA_FILE
    return [
        (
            "kuramoto-layout-relaxation focused coverage",
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
                *KURAMOTO_LAYOUT_RELAXATION_COVERAGE_COHORT,
            ],
        ),
        (
            "kuramoto-layout-relaxation exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hardware/kuramoto_layout_relaxation.py",
            ],
        ),
    ]


__all__ = [
    "KURAMOTO_LAYOUT_RELAXATION_COVERAGE_COHORT",
    "KURAMOTO_LAYOUT_RELAXATION_COVERAGE_DATA_FILE",
    "KURAMOTO_LAYOUT_RELAXATION_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
