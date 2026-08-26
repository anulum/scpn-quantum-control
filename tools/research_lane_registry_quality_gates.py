# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — research-lane registry quality gates
"""Build strict documentation, evidence-drift, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
RESEARCH_LANE_REGISTRY_QUALITY_RATCHET = [
    "src/scpn_quantum_control/analysis/research_lane_registry.py",
    "tests/test_research_lane_registry.py",
    "scripts/run_research_lane_registry.py",
    "tools/research_lane_registry_quality_gates.py",
    "tests/test_research_lane_registry_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
RESEARCH_LANE_REGISTRY_COVERAGE_COHORT = ["tests/test_research_lane_registry.py"]
"""Tests that own exact research-lane registry coverage."""
RESEARCH_LANE_REGISTRY_COVERAGE_DATA_FILE = ".coverage.research-lane-registry-quality"
"""Isolated coverage database for research-lane registry diagnostics."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing, NumPy-docstring, and evidence-drift gates."""
    return [
        (
            "mypy-strict-research-lane-registry-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *RESEARCH_LANE_REGISTRY_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D research-lane-registry quality ratchet",
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
                *RESEARCH_LANE_REGISTRY_QUALITY_RATCHET,
            ],
        ),
        (
            "research-lane-registry evidence drift",
            [python, "scripts/run_research_lane_registry.py", "--check"],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    data = RESEARCH_LANE_REGISTRY_COVERAGE_DATA_FILE
    return [
        (
            "research-lane-registry focused coverage",
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
                *RESEARCH_LANE_REGISTRY_COVERAGE_COHORT,
            ],
        ),
        (
            "research-lane-registry exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/analysis/research_lane_registry.py",
            ],
        ),
    ]


__all__ = [
    "RESEARCH_LANE_REGISTRY_COVERAGE_COHORT",
    "RESEARCH_LANE_REGISTRY_COVERAGE_DATA_FILE",
    "RESEARCH_LANE_REGISTRY_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
