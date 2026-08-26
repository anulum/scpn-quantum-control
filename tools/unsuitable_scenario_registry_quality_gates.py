# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — unsuitable scenario registry quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
UNSUITABLE_SCENARIO_REGISTRY_QUALITY_RATCHET = [
    "src/scpn_quantum_control/unsuitable_scenario_registry.py",
    "tests/test_unsuitable_scenario_registry.py",
    "tools/unsuitable_scenario_registry_quality_gates.py",
    "tests/test_unsuitable_scenario_registry_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
UNSUITABLE_SCENARIO_REGISTRY_COVERAGE_COHORT = ["tests/test_unsuitable_scenario_registry.py"]
"""Tests that own exact unsuitable-scenario-registry coverage."""
UNSUITABLE_SCENARIO_REGISTRY_COVERAGE_DATA_FILE = ".coverage.unsuitable-scenario-registry-quality"
"""Isolated coverage database for unsuitable-scenario-registry diagnostics."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-unsuitable-scenario-registry-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *UNSUITABLE_SCENARIO_REGISTRY_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D unsuitable-scenario-registry quality ratchet",
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
                *UNSUITABLE_SCENARIO_REGISTRY_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    data = UNSUITABLE_SCENARIO_REGISTRY_COVERAGE_DATA_FILE
    return [
        (
            "unsuitable-scenario-registry focused coverage",
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
                *UNSUITABLE_SCENARIO_REGISTRY_COVERAGE_COHORT,
            ],
        ),
        (
            "unsuitable-scenario-registry exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/unsuitable_scenario_registry.py",
            ],
        ),
    ]


__all__ = [
    "UNSUITABLE_SCENARIO_REGISTRY_COVERAGE_COHORT",
    "UNSUITABLE_SCENARIO_REGISTRY_COVERAGE_DATA_FILE",
    "UNSUITABLE_SCENARIO_REGISTRY_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
