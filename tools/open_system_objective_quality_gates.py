# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — open-system-objective quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
OPEN_SYSTEM_OBJECTIVE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/benchmarks/open_system_objective_evidence.py",
    "src/scpn_quantum_control/phase/open_system_objectives.py",
    "tests/test_open_system_objective_evidence.py",
    "tests/test_phase_open_system_objectives.py",
    "scripts/export_open_system_objective_evidence.py",
    "tools/open_system_objective_quality_gates.py",
    "tests/test_open_system_objective_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
OPEN_SYSTEM_OBJECTIVE_COVERAGE_COHORT = [
    "tests/test_open_system_objective_evidence.py",
    "tests/test_phase_open_system_objectives.py",
]
"""Tests that own exact open-system-objective coverage."""
OPEN_SYSTEM_OBJECTIVE_COVERAGE_DATA_FILE = "/tmp/scpn-qc-open-system-objective.coverage"  # nosec B108
"""Isolated coverage database for the open-system-objective owner."""
OPEN_SYSTEM_OBJECTIVE_COVERAGE_INCLUDE = (
    "*/benchmarks/open_system_objective_evidence.py,*/phase/open_system_objectives.py"
)
"""Exact two-source coverage boundary."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-open-system-objective-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *OPEN_SYSTEM_OBJECTIVE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D open-system-objective quality ratchet",
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
                *OPEN_SYSTEM_OBJECTIVE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact two-source coverage gates."""
    return [
        (
            "open-system-objective focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={OPEN_SYSTEM_OBJECTIVE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *OPEN_SYSTEM_OBJECTIVE_COVERAGE_COHORT,
            ],
        ),
        (
            "open-system-objective exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={OPEN_SYSTEM_OBJECTIVE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={OPEN_SYSTEM_OBJECTIVE_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "OPEN_SYSTEM_OBJECTIVE_COVERAGE_COHORT",
    "OPEN_SYSTEM_OBJECTIVE_COVERAGE_DATA_FILE",
    "OPEN_SYSTEM_OBJECTIVE_COVERAGE_INCLUDE",
    "OPEN_SYSTEM_OBJECTIVE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
