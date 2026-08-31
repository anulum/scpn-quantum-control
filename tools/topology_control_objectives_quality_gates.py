# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — persistent-H1 topology-control quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
TOPOLOGY_CONTROL_SOURCES = [
    "src/scpn_quantum_control/topology_control/constraints.py",
    "src/scpn_quantum_control/topology_control/objectives.py",
]
"""Constraint projection and persistent-H1 objective production owner."""
TOPOLOGY_CONTROL_TESTS = [
    "tests/test_topology_control_core.py",
    "tests/test_topology_constraints_guards.py",
    "tests/test_topology_objectives_branches.py",
]
"""Core, guard, projection, degeneracy, and optimizer-interaction tests."""
TOPOLOGY_CONTROL_TYPING_RATCHET = [
    *TOPOLOGY_CONTROL_SOURCES,
    *TOPOLOGY_CONTROL_TESTS,
    "tools/topology_control_objectives_quality_gates.py",
    "tests/test_topology_control_objectives_quality_gate.py",
    "tools/preflight.py",
    "tests/test_preflight_tool.py",
]
"""Production, test, helper, and preflight surfaces held to strict MyPy."""
TOPOLOGY_CONTROL_DOCSTRING_RATCHET = [
    *TOPOLOGY_CONTROL_SOURCES,
    *TOPOLOGY_CONTROL_TESTS,
    "tools/topology_control_objectives_quality_gates.py",
    "tests/test_topology_control_objectives_quality_gate.py",
    "tests/test_preflight_tool.py",
]
"""Whole owner cohort held to complete NumPy documentation."""
TOPOLOGY_CONTROL_COVERAGE_DATA_FILE = "/tmp/scpn-qc-topology-control-quality.coverage"  # nosec B108
TOPOLOGY_CONTROL_COVERAGE_INCLUDE = (
    "*/topology_control/constraints.py,*/topology_control/objectives.py"
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-documentation gates."""
    return [
        (
            "mypy-strict-topology-control-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *TOPOLOGY_CONTROL_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D topology-control quality ratchet",
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
                *TOPOLOGY_CONTROL_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real offline execution and exact combined coverage gates."""
    return [
        (
            "topology-control focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={TOPOLOGY_CONTROL_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *TOPOLOGY_CONTROL_TESTS,
            ],
        ),
        (
            "topology-control exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={TOPOLOGY_CONTROL_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={TOPOLOGY_CONTROL_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "TOPOLOGY_CONTROL_COVERAGE_DATA_FILE",
    "TOPOLOGY_CONTROL_COVERAGE_INCLUDE",
    "TOPOLOGY_CONTROL_DOCSTRING_RATCHET",
    "TOPOLOGY_CONTROL_SOURCES",
    "TOPOLOGY_CONTROL_TESTS",
    "TOPOLOGY_CONTROL_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
