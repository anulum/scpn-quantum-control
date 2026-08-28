# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA topology optimiser quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
DLA_TOPOLOGY_OPTIMIZER_SOURCE = "src/scpn_quantum_control/dla_topology_control/optimizer.py"
DLA_TOPOLOGY_OPTIMIZER_COVERAGE_COHORT = [
    "tests/test_dla_topology_control_optimizer.py",
]
DLA_TOPOLOGY_OPTIMIZER_TYPING_RATCHET = [
    DLA_TOPOLOGY_OPTIMIZER_SOURCE,
    "tools/dla_topology_optimizer_quality_gates.py",
    "tests/test_dla_topology_optimizer_quality_gate.py",
]
DLA_TOPOLOGY_OPTIMIZER_DOCSTRING_RATCHET = [
    DLA_TOPOLOGY_OPTIMIZER_SOURCE,
    *DLA_TOPOLOGY_OPTIMIZER_COVERAGE_COHORT,
    "tools/dla_topology_optimizer_quality_gates.py",
    "tests/test_dla_topology_optimizer_quality_gate.py",
]
DLA_TOPOLOGY_OPTIMIZER_COVERAGE_DATA_FILE = "/tmp/scpn-qc-dla-topology-optimizer-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-dla-topology-optimizer-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DLA_TOPOLOGY_OPTIMIZER_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D dla-topology-optimizer quality ratchet",
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
                *DLA_TOPOLOGY_OPTIMIZER_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    return [
        (
            "dla-topology-optimizer focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DLA_TOPOLOGY_OPTIMIZER_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DLA_TOPOLOGY_OPTIMIZER_COVERAGE_COHORT,
            ],
        ),
        (
            "dla-topology-optimizer exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DLA_TOPOLOGY_OPTIMIZER_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/dla_topology_control/optimizer.py",
            ],
        ),
    ]


__all__ = [
    "DLA_TOPOLOGY_OPTIMIZER_COVERAGE_COHORT",
    "DLA_TOPOLOGY_OPTIMIZER_COVERAGE_DATA_FILE",
    "DLA_TOPOLOGY_OPTIMIZER_DOCSTRING_RATCHET",
    "DLA_TOPOLOGY_OPTIMIZER_SOURCE",
    "DLA_TOPOLOGY_OPTIMIZER_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
