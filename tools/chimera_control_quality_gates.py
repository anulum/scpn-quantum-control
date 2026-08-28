# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Chimera-control quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

CHIMERA_CONTROL_SOURCE_COHORT = [
    "src/scpn_quantum_control/chimera_control/__init__.py",
    "src/scpn_quantum_control/chimera_control/evidence.py",
    "src/scpn_quantum_control/chimera_control/objectives.py",
    "src/scpn_quantum_control/chimera_control/observables.py",
    "src/scpn_quantum_control/chimera_control/schema.py",
    "src/scpn_quantum_control/chimera_control/synthetic.py",
    "src/scpn_quantum_control/chimera_control/topology.py",
]
"""Complete production source cohort for the Chimera-control owner."""

CHIMERA_CONTROL_COVERAGE_COHORT = [
    "tests/test_chimera_control_e2e.py",
    "tests/test_chimera_control_evidence.py",
    "tests/test_chimera_control_objectives.py",
    "tests/test_chimera_control_observables.py",
    "tests/test_chimera_control_schema.py",
    "tests/test_chimera_control_synthetic.py",
    "tests/test_chimera_control_topology.py",
]
"""Tests that own exact Chimera-control package coverage."""

CHIMERA_CONTROL_TYPING_RATCHET = [
    *CHIMERA_CONTROL_SOURCE_COHORT,
    "tools/chimera_control_quality_gates.py",
    "tests/test_chimera_control_quality_gate.py",
]
"""Ordered strict-typing cohort for production and gate contracts."""

CHIMERA_CONTROL_DOCSTRING_RATCHET = [
    *CHIMERA_CONTROL_SOURCE_COHORT,
    *CHIMERA_CONTROL_COVERAGE_COHORT,
    "tools/chimera_control_quality_gates.py",
    "tests/test_chimera_control_quality_gate.py",
]
"""Complete production, owner-test, and gate-contract docstring cohort."""

CHIMERA_CONTROL_COVERAGE_DATA_FILE = "/tmp/scpn-qc-chimera-control-quality.coverage"  # nosec B108
"""Isolated coverage database for the Chimera-control owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-chimera-control-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *CHIMERA_CONTROL_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D chimera-control quality ratchet",
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
                *CHIMERA_CONTROL_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact package coverage gates."""
    return [
        (
            "chimera-control focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={CHIMERA_CONTROL_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *CHIMERA_CONTROL_COVERAGE_COHORT,
            ],
        ),
        (
            "chimera-control exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={CHIMERA_CONTROL_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/chimera_control/*.py",
            ],
        ),
    ]


__all__ = [
    "CHIMERA_CONTROL_COVERAGE_COHORT",
    "CHIMERA_CONTROL_COVERAGE_DATA_FILE",
    "CHIMERA_CONTROL_DOCSTRING_RATCHET",
    "CHIMERA_CONTROL_SOURCE_COHORT",
    "CHIMERA_CONTROL_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
