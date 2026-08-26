# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — synchronisation witness quality gates
"""Build strict documentation and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
SYNCHRONISATION_WITNESS_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/synchronisation_witness.py",
    "src/scpn_quantum_control/benchmarks/sync_witness_evidence.py",
    "scripts/export_sync_witness_evidence.py",
    "tests/test_phase_synchronisation_witness.py",
    "tests/test_sync_witness_evidence.py",
    "tools/synchronisation_witness_quality_gates.py",
    "tests/test_synchronisation_witness_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
SYNCHRONISATION_WITNESS_COVERAGE_COHORT = [
    "tests/test_phase_synchronisation_witness.py",
    "tests/test_sync_witness_evidence.py",
]
"""Tests that own synchronisation computation and evidence rendering."""
SYNCHRONISATION_WITNESS_COVERAGE_DATA_FILE = ".coverage.synchronisation-witness-quality"
"""Isolated coverage database for synchronisation-witness diagnostics."""
SYNCHRONISATION_WITNESS_COVERAGE_INCLUDE = (
    "*/phase/synchronisation_witness.py,*/benchmarks/sync_witness_evidence.py"
)
"""Exact production modules owned by the coverage threshold."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-synchronisation-witness-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *SYNCHRONISATION_WITNESS_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D synchronisation-witness quality ratchet",
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
                *SYNCHRONISATION_WITNESS_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact production coverage gates."""
    data = SYNCHRONISATION_WITNESS_COVERAGE_DATA_FILE
    return [
        (
            "synchronisation-witness focused coverage",
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
                *SYNCHRONISATION_WITNESS_COVERAGE_COHORT,
            ],
        ),
        (
            "synchronisation-witness exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                f"--include={SYNCHRONISATION_WITNESS_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "SYNCHRONISATION_WITNESS_COVERAGE_COHORT",
    "SYNCHRONISATION_WITNESS_COVERAGE_DATA_FILE",
    "SYNCHRONISATION_WITNESS_COVERAGE_INCLUDE",
    "SYNCHRONISATION_WITNESS_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
