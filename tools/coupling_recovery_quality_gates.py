# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — coupling-recovery quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

COUPLING_RECOVERY_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/coupling_time_series_recovery.py",
    "src/scpn_quantum_control/benchmarks/coupling_recovery_evidence.py",
    "tests/test_phase_coupling_time_series_recovery.py",
    "tests/test_coupling_recovery_evidence.py",
    "scripts/export_coupling_recovery_evidence.py",
    "tools/coupling_recovery_quality_gates.py",
    "tests/test_coupling_recovery_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

COUPLING_RECOVERY_COVERAGE_COHORT = [
    "tests/test_phase_coupling_time_series_recovery.py",
    "tests/test_coupling_recovery_evidence.py",
]
"""Tests that own exact coupling-recovery coverage."""

COUPLING_RECOVERY_COVERAGE_DATA_FILE = "/tmp/scpn-qc-coupling-recovery-quality.coverage"  # nosec B108
"""Isolated coverage database for the coupling-recovery owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-coupling-recovery-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *COUPLING_RECOVERY_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D coupling-recovery quality ratchet",
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
                *COUPLING_RECOVERY_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "coupling-recovery focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={COUPLING_RECOVERY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *COUPLING_RECOVERY_COVERAGE_COHORT,
            ],
        ),
        (
            "coupling-recovery exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={COUPLING_RECOVERY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/coupling_time_series_recovery.py,*/benchmarks/coupling_recovery_evidence.py",
            ],
        ),
    ]


__all__ = [
    "COUPLING_RECOVERY_COVERAGE_COHORT",
    "COUPLING_RECOVERY_COVERAGE_DATA_FILE",
    "COUPLING_RECOVERY_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
