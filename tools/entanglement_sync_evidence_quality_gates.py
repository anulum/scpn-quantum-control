# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — entanglement-sync evidence quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

ENTANGLEMENT_SYNC_EVIDENCE_COVERAGE_INCLUDE = (
    "*/entanglement_sync_evidence.py,*/entanglement_enhanced_sync.py,"
    "*/sync_entanglement_witness.py"
)

ENTANGLEMENT_SYNC_EVIDENCE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/analysis/entanglement_sync_evidence.py",
    "src/scpn_quantum_control/analysis/entanglement_enhanced_sync.py",
    "src/scpn_quantum_control/analysis/sync_entanglement_witness.py",
    "tests/test_entanglement_sync_evidence.py",
    "tests/test_entanglement_enhanced_sync.py",
    "tests/test_sync_entanglement_witness.py",
    "scripts/run_entanglement_sync_evidence.py",
    "src/scpn_quantum_control/analysis/quantum_speed_limit.py",
    "src/scpn_quantum_control/advantage_language_protocol.py",
    "tests/test_advantage_language_protocol.py",
    "tools/entanglement_sync_evidence_quality_gates.py",
    "tests/test_entanglement_sync_evidence_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

ENTANGLEMENT_SYNC_EVIDENCE_COVERAGE_COHORT = [
    "tests/test_entanglement_sync_evidence.py",
    "tests/test_entanglement_enhanced_sync.py",
    "tests/test_sync_entanglement_witness.py",
    "tests/test_analysis_entanglement_sync_contracts.py",
    "tests/test_phase_dynamics_contracts.py",
]
"""Tests that own exact entanglement-sync evidence coverage."""

ENTANGLEMENT_SYNC_EVIDENCE_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-entanglement-sync-evidence-quality.coverage"  # nosec B108
)
"""Isolated coverage database for the entanglement-sync evidence owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-entanglement-sync-evidence-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ENTANGLEMENT_SYNC_EVIDENCE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D entanglement-sync-evidence quality ratchet",
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
                'lint.pydocstyle.convention = "numpy"',
                *ENTANGLEMENT_SYNC_EVIDENCE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "entanglement-sync-evidence focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ENTANGLEMENT_SYNC_EVIDENCE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ENTANGLEMENT_SYNC_EVIDENCE_COVERAGE_COHORT,
            ],
        ),
        (
            "entanglement-sync-evidence exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ENTANGLEMENT_SYNC_EVIDENCE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={ENTANGLEMENT_SYNC_EVIDENCE_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "ENTANGLEMENT_SYNC_EVIDENCE_COVERAGE_COHORT",
    "ENTANGLEMENT_SYNC_EVIDENCE_COVERAGE_DATA_FILE",
    "ENTANGLEMENT_SYNC_EVIDENCE_COVERAGE_INCLUDE",
    "ENTANGLEMENT_SYNC_EVIDENCE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
