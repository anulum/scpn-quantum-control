# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — ENAQT evidence quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

ENAQT_EVIDENCE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/analysis/enaqt_evidence.py",
    "tests/test_enaqt_evidence.py",
    "scripts/run_enaqt_evidence.py",
    "tools/enaqt_evidence_quality_gates.py",
    "tests/test_enaqt_evidence_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

ENAQT_EVIDENCE_COVERAGE_COHORT = [
    "tests/test_enaqt_evidence.py",
]
"""Tests that own exact ENAQT evidence coverage."""

ENAQT_EVIDENCE_COVERAGE_DATA_FILE = "/tmp/scpn-qc-enaqt-evidence-quality.coverage"  # nosec B108
"""Isolated coverage database for the ENAQT evidence owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-enaqt-evidence-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ENAQT_EVIDENCE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D enaqt-evidence quality ratchet",
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
                *ENAQT_EVIDENCE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "enaqt-evidence focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ENAQT_EVIDENCE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ENAQT_EVIDENCE_COVERAGE_COHORT,
            ],
        ),
        (
            "enaqt-evidence exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ENAQT_EVIDENCE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/analysis/enaqt_evidence.py",
            ],
        ),
    ]


__all__ = [
    "ENAQT_EVIDENCE_COVERAGE_COHORT",
    "ENAQT_EVIDENCE_COVERAGE_DATA_FILE",
    "ENAQT_EVIDENCE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
