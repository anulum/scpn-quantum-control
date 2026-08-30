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
SYNCHRONISATION_WITNESS_SOURCE = "src/scpn_quantum_control/phase/synchronisation_witness.py"
SYNC_WITNESS_EVIDENCE_SOURCE = "src/scpn_quantum_control/benchmarks/sync_witness_evidence.py"
STUDIO_ANALYSE_SOURCE = "src/scpn_quantum_control/studio/executive_analyse.py"
STUDIO_EXECUTIVE_CLI_SOURCE = "src/scpn_quantum_control/studio/executive_cli.py"
PHASE_SYNCHRONISATION_WITNESS_TEST = "tests/test_phase_synchronisation_witness.py"
SYNC_WITNESS_EVIDENCE_TEST = "tests/test_sync_witness_evidence.py"
STUDIO_ANALYSE_TEST = "tests/test_studio_executive_analyse.py"
STUDIO_EXECUTIVE_CLI_TEST = "tests/test_studio_executive_cli.py"
SYNCHRONISATION_WITNESS_TYPING_RATCHET = [
    SYNCHRONISATION_WITNESS_SOURCE,
    SYNC_WITNESS_EVIDENCE_SOURCE,
    "scripts/export_sync_witness_evidence.py",
    STUDIO_ANALYSE_SOURCE,
    STUDIO_EXECUTIVE_CLI_SOURCE,
    PHASE_SYNCHRONISATION_WITNESS_TEST,
    SYNC_WITNESS_EVIDENCE_TEST,
    STUDIO_ANALYSE_TEST,
    STUDIO_EXECUTIVE_CLI_TEST,
    "tools/synchronisation_witness_quality_gates.py",
    "tests/test_synchronisation_witness_quality_gate.py",
]
"""Ordered strict-typing cohort for the witness and public Studio consumer."""
SYNCHRONISATION_WITNESS_DOCSTRING_RATCHET = [
    SYNCHRONISATION_WITNESS_SOURCE,
    SYNC_WITNESS_EVIDENCE_SOURCE,
    "scripts/export_sync_witness_evidence.py",
    STUDIO_ANALYSE_SOURCE,
    PHASE_SYNCHRONISATION_WITNESS_TEST,
    SYNC_WITNESS_EVIDENCE_TEST,
    STUDIO_ANALYSE_TEST,
    "tools/synchronisation_witness_quality_gates.py",
    "tests/test_synchronisation_witness_quality_gate.py",
]
"""Ordered complete NumPy-docstring cohort for the direct owner."""
SYNCHRONISATION_WITNESS_COVERAGE_COHORT = [
    PHASE_SYNCHRONISATION_WITNESS_TEST,
    SYNC_WITNESS_EVIDENCE_TEST,
    STUDIO_ANALYSE_TEST,
    STUDIO_EXECUTIVE_CLI_TEST,
]
"""Tests that own synchronisation computation, evidence, and Studio routing."""
SYNCHRONISATION_WITNESS_COVERAGE_DATA_FILE = (  # nosec B108
    "/tmp/scpn-qc-synchronisation-witness-quality.coverage"
)
"""Isolated coverage database for synchronisation-witness diagnostics."""
SYNCHRONISATION_WITNESS_COVERAGE_INCLUDE = ",".join(
    [
        "*/phase/synchronisation_witness.py",
        "*/benchmarks/sync_witness_evidence.py",
        "*/studio/executive_analyse.py",
        "*/studio/executive_cli.py",
    ]
)
"""Exact production modules owned by the coverage threshold."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-synchronisation-witness-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *SYNCHRONISATION_WITNESS_TYPING_RATCHET,
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
                "--preview",
                "--select",
                "D,D413,D417,D420",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *SYNCHRONISATION_WITNESS_DOCSTRING_RATCHET,
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
    "PHASE_SYNCHRONISATION_WITNESS_TEST",
    "STUDIO_ANALYSE_SOURCE",
    "STUDIO_ANALYSE_TEST",
    "STUDIO_EXECUTIVE_CLI_SOURCE",
    "STUDIO_EXECUTIVE_CLI_TEST",
    "SYNCHRONISATION_WITNESS_COVERAGE_COHORT",
    "SYNCHRONISATION_WITNESS_COVERAGE_DATA_FILE",
    "SYNCHRONISATION_WITNESS_COVERAGE_INCLUDE",
    "SYNCHRONISATION_WITNESS_DOCSTRING_RATCHET",
    "SYNCHRONISATION_WITNESS_SOURCE",
    "SYNCHRONISATION_WITNESS_TYPING_RATCHET",
    "SYNC_WITNESS_EVIDENCE_SOURCE",
    "SYNC_WITNESS_EVIDENCE_TEST",
    "build_coverage_gates",
    "build_static_quality_gates",
]
