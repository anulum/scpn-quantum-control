# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase-artifact quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PHASE_ARTIFACT_SOURCE = "src/scpn_quantum_control/bridge/phase_artifact.py"
SCPN_UPDE_EDGE_SOURCE = "src/scpn_quantum_control/bridge/scpn_upde_edge.py"
PHASE_ARTIFACT_COVERAGE_COHORT = [
    "tests/test_phase_artifact.py",
    "tests/test_phase_artifact_errors.py",
    "tests/test_phase_artifact_fuzz.py",
    "tests/test_scpn_upde_edge.py",
]
PHASE_ARTIFACT_TYPING_RATCHET = [
    PHASE_ARTIFACT_SOURCE,
    SCPN_UPDE_EDGE_SOURCE,
    *PHASE_ARTIFACT_COVERAGE_COHORT,
    "tools/phase_artifact_quality_gates.py",
    "tests/test_phase_artifact_quality_gate.py",
]
PHASE_ARTIFACT_DOCSTRING_RATCHET = [
    *PHASE_ARTIFACT_TYPING_RATCHET,
]
PHASE_ARTIFACT_COVERAGE_DATA_FILE = "/tmp/scpn-qc-phase-artifact-quality.coverage"  # nosec B108
PHASE_ARTIFACT_COVERAGE_INCLUDE = "*/bridge/phase_artifact.py,*/bridge/scpn_upde_edge.py"


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-phase-artifact-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PHASE_ARTIFACT_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D phase-artifact quality ratchet",
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
                *PHASE_ARTIFACT_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real artifact execution and exact source-coverage gates."""
    return [
        (
            "phase-artifact focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_ARTIFACT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PHASE_ARTIFACT_COVERAGE_COHORT,
            ],
        ),
        (
            "phase-artifact exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_ARTIFACT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={PHASE_ARTIFACT_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "PHASE_ARTIFACT_COVERAGE_COHORT",
    "PHASE_ARTIFACT_COVERAGE_DATA_FILE",
    "PHASE_ARTIFACT_COVERAGE_INCLUDE",
    "PHASE_ARTIFACT_DOCSTRING_RATCHET",
    "PHASE_ARTIFACT_SOURCE",
    "PHASE_ARTIFACT_TYPING_RATCHET",
    "SCPN_UPDE_EDGE_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
