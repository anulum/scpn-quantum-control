# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MPS-evolution quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
MPS_EVOLUTION_SOURCE = "src/scpn_quantum_control/phase/mps_evolution.py"
MPS_EVOLUTION_COVERAGE_COHORT = [
    "tests/test_mps_evolution.py",
    "tests/test_backend_selector.py",
    "tests/test_classical_baselines.py",
]
MPS_EVOLUTION_TYPING_RATCHET = [
    MPS_EVOLUTION_SOURCE,
    "tools/mps_evolution_quality_gates.py",
    "tests/test_mps_evolution_quality_gate.py",
]
MPS_EVOLUTION_DOCSTRING_RATCHET = [
    MPS_EVOLUTION_SOURCE,
    "tests/test_mps_evolution.py",
    "tools/mps_evolution_quality_gates.py",
    "tests/test_mps_evolution_quality_gate.py",
]
MPS_EVOLUTION_COVERAGE_DATA_FILE = "/tmp/scpn-qc-mps-evolution-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-mps-evolution-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *MPS_EVOLUTION_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D MPS-evolution quality ratchet",
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
                *MPS_EVOLUTION_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build connected real-quimb execution and exact coverage gates."""
    return [
        (
            "MPS-evolution focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={MPS_EVOLUTION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *MPS_EVOLUTION_COVERAGE_COHORT,
            ],
        ),
        (
            "MPS-evolution exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={MPS_EVOLUTION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/mps_evolution.py",
            ],
        ),
    ]


__all__ = [
    "MPS_EVOLUTION_COVERAGE_COHORT",
    "MPS_EVOLUTION_COVERAGE_DATA_FILE",
    "MPS_EVOLUTION_DOCSTRING_RATCHET",
    "MPS_EVOLUTION_SOURCE",
    "MPS_EVOLUTION_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
