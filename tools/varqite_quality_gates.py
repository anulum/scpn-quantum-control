# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — VarQITE quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
VARQITE_SOURCE = "src/scpn_quantum_control/phase/varqite.py"
VARQITE_OWNER_TEST = "tests/test_varqite.py"
VARQITE_DOWNSTREAM_TEST = "tests/test_phase_dynamics_contracts.py"
VARQITE_TYPING_RATCHET = [
    VARQITE_SOURCE,
    VARQITE_OWNER_TEST,
    "tools/varqite_quality_gates.py",
    "tests/test_varqite_quality_gate.py",
]
VARQITE_DOCSTRING_RATCHET = [*VARQITE_TYPING_RATCHET]
VARQITE_COVERAGE_COHORT = [VARQITE_OWNER_TEST, VARQITE_DOWNSTREAM_TEST]
VARQITE_COVERAGE_DATA_FILE = "/tmp/scpn-qc-varqite-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-varqite-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *VARQITE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D VarQITE quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--select",
                "D,D413,D417,D420",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *VARQITE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real offline execution and exact source coverage gates."""
    return [
        (
            "VarQITE focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={VARQITE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *VARQITE_COVERAGE_COHORT,
            ],
        ),
        (
            "VarQITE exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={VARQITE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/varqite.py",
            ],
        ),
    ]


__all__ = [
    "VARQITE_COVERAGE_COHORT",
    "VARQITE_COVERAGE_DATA_FILE",
    "VARQITE_DOCSTRING_RATCHET",
    "VARQITE_DOWNSTREAM_TEST",
    "VARQITE_OWNER_TEST",
    "VARQITE_SOURCE",
    "VARQITE_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
