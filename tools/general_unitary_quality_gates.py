# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — general-unitary quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
GENERAL_UNITARY_SOURCE = "src/scpn_quantum_control/phase/general_unitary.py"
GENERAL_UNITARY_PRIMARY_TEST = "tests/test_phase_general_unitary_gates.py"
GENERAL_UNITARY_COVERAGE_COHORT = [GENERAL_UNITARY_PRIMARY_TEST]
GENERAL_UNITARY_TYPING_RATCHET = [
    GENERAL_UNITARY_SOURCE,
    GENERAL_UNITARY_PRIMARY_TEST,
    "tools/general_unitary_quality_gates.py",
    "tests/test_general_unitary_quality_gate.py",
]
GENERAL_UNITARY_DOCSTRING_RATCHET = [*GENERAL_UNITARY_TYPING_RATCHET]
GENERAL_UNITARY_COVERAGE_DATA_FILE = "/tmp/scpn-qc-general-unitary-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-general-unitary-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *GENERAL_UNITARY_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D general-unitary quality ratchet",
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
                *GENERAL_UNITARY_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build bounded local decomposition execution and exact coverage gates."""
    return [
        (
            "general-unitary focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={GENERAL_UNITARY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *GENERAL_UNITARY_COVERAGE_COHORT,
            ],
        ),
        (
            "general-unitary exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={GENERAL_UNITARY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/general_unitary.py",
            ],
        ),
    ]


__all__ = [
    "GENERAL_UNITARY_COVERAGE_COHORT",
    "GENERAL_UNITARY_COVERAGE_DATA_FILE",
    "GENERAL_UNITARY_DOCSTRING_RATCHET",
    "GENERAL_UNITARY_PRIMARY_TEST",
    "GENERAL_UNITARY_SOURCE",
    "GENERAL_UNITARY_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
