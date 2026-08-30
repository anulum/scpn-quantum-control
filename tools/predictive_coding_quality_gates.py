# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Predictive-coding quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PREDICTIVE_CODING_SOURCE = "src/scpn_quantum_control/fep/predictive_coding.py"
PREDICTIVE_CODING_PRIMARY_TEST = "tests/test_fep.py"
PREDICTIVE_CODING_COVERAGE_COHORT = [PREDICTIVE_CODING_PRIMARY_TEST]
PREDICTIVE_CODING_TYPING_RATCHET = [
    PREDICTIVE_CODING_SOURCE,
    PREDICTIVE_CODING_PRIMARY_TEST,
    "tools/predictive_coding_quality_gates.py",
    "tests/test_predictive_coding_quality_gate.py",
]
PREDICTIVE_CODING_DOCSTRING_RATCHET = [*PREDICTIVE_CODING_TYPING_RATCHET]
PREDICTIVE_CODING_COVERAGE_DATA_FILE = "/tmp/scpn-qc-predictive-coding-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-predictive-coding-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PREDICTIVE_CODING_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D predictive-coding quality ratchet",
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
                *PREDICTIVE_CODING_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build connected FEP execution and exact source-coverage gates."""
    return [
        (
            "predictive-coding focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PREDICTIVE_CODING_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PREDICTIVE_CODING_COVERAGE_COHORT,
            ],
        ),
        (
            "predictive-coding exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PREDICTIVE_CODING_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/fep/predictive_coding.py",
            ],
        ),
    ]


__all__ = [
    "PREDICTIVE_CODING_COVERAGE_COHORT",
    "PREDICTIVE_CODING_COVERAGE_DATA_FILE",
    "PREDICTIVE_CODING_DOCSTRING_RATCHET",
    "PREDICTIVE_CODING_PRIMARY_TEST",
    "PREDICTIVE_CODING_SOURCE",
    "PREDICTIVE_CODING_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
