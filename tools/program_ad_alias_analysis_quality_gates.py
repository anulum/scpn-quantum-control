# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Program AD alias-analysis quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PROGRAM_AD_ALIAS_ANALYSIS_SOURCE = "src/scpn_quantum_control/program_ad_alias_analysis.py"
PROGRAM_AD_ALIAS_ANALYSIS_TESTS = [
    "tests/test_program_ad_alias_effects.py",
    "tests/test_program_ad_alias_contracts.py",
]
PROGRAM_AD_ALIAS_ANALYSIS_COVERAGE_COHORT = [*PROGRAM_AD_ALIAS_ANALYSIS_TESTS]
PROGRAM_AD_ALIAS_ANALYSIS_TYPING_RATCHET = [
    PROGRAM_AD_ALIAS_ANALYSIS_SOURCE,
    *PROGRAM_AD_ALIAS_ANALYSIS_TESTS,
    "tools/program_ad_alias_analysis_quality_gates.py",
    "tests/test_program_ad_alias_analysis_quality_gate.py",
]
PROGRAM_AD_ALIAS_ANALYSIS_DOCSTRING_RATCHET = [*PROGRAM_AD_ALIAS_ANALYSIS_TYPING_RATCHET]
PROGRAM_AD_ALIAS_ANALYSIS_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-program-ad-alias-analysis-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-program-ad-alias-analysis-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PROGRAM_AD_ALIAS_ANALYSIS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D program-ad alias-analysis quality ratchet",
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
                *PROGRAM_AD_ALIAS_ANALYSIS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build public alias-analysis execution and exact coverage gates."""
    return [
        (
            "program-ad alias-analysis focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PROGRAM_AD_ALIAS_ANALYSIS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PROGRAM_AD_ALIAS_ANALYSIS_COVERAGE_COHORT,
            ],
        ),
        (
            "program-ad alias-analysis exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PROGRAM_AD_ALIAS_ANALYSIS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/program_ad_alias_analysis.py",
            ],
        ),
    ]


__all__ = [
    "PROGRAM_AD_ALIAS_ANALYSIS_COVERAGE_COHORT",
    "PROGRAM_AD_ALIAS_ANALYSIS_COVERAGE_DATA_FILE",
    "PROGRAM_AD_ALIAS_ANALYSIS_DOCSTRING_RATCHET",
    "PROGRAM_AD_ALIAS_ANALYSIS_SOURCE",
    "PROGRAM_AD_ALIAS_ANALYSIS_TESTS",
    "PROGRAM_AD_ALIAS_ANALYSIS_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
