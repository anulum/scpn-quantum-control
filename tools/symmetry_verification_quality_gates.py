# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Symmetry-verification quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
SYMMETRY_VERIFICATION_SOURCE = "src/scpn_quantum_control/mitigation/symmetry_verification.py"
SYMMETRY_VERIFICATION_PRIMARY_TEST = "tests/test_symmetry_verification.py"
SYMMETRY_VERIFICATION_COVERAGE_COHORT = [SYMMETRY_VERIFICATION_PRIMARY_TEST]
SYMMETRY_VERIFICATION_TYPING_RATCHET = [
    SYMMETRY_VERIFICATION_SOURCE,
    SYMMETRY_VERIFICATION_PRIMARY_TEST,
    "tools/symmetry_verification_quality_gates.py",
    "tests/test_symmetry_verification_quality_gate.py",
]
SYMMETRY_VERIFICATION_DOCSTRING_RATCHET = [*SYMMETRY_VERIFICATION_TYPING_RATCHET]
SYMMETRY_VERIFICATION_COVERAGE_DATA_FILE = "/tmp/scpn-qc-symmetry-verification-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-symmetry-verification-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *SYMMETRY_VERIFICATION_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D symmetry-verification quality ratchet",
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
                *SYMMETRY_VERIFICATION_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build parity-mitigation execution and exact source-coverage gates."""
    return [
        (
            "symmetry-verification focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={SYMMETRY_VERIFICATION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *SYMMETRY_VERIFICATION_COVERAGE_COHORT,
            ],
        ),
        (
            "symmetry-verification exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={SYMMETRY_VERIFICATION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/mitigation/symmetry_verification.py",
            ],
        ),
    ]


__all__ = [
    "SYMMETRY_VERIFICATION_COVERAGE_COHORT",
    "SYMMETRY_VERIFICATION_COVERAGE_DATA_FILE",
    "SYMMETRY_VERIFICATION_DOCSTRING_RATCHET",
    "SYMMETRY_VERIFICATION_PRIMARY_TEST",
    "SYMMETRY_VERIFICATION_SOURCE",
    "SYMMETRY_VERIFICATION_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
