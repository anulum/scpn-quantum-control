# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — metamorphic-AD quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
METAMORPHIC_AD_VERIFICATION_QUALITY_RATCHET = [
    "src/scpn_quantum_control/metamorphic_ad_verification.py",
    "tests/test_metamorphic_ad_verification.py",
    "tools/metamorphic_ad_verification_quality_gates.py",
    "tests/test_metamorphic_ad_verification_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
METAMORPHIC_AD_VERIFICATION_COVERAGE_COHORT = ["tests/test_metamorphic_ad_verification.py"]
"""Tests that own exact metamorphic-AD coverage."""
METAMORPHIC_AD_VERIFICATION_COVERAGE_DATA_FILE = ".coverage.metamorphic-ad-verification-quality"
"""Isolated coverage database for the metamorphic-AD owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-metamorphic-ad-verification-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *METAMORPHIC_AD_VERIFICATION_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D metamorphic-ad-verification quality ratchet",
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
                *METAMORPHIC_AD_VERIFICATION_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "metamorphic-ad-verification focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={METAMORPHIC_AD_VERIFICATION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *METAMORPHIC_AD_VERIFICATION_COVERAGE_COHORT,
            ],
        ),
        (
            "metamorphic-ad-verification exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={METAMORPHIC_AD_VERIFICATION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/metamorphic_ad_verification.py",
            ],
        ),
    ]


__all__ = [
    "METAMORPHIC_AD_VERIFICATION_COVERAGE_COHORT",
    "METAMORPHIC_AD_VERIFICATION_COVERAGE_DATA_FILE",
    "METAMORPHIC_AD_VERIFICATION_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
