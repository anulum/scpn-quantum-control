# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — advantage-language quality-gate specification
"""Build strict documentation, typing, and coverage gates for BL-19."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
ADVANTAGE_LANGUAGE_PROTOCOL_QUALITY_RATCHET = [
    "src/scpn_quantum_control/advantage_language_protocol.py",
    "tests/test_advantage_language_protocol.py",
    "tools/advantage_language_protocol_quality_gates.py",
    "tests/test_advantage_language_protocol_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_COHORT = ["tests/test_advantage_language_protocol.py"]
"""Tests that own exact advantage-language protocol coverage."""
ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_DATA_FILE = ".coverage.advantage-language-protocol-quality"
"""Isolated coverage database for the advantage-language protocol owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-advantage-language-protocol-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ADVANTAGE_LANGUAGE_PROTOCOL_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D advantage-language-protocol quality ratchet",
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
                *ADVANTAGE_LANGUAGE_PROTOCOL_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "advantage-language-protocol focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_COHORT,
            ],
        ),
        (
            "advantage-language-protocol exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/advantage_language_protocol.py",
            ],
        ),
    ]


__all__ = [
    "ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_COHORT",
    "ADVANTAGE_LANGUAGE_PROTOCOL_COVERAGE_DATA_FILE",
    "ADVANTAGE_LANGUAGE_PROTOCOL_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
