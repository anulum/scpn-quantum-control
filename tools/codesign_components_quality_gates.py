# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — co-design components quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

CODESIGN_COMPONENTS_QUALITY_RATCHET = [
    "src/scpn_quantum_control/codesign/components.py",
    "tests/test_codesign_evaluation.py",
    "tools/codesign_components_quality_gates.py",
    "tests/test_codesign_components_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

CODESIGN_COMPONENTS_COVERAGE_COHORT = [
    "tests/test_codesign_evaluation.py",
]
"""Tests that own exact co-design component coverage."""

CODESIGN_COMPONENTS_COVERAGE_DATA_FILE = "/tmp/scpn-qc-codesign-components-quality.coverage"  # nosec B108
"""Isolated coverage database for the co-design components owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-codesign-components-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *CODESIGN_COMPONENTS_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D codesign-components quality ratchet",
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
                *CODESIGN_COMPONENTS_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "codesign-components focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={CODESIGN_COMPONENTS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *CODESIGN_COMPONENTS_COVERAGE_COHORT,
            ],
        ),
        (
            "codesign-components exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={CODESIGN_COMPONENTS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/codesign/components.py",
            ],
        ),
    ]


__all__ = [
    "CODESIGN_COMPONENTS_COVERAGE_COHORT",
    "CODESIGN_COMPONENTS_COVERAGE_DATA_FILE",
    "CODESIGN_COMPONENTS_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
