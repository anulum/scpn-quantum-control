# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — backend-selector quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

BACKEND_SELECTOR_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/backend_selector.py",
    "tests/test_backend_selector.py",
    "tools/backend_selector_quality_gates.py",
    "tests/test_backend_selector_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

BACKEND_SELECTOR_COVERAGE_COHORT = ["tests/test_backend_selector.py"]
"""Tests that own exact backend-selector coverage."""

BACKEND_SELECTOR_COVERAGE_DATA_FILE = "/tmp/scpn-qc-backend-selector-quality.coverage"  # nosec B108
"""Isolated coverage database for the backend-selector owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-backend-selector-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *BACKEND_SELECTOR_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D backend-selector quality ratchet",
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
                *BACKEND_SELECTOR_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "backend-selector focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={BACKEND_SELECTOR_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *BACKEND_SELECTOR_COVERAGE_COHORT,
            ],
        ),
        (
            "backend-selector exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={BACKEND_SELECTOR_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/backend_selector.py",
            ],
        ),
    ]


__all__ = [
    "BACKEND_SELECTOR_COVERAGE_COHORT",
    "BACKEND_SELECTOR_COVERAGE_DATA_FILE",
    "BACKEND_SELECTOR_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
