# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hermetic-reproduction quality-gate specification
"""Build strict documentation, typing, and coverage gates for BL-19."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
HERMETIC_REPRODUCTION_KIT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/hermetic_reproduction_kit.py",
    "tests/test_hermetic_reproduction_kit.py",
    "tools/hermetic_reproduction_kit_quality_gates.py",
    "tests/test_hermetic_reproduction_kit_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
HERMETIC_REPRODUCTION_KIT_COVERAGE_COHORT = ["tests/test_hermetic_reproduction_kit.py"]
"""Tests that own exact hermetic-kit coverage."""
HERMETIC_REPRODUCTION_KIT_COVERAGE_DATA_FILE = ".coverage.hermetic-reproduction-kit-quality"
"""Isolated coverage database for the hermetic-kit owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-hermetic-reproduction-kit-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *HERMETIC_REPRODUCTION_KIT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D hermetic-reproduction-kit quality ratchet",
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
                *HERMETIC_REPRODUCTION_KIT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "hermetic-reproduction-kit focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={HERMETIC_REPRODUCTION_KIT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *HERMETIC_REPRODUCTION_KIT_COVERAGE_COHORT,
            ],
        ),
        (
            "hermetic-reproduction-kit exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={HERMETIC_REPRODUCTION_KIT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hermetic_reproduction_kit.py",
            ],
        ),
    ]


__all__ = [
    "HERMETIC_REPRODUCTION_KIT_COVERAGE_COHORT",
    "HERMETIC_REPRODUCTION_KIT_COVERAGE_DATA_FILE",
    "HERMETIC_REPRODUCTION_KIT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
