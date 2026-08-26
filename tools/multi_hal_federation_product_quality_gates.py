# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — multi-HAL federation quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
MULTI_HAL_FEDERATION_QUALITY_RATCHET = [
    "src/scpn_quantum_control/multi_hal_federation_product.py",
    "tests/test_multi_hal_federation_product.py",
    "tools/multi_hal_federation_product_quality_gates.py",
    "tests/test_multi_hal_federation_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
MULTI_HAL_FEDERATION_COVERAGE_COHORT = ["tests/test_multi_hal_federation_product.py"]
"""Tests that own exact multi-HAL federation coverage."""
MULTI_HAL_FEDERATION_COVERAGE_DATA_FILE = ".coverage.multi-hal-federation-quality"
"""Isolated coverage database for the multi-HAL federation owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-multi-hal-federation-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *MULTI_HAL_FEDERATION_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D multi-hal-federation quality ratchet",
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
                *MULTI_HAL_FEDERATION_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "multi-hal-federation focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={MULTI_HAL_FEDERATION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *MULTI_HAL_FEDERATION_COVERAGE_COHORT,
            ],
        ),
        (
            "multi-hal-federation exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={MULTI_HAL_FEDERATION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/multi_hal_federation_product.py",
            ],
        ),
    ]


__all__ = [
    "MULTI_HAL_FEDERATION_COVERAGE_COHORT",
    "MULTI_HAL_FEDERATION_COVERAGE_DATA_FILE",
    "MULTI_HAL_FEDERATION_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
