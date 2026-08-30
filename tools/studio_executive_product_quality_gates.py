# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio-executive quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
STUDIO_EXECUTIVE_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/studio_executive_product.py",
    "src/scpn_quantum_control/studio/manifest.py",
    "tests/test_studio_executive_product.py",
    "tests/test_studio_manifest.py",
    "tools/studio_executive_product_quality_gates.py",
    "tests/test_studio_executive_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
STUDIO_EXECUTIVE_PRODUCT_COVERAGE_COHORT = [
    "tests/test_studio_executive_product.py",
    "tests/test_studio_manifest.py",
]
"""Tests that own exact Studio-executive product coverage."""
STUDIO_EXECUTIVE_PRODUCT_COVERAGE_DATA_FILE = ".coverage.studio-executive-product-quality"
"""Isolated coverage database for the Studio-executive product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-studio-executive-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *STUDIO_EXECUTIVE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D studio-executive-product quality ratchet",
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
                *STUDIO_EXECUTIVE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "studio-executive-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_EXECUTIVE_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *STUDIO_EXECUTIVE_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "studio-executive-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_EXECUTIVE_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/studio_executive_product.py,*/studio/manifest.py",
            ],
        ),
    ]


__all__ = [
    "STUDIO_EXECUTIVE_PRODUCT_COVERAGE_COHORT",
    "STUDIO_EXECUTIVE_PRODUCT_COVERAGE_DATA_FILE",
    "STUDIO_EXECUTIVE_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
