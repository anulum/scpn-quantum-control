# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — campaign-harness product quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

CAMPAIGN_HARNESS_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/campaign_harness_product.py",
    "tests/test_campaign_harness_product.py",
    "tools/campaign_harness_product_quality_gates.py",
    "tests/test_campaign_harness_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

CAMPAIGN_HARNESS_PRODUCT_COVERAGE_COHORT = ["tests/test_campaign_harness_product.py"]
"""Tests that own exact campaign-harness product coverage."""

CAMPAIGN_HARNESS_PRODUCT_COVERAGE_DATA_FILE = ".coverage.campaign-harness-product-quality"
"""Isolated coverage database for the campaign-harness product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Ordered static gates for the product owner cohort.

    """
    return [
        (
            "mypy-strict-campaign-harness-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *CAMPAIGN_HARNESS_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D campaign-harness-product quality ratchet",
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
                *CAMPAIGN_HARNESS_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Focused owner execution followed by the exact report.

    """
    return [
        (
            "campaign-harness-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={CAMPAIGN_HARNESS_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *CAMPAIGN_HARNESS_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "campaign-harness-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={CAMPAIGN_HARNESS_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/campaign_harness_product.py",
            ],
        ),
    ]


__all__ = [
    "CAMPAIGN_HARNESS_PRODUCT_COVERAGE_COHORT",
    "CAMPAIGN_HARNESS_PRODUCT_COVERAGE_DATA_FILE",
    "CAMPAIGN_HARNESS_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
