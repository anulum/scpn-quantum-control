# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — public API stability quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

PUBLIC_API_STABILITY_QUALITY_RATCHET = [
    "src/scpn_quantum_control/public_api_stability.py",
    "tests/test_public_api_stability.py",
    "tools/public_api_stability_quality_gates.py",
    "tests/test_public_api_stability_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

PUBLIC_API_STABILITY_COVERAGE_COHORT = ["tests/test_public_api_stability.py"]
"""Tests that own exact public API stability coverage."""

PUBLIC_API_STABILITY_COVERAGE_DATA_FILE = "/tmp/scpn-qc-public-api-stability.coverage"  # nosec B108
"""Isolated coverage database for the public API stability owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Ordered static gates for the stability owner cohort.

    """
    return [
        (
            "mypy-strict-public-api-stability-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PUBLIC_API_STABILITY_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D public-api-stability quality ratchet",
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
                *PUBLIC_API_STABILITY_QUALITY_RATCHET,
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
            "public-api-stability focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PUBLIC_API_STABILITY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PUBLIC_API_STABILITY_COVERAGE_COHORT,
            ],
        ),
        (
            "public-api-stability exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PUBLIC_API_STABILITY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/public_api_stability.py",
            ],
        ),
    ]


__all__ = [
    "PUBLIC_API_STABILITY_COVERAGE_COHORT",
    "PUBLIC_API_STABILITY_COVERAGE_DATA_FILE",
    "PUBLIC_API_STABILITY_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
