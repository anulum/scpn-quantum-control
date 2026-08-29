# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — cloud-native deployment quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

CLOUD_NATIVE_DEPLOYMENT_PRODUCT_SOURCE = (
    "src/scpn_quantum_control/cloud_native_deployment_product.py"
)
CLOUD_NATIVE_MANIFEST_SOURCE = "src/scpn_quantum_control/deployment/cloud_native.py"
CLOUD_NATIVE_DEPLOYMENT_SOURCES = [
    CLOUD_NATIVE_DEPLOYMENT_PRODUCT_SOURCE,
    CLOUD_NATIVE_MANIFEST_SOURCE,
]
CLOUD_NATIVE_DEPLOYMENT_COVERAGE_COHORT = [
    "tests/test_cloud_native_deployment_product.py",
    "tests/test_cloud_native.py",
    "tests/test_cloud_native_branches.py",
]
"""Tests that own exact cloud-native deployment coverage."""

CLOUD_NATIVE_DEPLOYMENT_QUALITY_RATCHET = [
    *CLOUD_NATIVE_DEPLOYMENT_SOURCES,
    *CLOUD_NATIVE_DEPLOYMENT_COVERAGE_COHORT,
    "tools/cloud_native_deployment_product_quality_gates.py",
    "tests/test_cloud_native_deployment_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

CLOUD_NATIVE_DEPLOYMENT_COVERAGE_DATA_FILE = ".coverage.cloud-native-deployment-quality"
"""Isolated coverage database for the cloud-native deployment owner."""


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
            "mypy-strict-cloud-native-deployment-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *CLOUD_NATIVE_DEPLOYMENT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D cloud-native-deployment quality ratchet",
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
                *CLOUD_NATIVE_DEPLOYMENT_QUALITY_RATCHET,
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
            "cloud-native-deployment focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={CLOUD_NATIVE_DEPLOYMENT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *CLOUD_NATIVE_DEPLOYMENT_COVERAGE_COHORT,
            ],
        ),
        (
            "cloud-native-deployment exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={CLOUD_NATIVE_DEPLOYMENT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/cloud_native_deployment_product.py,*/deployment/cloud_native.py",
            ],
        ),
    ]


__all__ = [
    "CLOUD_NATIVE_DEPLOYMENT_COVERAGE_COHORT",
    "CLOUD_NATIVE_DEPLOYMENT_COVERAGE_DATA_FILE",
    "CLOUD_NATIVE_DEPLOYMENT_PRODUCT_SOURCE",
    "CLOUD_NATIVE_DEPLOYMENT_QUALITY_RATCHET",
    "CLOUD_NATIVE_DEPLOYMENT_SOURCES",
    "CLOUD_NATIVE_MANIFEST_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
