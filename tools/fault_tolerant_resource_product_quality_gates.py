# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — fault-tolerant resource product quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
FAULT_TOLERANT_RESOURCE_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/fault_tolerant_resource_product.py",
    "tests/test_fault_tolerant_resource_product.py",
    "tools/fault_tolerant_resource_product_quality_gates.py",
    "tests/test_fault_tolerant_resource_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
FAULT_TOLERANT_RESOURCE_PRODUCT_COVERAGE_COHORT = ["tests/test_fault_tolerant_resource_product.py"]
"""Tests that own exact fault-tolerant resource product coverage."""
FAULT_TOLERANT_RESOURCE_PRODUCT_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-fault-tolerant-resource-product.coverage"  # nosec B108
)
"""Isolated coverage database for the fault-tolerant resource product."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-fault-tolerant-resource-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *FAULT_TOLERANT_RESOURCE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D fault-tolerant-resource-product quality ratchet",
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
                *FAULT_TOLERANT_RESOURCE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "fault-tolerant-resource-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={FAULT_TOLERANT_RESOURCE_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *FAULT_TOLERANT_RESOURCE_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "fault-tolerant-resource-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={FAULT_TOLERANT_RESOURCE_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/fault_tolerant_resource_product.py",
            ],
        ),
    ]


__all__ = [
    "FAULT_TOLERANT_RESOURCE_PRODUCT_COVERAGE_COHORT",
    "FAULT_TOLERANT_RESOURCE_PRODUCT_COVERAGE_DATA_FILE",
    "FAULT_TOLERANT_RESOURCE_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
