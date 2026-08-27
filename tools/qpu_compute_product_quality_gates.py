# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QPU-compute quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
QPU_COMPUTE_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/qpu_compute_product.py",
    "tests/test_qpu_compute_product.py",
    "tools/qpu_compute_product_quality_gates.py",
    "tests/test_qpu_compute_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
QPU_COMPUTE_PRODUCT_COVERAGE_COHORT = ["tests/test_qpu_compute_product.py"]
"""Tests that own exact qpu-compute product coverage."""
QPU_COMPUTE_PRODUCT_COVERAGE_DATA_FILE = "/tmp/scpn-qc-qpu-compute-product-quality.coverage"  # nosec B108
"""Isolated coverage database for the qpu-compute product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-qpu-compute-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *QPU_COMPUTE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D qpu-compute-product quality ratchet",
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
                *QPU_COMPUTE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "qpu-compute-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={QPU_COMPUTE_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *QPU_COMPUTE_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "qpu-compute-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={QPU_COMPUTE_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/qpu_compute_product.py",
            ],
        ),
    ]


__all__ = [
    "QPU_COMPUTE_PRODUCT_COVERAGE_COHORT",
    "QPU_COMPUTE_PRODUCT_COVERAGE_DATA_FILE",
    "QPU_COMPUTE_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
