# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode product quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PHASE_QNODE_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase_qnode_product.py",
    "tests/test_phase_qnode_product.py",
    "tools/phase_qnode_product_quality_gates.py",
    "tests/test_phase_qnode_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
PHASE_QNODE_PRODUCT_COVERAGE_COHORT = ["tests/test_phase_qnode_product.py"]
"""Tests that own exact Phase-QNode product coverage."""
PHASE_QNODE_PRODUCT_COVERAGE_DATA_FILE = "/tmp/scpn-qc-phase-qnode-product-quality.coverage"  # nosec B108
"""Isolated coverage database for the Phase-QNode product."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-phase-qnode-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PHASE_QNODE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D phase-qnode-product quality ratchet",
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
                *PHASE_QNODE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    data = PHASE_QNODE_PRODUCT_COVERAGE_DATA_FILE
    return [
        (
            "phase-qnode-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PHASE_QNODE_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "phase-qnode-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase_qnode_product.py",
            ],
        ),
    ]


__all__ = [
    "PHASE_QNODE_PRODUCT_COVERAGE_COHORT",
    "PHASE_QNODE_PRODUCT_COVERAGE_DATA_FILE",
    "PHASE_QNODE_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
