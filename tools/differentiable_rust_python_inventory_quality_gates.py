# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable Rust/Python inventory quality gates
"""Build exact quality gates for the differentiable Rust/Python inventory."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

DIFFERENTIABLE_RUST_PYTHON_INVENTORY_QUALITY_RATCHET = [
    "src/scpn_quantum_control/differentiable_rust_python_inventory.py",
    "tests/test_differentiable_rust_python_inventory.py",
    "tools/differentiable_rust_python_inventory_quality_gates.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

DIFFERENTIABLE_RUST_PYTHON_INVENTORY_COVERAGE_COHORT = [
    "tests/test_differentiable_rust_python_inventory.py",
]
"""Tests that own exact inventory statement and branch coverage."""

DIFFERENTIABLE_RUST_PYTHON_INVENTORY_COVERAGE_DATA_FILE = ".coverage.rust-python-inventory-quality"
"""Isolated coverage database for the inventory owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered static quality gates for the source, owner test, and gate spec.

    """
    return [
        (
            "mypy-strict-differentiable-rust-python-inventory-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DIFFERENTIABLE_RUST_PYTHON_INVENTORY_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D differentiable-rust-python-inventory quality ratchet",
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
                *DIFFERENTIABLE_RUST_PYTHON_INVENTORY_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build isolated exact statement and branch coverage gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Focused execution followed by the exact owner-only report.

    """
    return [
        (
            "differentiable-rust-python-inventory focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_RUST_PYTHON_INVENTORY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DIFFERENTIABLE_RUST_PYTHON_INVENTORY_COVERAGE_COHORT,
            ],
        ),
        (
            "differentiable-rust-python-inventory exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_RUST_PYTHON_INVENTORY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/differentiable_rust_python_inventory.py",
            ],
        ),
    ]


__all__ = [
    "DIFFERENTIABLE_RUST_PYTHON_INVENTORY_COVERAGE_COHORT",
    "DIFFERENTIABLE_RUST_PYTHON_INVENTORY_COVERAGE_DATA_FILE",
    "DIFFERENTIABLE_RUST_PYTHON_INVENTORY_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
