# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — control-stack compose quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

CONTROL_STACK_COMPOSE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/control_stack_compose_product.py",
    "src/scpn_quantum_control/control_stack_runtime_adapters.py",
    "tests/test_control_stack_compose_product.py",
    "tests/test_control_stack_runtime_adapters.py",
    "tools/control_stack_compose_product_quality_gates.py",
    "tests/test_control_stack_compose_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

CONTROL_STACK_COMPOSE_COVERAGE_COHORT = [
    "tests/test_control_stack_compose_product.py",
    "tests/test_control_stack_runtime_adapters.py",
]
"""Tests that own exact control-stack compose product coverage."""

CONTROL_STACK_COMPOSE_COVERAGE_DATA_FILE = ".coverage.control-stack-compose-quality"
"""Isolated coverage database for the control-stack compose owner."""


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
            "mypy-strict-control-stack-compose-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *CONTROL_STACK_COMPOSE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D control-stack-compose quality ratchet",
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
                *CONTROL_STACK_COMPOSE_QUALITY_RATCHET,
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
            "control-stack-compose focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={CONTROL_STACK_COMPOSE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *CONTROL_STACK_COMPOSE_COVERAGE_COHORT,
            ],
        ),
        (
            "control-stack-compose exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={CONTROL_STACK_COMPOSE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/control_stack_compose_product.py,*/control_stack_runtime_adapters.py",
            ],
        ),
    ]


__all__ = [
    "CONTROL_STACK_COMPOSE_COVERAGE_COHORT",
    "CONTROL_STACK_COMPOSE_COVERAGE_DATA_FILE",
    "CONTROL_STACK_COMPOSE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
