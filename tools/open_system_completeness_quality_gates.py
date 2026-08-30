# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — open-system completeness quality-gate specification
"""Build documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

OPEN_SYSTEM_COMPLETENESS_TYPING_RATCHET = [
    "src/scpn_quantum_control/open_system_mcwf_product.py",
    "src/scpn_quantum_control/phase/tensor_jump.py",
    "tests/test_open_system_mcwf_product.py",
    "tests/test_tensor_jump_contracts.py",
    "tools/open_system_completeness_quality_gates.py",
    "tests/test_open_system_completeness_quality_gate.py",
]
"""Ordered strict-typing cohort."""

OPEN_SYSTEM_COMPLETENESS_DOCSTRING_RATCHET = [
    "src/scpn_quantum_control/open_system_mcwf_product.py",
    "src/scpn_quantum_control/phase/tensor_jump.py",
    "tests/test_tensor_jump_contracts.py",
    "tools/open_system_completeness_quality_gates.py",
    "tests/test_open_system_completeness_quality_gate.py",
]
"""Public source and gate-contract NumPy-docstring cohort."""

OPEN_SYSTEM_COMPLETENESS_COVERAGE_COHORT = [
    "tests/test_open_system_mcwf_product.py",
    "tests/test_tensor_jump.py",
    "tests/test_tensor_jump_contracts.py",
]
"""Tests that own exact open-system completeness statement and branch coverage."""

OPEN_SYSTEM_COMPLETENESS_COVERAGE_DATA_FILE = ".coverage.open-system-completeness-quality"
"""Isolated coverage database for the open-system completeness source owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and public-source NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Ordered static gates for the owned product cohort.

    """
    return [
        (
            "mypy-strict-open-system-completeness-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *OPEN_SYSTEM_COMPLETENESS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D open-system-completeness quality ratchet",
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
                *OPEN_SYSTEM_COMPLETENESS_DOCSTRING_RATCHET,
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
            "open-system-completeness focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={OPEN_SYSTEM_COMPLETENESS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *OPEN_SYSTEM_COMPLETENESS_COVERAGE_COHORT,
            ],
        ),
        (
            "open-system-completeness exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={OPEN_SYSTEM_COMPLETENESS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/open_system_mcwf_product.py,*/phase/tensor_jump.py",
            ],
        ),
    ]


__all__ = [
    "OPEN_SYSTEM_COMPLETENESS_COVERAGE_COHORT",
    "OPEN_SYSTEM_COMPLETENESS_COVERAGE_DATA_FILE",
    "OPEN_SYSTEM_COMPLETENESS_DOCSTRING_RATCHET",
    "OPEN_SYSTEM_COMPLETENESS_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
