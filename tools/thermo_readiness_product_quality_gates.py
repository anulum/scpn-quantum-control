# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — thermo-readiness product quality-gate specification
"""Build documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

THERMO_READINESS_PRODUCT_SOURCE = "src/scpn_quantum_control/thermo_readiness_product.py"
"""Public fail-closed product surface."""
QUANTUM_THERMO_READINESS_SOURCE = "src/scpn_quantum_control/thermodynamics/readiness.py"
"""Ambient S9 no-submit thermodynamics protocol model."""
QUANTUM_THERMO_READINESS_TEST = "tests/test_quantum_thermo_readiness.py"
"""Public S9 protocol and payload tests."""
QUANTUM_THERMO_READINESS_BRANCH_TEST = "tests/test_quantum_thermo_readiness_branches.py"
"""Validation and serialisation branch tests."""
QUANTUM_THERMO_READINESS_EXPORTER = "scripts/export_s9_quantum_thermo_readiness.py"
"""Executable S9 JSON and Markdown readiness exporter."""
QUANTUM_THERMO_READINESS_EXPORT_TEST = "tests/test_export_s9_quantum_thermo_readiness.py"
"""Real filesystem exporter test."""
THERMO_READINESS_PRODUCT_TYPING_RATCHET = [
    THERMO_READINESS_PRODUCT_SOURCE,
    "tests/test_thermo_readiness_product.py",
    QUANTUM_THERMO_READINESS_SOURCE,
    QUANTUM_THERMO_READINESS_TEST,
    QUANTUM_THERMO_READINESS_BRANCH_TEST,
    QUANTUM_THERMO_READINESS_EXPORTER,
    QUANTUM_THERMO_READINESS_EXPORT_TEST,
    "tools/thermo_readiness_product_quality_gates.py",
    "tests/test_thermo_readiness_product_quality_gate.py",
]
"""Ordered strict-typing cohort."""

THERMO_READINESS_PRODUCT_DOCSTRING_RATCHET = [
    THERMO_READINESS_PRODUCT_SOURCE,
    QUANTUM_THERMO_READINESS_SOURCE,
    QUANTUM_THERMO_READINESS_TEST,
    QUANTUM_THERMO_READINESS_BRANCH_TEST,
    QUANTUM_THERMO_READINESS_EXPORTER,
    QUANTUM_THERMO_READINESS_EXPORT_TEST,
    "tools/thermo_readiness_product_quality_gates.py",
    "tests/test_thermo_readiness_product_quality_gate.py",
]
"""Public product, ambient readiness, exporter, and gate docstring cohort."""

THERMO_READINESS_PRODUCT_COVERAGE_COHORT = [
    "tests/test_thermo_readiness_product.py",
    QUANTUM_THERMO_READINESS_TEST,
    QUANTUM_THERMO_READINESS_BRANCH_TEST,
    QUANTUM_THERMO_READINESS_EXPORT_TEST,
]
"""Tests that own exact product and ambient-readiness branch coverage."""

THERMO_READINESS_PRODUCT_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-thermo-readiness-product-quality.coverage"  # nosec B108
)
"""Isolated coverage database for the thermo-readiness source owner."""
THERMO_READINESS_PRODUCT_COVERAGE_INCLUDE = (
    "*/thermo_readiness_product.py,*/thermodynamics/readiness.py"
)
"""Exact public product and ambient-readiness source include."""


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
            "mypy-strict-thermo-readiness-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *THERMO_READINESS_PRODUCT_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D thermo-readiness-product quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "D,D413,D417,D420",
                "--config",
                "lint.explicit-preview-rules = true",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *THERMO_READINESS_PRODUCT_DOCSTRING_RATCHET,
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
            "thermo-readiness-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={THERMO_READINESS_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *THERMO_READINESS_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "thermo-readiness-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={THERMO_READINESS_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={THERMO_READINESS_PRODUCT_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "THERMO_READINESS_PRODUCT_COVERAGE_COHORT",
    "THERMO_READINESS_PRODUCT_COVERAGE_DATA_FILE",
    "THERMO_READINESS_PRODUCT_COVERAGE_INCLUDE",
    "THERMO_READINESS_PRODUCT_DOCSTRING_RATCHET",
    "THERMO_READINESS_PRODUCT_SOURCE",
    "THERMO_READINESS_PRODUCT_TYPING_RATCHET",
    "QUANTUM_THERMO_READINESS_BRANCH_TEST",
    "QUANTUM_THERMO_READINESS_EXPORTER",
    "QUANTUM_THERMO_READINESS_EXPORT_TEST",
    "QUANTUM_THERMO_READINESS_SOURCE",
    "QUANTUM_THERMO_READINESS_TEST",
    "build_coverage_gates",
    "build_static_quality_gates",
]
