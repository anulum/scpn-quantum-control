# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — SSGF geometry-gradient quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

SSGF_GEOMETRY_GRADIENT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/ssgf_geometry_gradient_product.py",
    "src/scpn_quantum_control/ssgf/quantum_gradient.py",
    "src/scpn_quantum_control/ssgf/quantum_outer_cycle.py",
    "src/scpn_quantum_control/bridge/ssgf_w_adapter.py",
    "tests/test_ssgf_geometry_gradient_product.py",
    "tests/test_ssgf_quantum_gradient.py",
    "tests/test_quantum_outer_cycle.py",
    "tests/test_quantum_outer_cycle_branches.py",
    "tests/test_ssgf_w_adapter.py",
    "tools/ssgf_geometry_gradient_quality_gates.py",
    "tests/test_ssgf_geometry_gradient_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

SSGF_GEOMETRY_GRADIENT_COVERAGE_COHORT = [
    "tests/test_ssgf_geometry_gradient_product.py",
    "tests/test_ssgf_quantum_gradient.py",
    "tests/test_quantum_outer_cycle.py",
    "tests/test_quantum_outer_cycle_branches.py",
    "tests/test_ssgf_w_adapter.py",
]
"""Tests that own exact SSGF geometry-gradient coverage."""

SSGF_GEOMETRY_GRADIENT_COVERAGE_DATA_FILE = "/tmp/scpn-qc-ssgf-geometry-gradient-quality.coverage"  # nosec B108
"""Isolated coverage database for the SSGF geometry-gradient owner."""
SSGF_GEOMETRY_GRADIENT_COVERAGE_INCLUDE = (
    "*/ssgf_geometry_gradient_product.py,*/ssgf/quantum_gradient.py,"
    "*/ssgf/quantum_outer_cycle.py,*/bridge/ssgf_w_adapter.py"
)
"""Production sources enforced at exact branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-ssgf-geometry-gradient-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *SSGF_GEOMETRY_GRADIENT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D ssgf-geometry-gradient quality ratchet",
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
                *SSGF_GEOMETRY_GRADIENT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "ssgf-geometry-gradient focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={SSGF_GEOMETRY_GRADIENT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *SSGF_GEOMETRY_GRADIENT_COVERAGE_COHORT,
            ],
        ),
        (
            "ssgf-geometry-gradient exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={SSGF_GEOMETRY_GRADIENT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={SSGF_GEOMETRY_GRADIENT_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "SSGF_GEOMETRY_GRADIENT_COVERAGE_COHORT",
    "SSGF_GEOMETRY_GRADIENT_COVERAGE_DATA_FILE",
    "SSGF_GEOMETRY_GRADIENT_COVERAGE_INCLUDE",
    "SSGF_GEOMETRY_GRADIENT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
