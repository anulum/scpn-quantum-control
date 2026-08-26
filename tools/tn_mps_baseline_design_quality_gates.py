# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tensor-network MPS baseline quality-gate specification
"""Build strict typing, documentation, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

TN_MPS_BASELINE_DESIGN_QUALITY_RATCHET = [
    "src/scpn_quantum_control/benchmarks/tn_mps_baseline_design.py",
    "tests/test_tn_mps_baseline_design.py",
    "scripts/export_tn_mps_baseline_design.py",
    "tools/tn_mps_baseline_design_quality_gates.py",
    "tests/test_tn_mps_baseline_design_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

TN_MPS_BASELINE_DESIGN_COVERAGE_COHORT = ["tests/test_tn_mps_baseline_design.py"]
"""Tests that own exact tensor-network MPS baseline coverage."""

TN_MPS_BASELINE_DESIGN_COVERAGE_DATA_FILE = ".coverage.tn-mps-baseline-design-quality"
"""Isolated coverage database for the tensor-network MPS baseline owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-tn-mps-baseline-design-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *TN_MPS_BASELINE_DESIGN_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D tn-mps-baseline-design quality ratchet",
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
                *TN_MPS_BASELINE_DESIGN_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "tn-mps-baseline-design focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={TN_MPS_BASELINE_DESIGN_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *TN_MPS_BASELINE_DESIGN_COVERAGE_COHORT,
            ],
        ),
        (
            "tn-mps-baseline-design exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={TN_MPS_BASELINE_DESIGN_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/tn_mps_baseline_design.py",
            ],
        ),
    ]


__all__ = [
    "TN_MPS_BASELINE_DESIGN_COVERAGE_COHORT",
    "TN_MPS_BASELINE_DESIGN_COVERAGE_DATA_FILE",
    "TN_MPS_BASELINE_DESIGN_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
