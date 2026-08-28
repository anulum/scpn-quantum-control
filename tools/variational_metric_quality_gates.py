# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — variational-metric quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

VARIATIONAL_METRIC_SOURCE = "src/scpn_quantum_control/phase/variational_metric.py"
"""Production source owned by the variational metric."""

VARIATIONAL_METRIC_COVERAGE_COHORT = ["tests/test_variational_metric.py"]
"""Tests that own exact variational-metric coverage."""

VARIATIONAL_METRIC_TYPING_RATCHET = [
    VARIATIONAL_METRIC_SOURCE,
    "tools/variational_metric_quality_gates.py",
    "tests/test_variational_metric_quality_gate.py",
]
"""Strict-typing cohort for production and gate contracts."""

VARIATIONAL_METRIC_DOCSTRING_RATCHET = [
    VARIATIONAL_METRIC_SOURCE,
    *VARIATIONAL_METRIC_COVERAGE_COHORT,
    "tools/variational_metric_quality_gates.py",
    "tests/test_variational_metric_quality_gate.py",
]
"""Complete production, owner-test, and gate-contract docstring cohort."""

VARIATIONAL_METRIC_COVERAGE_DATA_FILE = "/tmp/scpn-qc-variational-metric-quality.coverage"  # nosec B108
"""Isolated coverage database for the variational-metric owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-variational-metric-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *VARIATIONAL_METRIC_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D variational-metric quality ratchet",
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
                *VARIATIONAL_METRIC_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    return [
        (
            "variational-metric focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={VARIATIONAL_METRIC_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *VARIATIONAL_METRIC_COVERAGE_COHORT,
            ],
        ),
        (
            "variational-metric exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={VARIATIONAL_METRIC_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/variational_metric.py",
            ],
        ),
    ]


__all__ = [
    "VARIATIONAL_METRIC_COVERAGE_COHORT",
    "VARIATIONAL_METRIC_COVERAGE_DATA_FILE",
    "VARIATIONAL_METRIC_DOCSTRING_RATCHET",
    "VARIATIONAL_METRIC_SOURCE",
    "VARIATIONAL_METRIC_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
