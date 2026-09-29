# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — qualification projection quality-gate specification
"""Build strict documentation, typing, and exact coverage gates.

The owner is the source-bound qualification projection, the executable CI
ownership resolver it relies on, the release profile that consumes it and the
repository CI inventory reader that shares the resolver.
"""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
QUALIFICATION_PROJECTION_QUALITY_RATCHET = [
    "src/scpn_quantum_control/qualification_status_projection.py",
    "src/scpn_quantum_control/ci_workflow_ownership.py",
    "src/scpn_quantum_control/differentiable_baseline_scorecard.py",
    "tools/ci_workflow_inventory.py",
    "tests/_qualification_receipt_vectors.py",
    "tests/test_qualification_status_projection.py",
    "tests/test_ci_workflow_ownership.py",
    "tests/test_ci_workflow_inventory.py",
    "tests/test_differentiable_baseline_scorecard.py",
    "tools/qualification_projection_quality_gates.py",
    "tests/test_qualification_projection_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
QUALIFICATION_PROJECTION_COVERAGE_COHORT = [
    "tests/test_qualification_status_projection.py",
    "tests/test_ci_workflow_ownership.py",
    "tests/test_ci_workflow_inventory.py",
    "tests/test_differentiable_baseline_scorecard.py",
]
"""Tests that own exact qualification projection coverage."""
QUALIFICATION_PROJECTION_COVERAGE_INCLUDE = (
    "*/qualification_status_projection.py,*/ci_workflow_ownership.py,"
    "*/tools/ci_workflow_inventory.py,*/differentiable_baseline_scorecard.py"
)
"""Source modules measured at exact branch coverage."""
QUALIFICATION_PROJECTION_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-qualification-projection-quality.coverage"  # nosec B108
)
"""Isolated coverage database for the qualification projection owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Interpreter used to run mypy and ruff.

    Returns
    -------
    list[Gate]
        Named commands over the exact ratchet cohort, in execution order.

    """
    return [
        (
            "mypy-strict-qualification-projection-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *QUALIFICATION_PROJECTION_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D qualification-projection quality ratchet",
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
                'lint.pydocstyle.convention = "numpy"',
                *QUALIFICATION_PROJECTION_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates.

    Parameters
    ----------
    python
        Interpreter used to run coverage and pytest.

    Returns
    -------
    list[Gate]
        Branch-measured cohort execution followed by the exact 100% report.

    """
    return [
        (
            "qualification-projection focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={QUALIFICATION_PROJECTION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *QUALIFICATION_PROJECTION_COVERAGE_COHORT,
            ],
        ),
        (
            "qualification-projection exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={QUALIFICATION_PROJECTION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={QUALIFICATION_PROJECTION_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "QUALIFICATION_PROJECTION_COVERAGE_COHORT",
    "QUALIFICATION_PROJECTION_COVERAGE_DATA_FILE",
    "QUALIFICATION_PROJECTION_COVERAGE_INCLUDE",
    "QUALIFICATION_PROJECTION_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
