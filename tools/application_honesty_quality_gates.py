# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Application honesty quality gates
"""Build static, evidence-drift, and coverage gates for application honesty."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
APPLICATION_HONESTY_QUALITY_RATCHET = [
    "src/scpn_quantum_control/applications/honesty_kits.py",
    "src/scpn_quantum_control/applications/dataset_catalog.py",
    "tests/test_application_honesty_kits.py",
    "tests/test_dataset_catalog.py",
    "scripts/run_application_honesty_audit.py",
    "tools/application_honesty_quality_gates.py",
    "tests/test_application_honesty_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
APPLICATION_HONESTY_COVERAGE_DATA_FILE = ".coverage.application-honesty-quality"
"""Isolated coverage database for the application honesty module."""
APPLICATION_HONESTY_COVERAGE_COHORT = [
    "tests/test_application_honesty_kits.py",
    "tests/test_dataset_catalog.py",
]
"""Real honesty-policy and packaged-catalogue execution cohort."""
APPLICATION_HONESTY_COVERAGE_INCLUDE = (
    "*/applications/honesty_kits.py,*/applications/dataset_catalog.py"
)
"""Exact source owners enforced by the shared coverage report."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build typing, documentation, and evidence-drift gates."""
    return [
        (
            "mypy-strict-application-honesty-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *APPLICATION_HONESTY_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D application-honesty quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "D,D107,D413,D417,D420",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *APPLICATION_HONESTY_QUALITY_RATCHET,
            ],
        ),
        (
            "application-honesty evidence drift",
            [python, "scripts/run_application_honesty_audit.py", "--check"],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    data = APPLICATION_HONESTY_COVERAGE_DATA_FILE
    return [
        (
            "application-honesty focused coverage",
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
                *APPLICATION_HONESTY_COVERAGE_COHORT,
            ],
        ),
        (
            "application-honesty exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                f"--include={APPLICATION_HONESTY_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "APPLICATION_HONESTY_COVERAGE_COHORT",
    "APPLICATION_HONESTY_COVERAGE_DATA_FILE",
    "APPLICATION_HONESTY_COVERAGE_INCLUDE",
    "APPLICATION_HONESTY_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
