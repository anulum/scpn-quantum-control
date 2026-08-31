# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto core quality-gate specification
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
KURAMOTO_CORE_SOURCE = "src/scpn_quantum_control/kuramoto_core.py"
"""Stable public Kuramoto facade guarded by this owner."""
KURAMOTO_CORE_TYPING_RATCHET = [
    KURAMOTO_CORE_SOURCE,
    "tests/test_kuramoto_core.py",
    "tests/test_kuramoto_core_branches.py",
    "tools/kuramoto_core_quality_gates.py",
    "tests/test_kuramoto_core_quality_gate.py",
    "tools/preflight.py",
    "tests/test_preflight_tool.py",
]
"""Touched Python surfaces held to strict MyPy."""
KURAMOTO_CORE_DOCSTRING_RATCHET = [
    KURAMOTO_CORE_SOURCE,
    "tests/test_kuramoto_core.py",
    "tests/test_kuramoto_core_branches.py",
    "tools/kuramoto_core_quality_gates.py",
    "tests/test_kuramoto_core_quality_gate.py",
    "tests/test_preflight_tool.py",
]
"""Direct owner surfaces held to complete NumPy docstrings."""
KURAMOTO_CORE_COVERAGE_COHORT = [
    "tests/test_kuramoto_core.py",
    "tests/test_kuramoto_core_branches.py",
    "tests/test_kuramoto_input_hardening.py",
    "tests/test_kuramoto_variants.py",
    "tests/test_analog_kuramoto.py",
    "tests/test_hybrid_digital_analog.py",
]
"""Connected public-facade tests that exercise every production branch."""
KURAMOTO_CORE_COVERAGE_DATA_FILE = "/tmp/scpn-qc-kuramoto-core-quality.coverage"  # nosec B108
"""Isolated coverage database for the Kuramoto core owner."""
KURAMOTO_CORE_COVERAGE_INCLUDE = "*/kuramoto_core.py"
"""Exact production source include for the coverage report."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-kuramoto-core-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *KURAMOTO_CORE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D Kuramoto-core quality ratchet",
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
                *KURAMOTO_CORE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build connected execution and exact source-only coverage gates."""
    return [
        (
            "Kuramoto-core focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={KURAMOTO_CORE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *KURAMOTO_CORE_COVERAGE_COHORT,
            ],
        ),
        (
            "Kuramoto-core exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={KURAMOTO_CORE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={KURAMOTO_CORE_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "KURAMOTO_CORE_COVERAGE_COHORT",
    "KURAMOTO_CORE_COVERAGE_DATA_FILE",
    "KURAMOTO_CORE_COVERAGE_INCLUDE",
    "KURAMOTO_CORE_DOCSTRING_RATCHET",
    "KURAMOTO_CORE_SOURCE",
    "KURAMOTO_CORE_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
