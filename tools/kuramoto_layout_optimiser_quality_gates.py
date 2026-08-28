# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto layout-optimiser quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
KURAMOTO_LAYOUT_OPTIMISER_SOURCE = "src/scpn_quantum_control/hardware/kuramoto_layout_optimiser.py"
KURAMOTO_LAYOUT_OPTIMISER_COVERAGE_COHORT = [
    "tests/test_kuramoto_layout_optimiser.py",
]
KURAMOTO_LAYOUT_OPTIMISER_TYPING_RATCHET = [
    KURAMOTO_LAYOUT_OPTIMISER_SOURCE,
    *KURAMOTO_LAYOUT_OPTIMISER_COVERAGE_COHORT,
    "tools/kuramoto_layout_optimiser_quality_gates.py",
    "tests/test_kuramoto_layout_optimiser_quality_gate.py",
]
KURAMOTO_LAYOUT_OPTIMISER_DOCSTRING_RATCHET = [
    *KURAMOTO_LAYOUT_OPTIMISER_TYPING_RATCHET,
]
KURAMOTO_LAYOUT_OPTIMISER_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-kuramoto-layout-optimiser-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-kuramoto-layout-optimiser-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *KURAMOTO_LAYOUT_OPTIMISER_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D Kuramoto layout-optimiser quality ratchet",
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
                *KURAMOTO_LAYOUT_OPTIMISER_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline optimizer execution and exact source-coverage gates."""
    return [
        (
            "Kuramoto layout-optimiser focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={KURAMOTO_LAYOUT_OPTIMISER_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *KURAMOTO_LAYOUT_OPTIMISER_COVERAGE_COHORT,
            ],
        ),
        (
            "Kuramoto layout-optimiser exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={KURAMOTO_LAYOUT_OPTIMISER_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hardware/kuramoto_layout_optimiser.py",
            ],
        ),
    ]


__all__ = [
    "KURAMOTO_LAYOUT_OPTIMISER_COVERAGE_COHORT",
    "KURAMOTO_LAYOUT_OPTIMISER_COVERAGE_DATA_FILE",
    "KURAMOTO_LAYOUT_OPTIMISER_DOCSTRING_RATCHET",
    "KURAMOTO_LAYOUT_OPTIMISER_SOURCE",
    "KURAMOTO_LAYOUT_OPTIMISER_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
