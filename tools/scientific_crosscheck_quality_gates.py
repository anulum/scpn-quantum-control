# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — scientific crosscheck quality-gate specification
"""Build strict documentation and exact coverage gates for crosschecks."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

SCIENTIFIC_CROSSCHECK_QUALITY_RATCHET = [
    "src/scpn_quantum_control/analysis/qfi_geometric_crosscheck.py",
    "src/scpn_quantum_control/gauge/lattice_crosscheck.py",
    "tests/test_qfi_geometric_crosscheck.py",
    "tests/test_gauge_lattice_crosscheck.py",
    "tools/scientific_crosscheck_quality_gates.py",
    "tests/test_scientific_crosscheck_quality_gate.py",
]
"""Complete strict-typing and preview-documentation owner."""

SCIENTIFIC_CROSSCHECK_COVERAGE_COHORT = [
    "tests/test_qfi_geometric_crosscheck.py",
    "tests/test_gauge_lattice_crosscheck.py",
]
"""Real scientific crosscheck suites used for exact coverage."""

SCIENTIFIC_CROSSCHECK_COVERAGE_DATA_FILE = ".coverage.scientific-crosscheck-quality"
"""Isolated coverage database for the scientific crosscheck owner."""

SCIENTIFIC_CROSSCHECK_COVERAGE_INCLUDE = (
    "*/analysis/qfi_geometric_crosscheck.py,*/gauge/lattice_crosscheck.py"
)
"""Production crosscheck sources enforced at exact branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and preview NumPy-documentation gates."""
    return [
        (
            "mypy-strict-scientific-crosscheck-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *SCIENTIFIC_CROSSCHECK_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D scientific-crosscheck quality ratchet",
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
                *SCIENTIFIC_CROSSCHECK_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real execution and exact source-coverage gates."""
    return [
        (
            "scientific-crosscheck focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={SCIENTIFIC_CROSSCHECK_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *SCIENTIFIC_CROSSCHECK_COVERAGE_COHORT,
            ],
        ),
        (
            "scientific-crosscheck exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={SCIENTIFIC_CROSSCHECK_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={SCIENTIFIC_CROSSCHECK_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "SCIENTIFIC_CROSSCHECK_COVERAGE_COHORT",
    "SCIENTIFIC_CROSSCHECK_COVERAGE_DATA_FILE",
    "SCIENTIFIC_CROSSCHECK_COVERAGE_INCLUDE",
    "SCIENTIFIC_CROSSCHECK_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
