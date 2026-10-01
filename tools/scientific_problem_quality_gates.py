# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Scientific problem quality gates
"""Define the owning scientific-input typing, documentation and coverage cohort."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

SCIENTIFIC_PROBLEM_SOURCES = [
    "src/scpn_quantum_control/kuramoto_core.py",
    "src/scpn_quantum_control/scientific_design.py",
    "src/scpn_quantum_control/scientific_problem_parameters.py",
]
"""Original problem facade, supplied design and semantic input adapter owners."""
SCIENTIFIC_PROBLEM_TESTS = [
    "tests/test_scientific_design.py",
    "tests/test_scientific_problem_parameters.py",
    "tests/test_kuramoto_core.py",
    "tests/test_kuramoto_core_branches.py",
    "tests/test_kuramoto_variants.py",
    "tests/test_scientific_problem_quality_gate.py",
]
"""Real public design, schema/force/compiler and original facade consumers."""
SCIENTIFIC_PROBLEM_TYPING = [
    *SCIENTIFIC_PROBLEM_SOURCES,
    "tests/test_scientific_design.py",
    "tests/test_scientific_problem_parameters.py",
    "tools/scientific_problem_quality_gates.py",
    "tests/test_scientific_problem_quality_gate.py",
]
"""Strict typing and native documentation cohort, without relaxing old tests."""
SCIENTIFIC_PROBLEM_COVERAGE = "/tmp/scpn-qc-scientific-problem.coverage"  # nosec B108
"""Separate scientific-input coverage database."""
SCIENTIFIC_PROBLEM_INCLUDE = "*/kuramoto_core.py,*/scientific_design.py,*/scientific_problem_parameters.py,*/tools/scientific_problem_quality_gates.py"
"""Every changed owning module, including the reachable quality command builder."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and native NumPy documentation commands.

    Parameters
    ----------
    python
        Existing runner executable; this function does not install dependencies.

    Returns
    -------
    list[Gate]
        Ordered named commands scoped to the scientific-input owners.

    """
    return [
        (
            "mypy-strict-scientific-problem",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *SCIENTIFIC_PROBLEM_TYPING,
            ],
        ),
        (
            "ruff D scientific problem",
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
                *SCIENTIFIC_PROBLEM_TYPING,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real owning execution and exact statement/branch coverage checks.

    Parameters
    ----------
    python
        Existing runner executable used by both execution and report.

    Returns
    -------
    list[Gate]
        Commands requiring the full owned denominator at 100 percent.

    """
    return [
        (
            "scientific problem focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={SCIENTIFIC_PROBLEM_COVERAGE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                "--no-cov",
                *SCIENTIFIC_PROBLEM_TESTS,
            ],
        ),
        (
            "scientific problem exact coverage",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={SCIENTIFIC_PROBLEM_COVERAGE}",
                "--precision=2",
                "--show-missing",
                "--fail-under=100",
                f"--include={SCIENTIFIC_PROBLEM_INCLUDE}",
            ],
        ),
    ]
