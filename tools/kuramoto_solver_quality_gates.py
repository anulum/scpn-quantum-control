# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — existing solver contract quality commands
"""Own strict documentation/types and actual whole solver boundary coverage."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

KURAMOTO_SOLVER_SOURCES = [
    "oscillatools/src/oscillatools/accel/kuramoto_system.py",
    "oscillatools/src/oscillatools/accel/diff_kuramoto_adaptive.py",
    "oscillatools/src/oscillatools/accel/diff_kuramoto_dopri.py",
    "oscillatools/src/oscillatools/accel/kuramoto_adaptive.py",
    "oscillatools/src/oscillatools/accel/kuramoto_delayed.py",
    "oscillatools/src/oscillatools/accel/kuramoto_scipy_interop.py",
    "tools/kuramoto_solver_quality_gates.py",
]
"""Existing numerical owners and the coherent scoped command builder."""
KURAMOTO_SOLVER_TESTS = [
    "tests/test_kuramoto_solver_trajectories.py",
    "tests/test_kuramoto_solver_quality_gate.py",
    "oscillatools/tests/test_kuramoto_system.py",
    "oscillatools/tests/test_diff_kuramoto_adaptive.py",
    "oscillatools/tests/test_diff_kuramoto_dopri.py",
    "oscillatools/tests/test_kuramoto_adaptive.py",
    "oscillatools/tests/test_kuramoto_delayed.py",
    "oscillatools/tests/test_kuramoto_scipy_interop.py",
    "tests/test_kuramoto_model_conventions.py",
]
"""Independent equations, real adapters and their original direct tests."""
KURAMOTO_SOLVER_TYPING = [
    *KURAMOTO_SOLVER_SOURCES,
    "tests/test_kuramoto_solver_trajectories.py",
    "tests/test_kuramoto_solver_quality_gate.py",
    "examples/kuramoto_solver_outcomes.py",
]
"""Every changed Python production owner, new test and runnable example."""
KURAMOTO_SOLVER_COVERAGE = ".coverage.kuramoto-solvers"
"""Distinct CI database; local execution uses registered task allocations."""
KURAMOTO_SOLVER_INCLUDE = "*/kuramoto_system.py,*/diff_kuramoto_adaptive.py,*/diff_kuramoto_dopri.py,*/kuramoto_adaptive.py,*/kuramoto_delayed.py,*/kuramoto_scipy_interop.py,*/tools/kuramoto_solver_quality_gates.py"
"""Complete mandatory statements/branches; unchanged optional adapters retain native policy."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict types and native NumPy documentation commands.

    Parameters
    ----------
    python
        Existing runner executable; no installation or provider access occurs.

    Returns
    -------
    list[Gate]
        Exact original solver and new behaviour owners, without exclusions.

    """
    return [
        (
            "mypy-strict-kuramoto-solvers",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *KURAMOTO_SOLVER_TYPING,
            ],
        ),
        (
            "ruff D kuramoto solvers",
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
                *KURAMOTO_SOLVER_TYPING,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build actual public execution and full statement/branch enforcement.

    Parameters
    ----------
    python
        Runner shared by production tests and the exact denominator report.

    Returns
    -------
    list[Gate]
        Scoped scientific checks requiring 100 percent, preserving legacy tests.

    """
    return [
        (
            "kuramoto solver focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={KURAMOTO_SOLVER_COVERAGE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                "--no-cov",
                *KURAMOTO_SOLVER_TESTS,
            ],
        ),
        (
            "kuramoto solver exact owner coverage",
            [
                python,
                "-m",
                "coverage",
                "report",
                "--rcfile=oscillatools/pyproject.toml",
                f"--data-file={KURAMOTO_SOLVER_COVERAGE}",
                "--precision=2",
                "--show-missing",
                "--fail-under=100",
                f"--include={KURAMOTO_SOLVER_INCLUDE}",
            ],
        ),
    ]
