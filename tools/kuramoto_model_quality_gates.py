# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto model convention quality gates
"""Require actual finite-model, original-source and owning CI contract checks."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

KURAMOTO_MODEL_SOURCES = [
    "src/scpn_quantum_control/kuramoto_core.py",
    "src/scpn_quantum_control/kuramoto_model_conventions.py",
    "tools/build_kuramoto_conventions.py",
    "tools/kuramoto_model_quality_gates.py",
]
"""Original facade, pure model declarations and source/quality producers."""
KURAMOTO_MODEL_TESTS = [
    "tests/test_kuramoto_model_conventions.py",
    "tests/test_build_kuramoto_conventions.py",
    "tests/test_kuramoto_model_quality_gate.py",
    "tests/test_kuramoto_core.py",
    "tests/test_kuramoto_core_branches.py",
    "tests/test_kuramoto_variants.py",
    "tests/test_scientific_design.py",
    "tests/test_scientific_problem_parameters.py",
]
"""Dedicated owners and existing direct original-facade binding consumers."""
KURAMOTO_MODEL_TYPING = [
    *KURAMOTO_MODEL_SOURCES,
    "src/scpn_quantum_control/__init__.py",
    "tests/test_kuramoto_model_conventions.py",
    "tests/test_build_kuramoto_conventions.py",
    "tests/test_kuramoto_model_quality_gate.py",
]
"""Every new or changed public Python owner and its dedicated behaviour tests."""
KURAMOTO_MODEL_COVERAGE = ".coverage.kuramoto-model-conventions"
"""Distinct project-local CI database; local checks use their owned allocation."""
KURAMOTO_MODEL_INCLUDE = "*/kuramoto_core.py,*/kuramoto_model_conventions.py,*/tools/build_kuramoto_conventions.py,*/tools/kuramoto_model_quality_gates.py"
"""Complete statement and branch denominator for the changed original owners."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing, native documentation and generated-source checks.

    Parameters
    ----------
    python
        Existing runner executable; no dependency is installed by these commands.

    Returns
    -------
    list[Gate]
        Required commands over exact scientific convention and producer owners.

    """
    return [
        (
            "mypy-strict-kuramoto-model",
            [python, "-m", "mypy", "--strict", "--explicit-package-bases", *KURAMOTO_MODEL_TYPING],
        ),
        (
            "ruff D kuramoto model",
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
                *KURAMOTO_MODEL_TYPING,
            ],
        ),
        (
            "kuramoto convention source drift",
            [python, "tools/build_kuramoto_conventions.py", "--check"],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build the owning real execution and exact statement/branch checks.

    Parameters
    ----------
    python
        Runner shared by actual production tests and the coverage report.

    Returns
    -------
    list[Gate]
        Coherent facade and convention source checks requiring 100 percent.

    """
    return [
        (
            "kuramoto model focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={KURAMOTO_MODEL_COVERAGE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                "--no-cov",
                *KURAMOTO_MODEL_TESTS,
            ],
        ),
        (
            "kuramoto model exact owner coverage",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={KURAMOTO_MODEL_COVERAGE}",
                "--precision=2",
                "--show-missing",
                "--fail-under=100",
                f"--include={KURAMOTO_MODEL_INCLUDE}",
            ],
        ),
    ]
