# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — inverse-problem quality gates
"""Build strict typing, documentation and exact coverage gates for the network-inference owner."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
INVERSE_PROBLEM_SOURCES = [
    "oscillatools/src/oscillatools/accel/kuramoto_bayesian_inference.py",
    "oscillatools/src/oscillatools/accel/kuramoto_network_control.py",
]
INVERSE_PROBLEM_CONFORMANCE_TEST = "tests/test_inverse_problem_identifiability.py"
INVERSE_PROBLEM_COVERAGE_COHORT = [
    INVERSE_PROBLEM_CONFORMANCE_TEST,
    "oscillatools/tests/test_kuramoto_bayesian_inference.py",
    "tests/test_synchronisation_inference_control.py",
    "oscillatools/tests/test_kuramoto_network_control.py",
]
INVERSE_PROBLEM_COVERAGE_INCLUDE = (
    "*/accel/kuramoto_bayesian_inference.py,*/accel/kuramoto_network_control.py"
)
INVERSE_PROBLEM_TYPING_RATCHET = [
    *INVERSE_PROBLEM_SOURCES,
    *INVERSE_PROBLEM_COVERAGE_COHORT,
    "tools/inverse_problem_quality_gates.py",
    "tests/test_inverse_problem_quality_gate.py",
]
INVERSE_PROBLEM_DOCSTRING_RATCHET = [*INVERSE_PROBLEM_TYPING_RATCHET]
INVERSE_PROBLEM_COVERAGE_DATA_FILE = "/tmp/scpn-qc-inverse-problem-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates.

    Parameters
    ----------
    python
        Interpreter that runs the gates.

    Returns
    -------
    list[Gate]
        Named commands.

    """
    return [
        (
            "mypy-strict-inverse-problem-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *INVERSE_PROBLEM_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D inverse-problem quality ratchet",
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
                *INVERSE_PROBLEM_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build the focused run and the exact coverage threshold.

    Parameters
    ----------
    python
        Interpreter that runs the gates.

    Returns
    -------
    list[Gate]
        Named commands.

    """
    return [
        (
            "inverse-problem focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={INVERSE_PROBLEM_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *INVERSE_PROBLEM_COVERAGE_COHORT,
            ],
        ),
        (
            "inverse-problem exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={INVERSE_PROBLEM_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={INVERSE_PROBLEM_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "INVERSE_PROBLEM_CONFORMANCE_TEST",
    "INVERSE_PROBLEM_COVERAGE_COHORT",
    "INVERSE_PROBLEM_COVERAGE_DATA_FILE",
    "INVERSE_PROBLEM_COVERAGE_INCLUDE",
    "INVERSE_PROBLEM_DOCSTRING_RATCHET",
    "INVERSE_PROBLEM_SOURCES",
    "INVERSE_PROBLEM_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
