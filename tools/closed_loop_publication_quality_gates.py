# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Closed-loop publication quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
CLOSED_LOOP_PUBLICATION_SOURCE = (
    "src/scpn_quantum_control/benchmarks/closed_loop_publication_run.py"
)
CLOSED_LOOP_ANALYSIS_SOURCE = "src/scpn_quantum_control/control/closed_loop_analysis.py"
CLOSED_LOOP_PUBLICATION_SOURCES = [
    CLOSED_LOOP_PUBLICATION_SOURCE,
    CLOSED_LOOP_ANALYSIS_SOURCE,
]
CLOSED_LOOP_PUBLICATION_COVERAGE_COHORT = [
    "tests/test_closed_loop_publication_run.py",
    "tests/test_run_closed_loop_publication.py",
    "tests/test_closed_loop_analysis.py",
    "tests/test_closed_loop_analysis_wall_clock.py",
]
CLOSED_LOOP_PUBLICATION_TYPING_RATCHET = [
    *CLOSED_LOOP_PUBLICATION_SOURCES,
    *CLOSED_LOOP_PUBLICATION_COVERAGE_COHORT,
    "tools/closed_loop_publication_quality_gates.py",
    "tests/test_closed_loop_publication_quality_gate.py",
]
CLOSED_LOOP_PUBLICATION_DOCSTRING_RATCHET = [
    *CLOSED_LOOP_PUBLICATION_TYPING_RATCHET,
]
CLOSED_LOOP_PUBLICATION_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-closed-loop-publication-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-closed-loop-publication-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *CLOSED_LOOP_PUBLICATION_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D closed-loop publication quality ratchet",
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
                *CLOSED_LOOP_PUBLICATION_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline execution and exact source-coverage gates."""
    return [
        (
            "closed-loop publication focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={CLOSED_LOOP_PUBLICATION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *CLOSED_LOOP_PUBLICATION_COVERAGE_COHORT,
            ],
        ),
        (
            "closed-loop publication exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={CLOSED_LOOP_PUBLICATION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/benchmarks/closed_loop_publication_run.py,*/control/closed_loop_analysis.py",
            ],
        ),
    ]


__all__ = [
    "CLOSED_LOOP_ANALYSIS_SOURCE",
    "CLOSED_LOOP_PUBLICATION_COVERAGE_COHORT",
    "CLOSED_LOOP_PUBLICATION_COVERAGE_DATA_FILE",
    "CLOSED_LOOP_PUBLICATION_DOCSTRING_RATCHET",
    "CLOSED_LOOP_PUBLICATION_SOURCE",
    "CLOSED_LOOP_PUBLICATION_SOURCES",
    "CLOSED_LOOP_PUBLICATION_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
