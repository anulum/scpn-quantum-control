# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — typed phase-result quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PHASE_RESULTS_SOURCE = "src/scpn_quantum_control/phase/results.py"
PHASE_RESULTS_COVERAGE_COHORT = [
    "tests/test_phase_results.py",
    "tests/test_xy_kuramoto.py",
]
PHASE_RESULTS_TYPING_RATCHET = [
    PHASE_RESULTS_SOURCE,
    "tools/phase_results_quality_gates.py",
    "tests/test_phase_results_quality_gate.py",
]
PHASE_RESULTS_DOCSTRING_RATCHET = [
    PHASE_RESULTS_SOURCE,
    "tests/test_phase_results.py",
    "tools/phase_results_quality_gates.py",
    "tests/test_phase_results_quality_gate.py",
]
PHASE_RESULTS_COVERAGE_DATA_FILE = "/tmp/scpn-qc-phase-results-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-phase-results-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PHASE_RESULTS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D typed phase-result quality ratchet",
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
                *PHASE_RESULTS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build connected typed-result execution and exact coverage gates."""
    return [
        (
            "typed phase-result focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_RESULTS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PHASE_RESULTS_COVERAGE_COHORT,
            ],
        ),
        (
            "typed phase-result exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_RESULTS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/results.py",
            ],
        ),
    ]


__all__ = [
    "PHASE_RESULTS_COVERAGE_COHORT",
    "PHASE_RESULTS_COVERAGE_DATA_FILE",
    "PHASE_RESULTS_DOCSTRING_RATCHET",
    "PHASE_RESULTS_SOURCE",
    "PHASE_RESULTS_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
