# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — feedback-loop quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

FEEDBACK_LOOP_SOURCE = "src/scpn_quantum_control/hardware/feedback_loop.py"
"""Production source owned by the cross-shot feedback loop."""
ORCHESTRATOR_FEEDBACK_SOURCE = "src/scpn_quantum_control/bridge/orchestrator_feedback.py"
"""Production source owned by the phase-orchestrator feedback path."""

FEEDBACK_LOOP_COVERAGE_COHORT = [
    "tests/test_feedback_loop.py",
    "tests/test_feedback_loop_branches.py",
    "tests/test_orchestrator_feedback.py",
]
"""Offline tests that own exact feedback-loop coverage."""

FEEDBACK_LOOP_TYPING_RATCHET = [
    FEEDBACK_LOOP_SOURCE,
    ORCHESTRATOR_FEEDBACK_SOURCE,
    "tests/test_orchestrator_feedback.py",
    "tools/feedback_loop_quality_gates.py",
    "tests/test_feedback_loop_quality_gate.py",
]
"""Strict-typing cohort for production and gate contracts."""

FEEDBACK_LOOP_DOCSTRING_RATCHET = [
    FEEDBACK_LOOP_SOURCE,
    ORCHESTRATOR_FEEDBACK_SOURCE,
    *FEEDBACK_LOOP_COVERAGE_COHORT,
    "tools/feedback_loop_quality_gates.py",
    "tests/test_feedback_loop_quality_gate.py",
]
"""Complete production, owner-test, and gate-contract docstring cohort."""

FEEDBACK_LOOP_COVERAGE_DATA_FILE = "/tmp/scpn-qc-feedback-loop-quality.coverage"  # nosec B108
"""Isolated coverage database for the feedback-loop owner."""
FEEDBACK_LOOP_COVERAGE_INCLUDE = "*/hardware/feedback_loop.py,*/bridge/orchestrator_feedback.py"
"""Exact production sources enforced by the shared coverage report."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-feedback-loop-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *FEEDBACK_LOOP_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D feedback-loop quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--select",
                "D,D413",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *FEEDBACK_LOOP_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    return [
        (
            "feedback-loop focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={FEEDBACK_LOOP_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *FEEDBACK_LOOP_COVERAGE_COHORT,
            ],
        ),
        (
            "feedback-loop exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={FEEDBACK_LOOP_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={FEEDBACK_LOOP_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "FEEDBACK_LOOP_COVERAGE_COHORT",
    "FEEDBACK_LOOP_COVERAGE_DATA_FILE",
    "FEEDBACK_LOOP_COVERAGE_INCLUDE",
    "FEEDBACK_LOOP_DOCSTRING_RATCHET",
    "FEEDBACK_LOOP_SOURCE",
    "FEEDBACK_LOOP_TYPING_RATCHET",
    "ORCHESTRATOR_FEEDBACK_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
