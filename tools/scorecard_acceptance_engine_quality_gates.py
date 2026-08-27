# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — scorecard acceptance engine quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
SCORECARD_ACCEPTANCE_ENGINE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/scorecard_acceptance_engine.py",
    "tests/test_scorecard_acceptance_engine.py",
    "tools/scorecard_acceptance_engine_quality_gates.py",
    "tests/test_scorecard_acceptance_engine_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
SCORECARD_ACCEPTANCE_ENGINE_COVERAGE_COHORT = ["tests/test_scorecard_acceptance_engine.py"]
"""Tests that own exact scorecard-acceptance-engine coverage."""
SCORECARD_ACCEPTANCE_ENGINE_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-scorecard-acceptance-engine.coverage"  # nosec B108
)
"""Isolated coverage database for scorecard-acceptance-engine diagnostics."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-scorecard-acceptance-engine-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *SCORECARD_ACCEPTANCE_ENGINE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D scorecard-acceptance-engine quality ratchet",
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
                *SCORECARD_ACCEPTANCE_ENGINE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    data = SCORECARD_ACCEPTANCE_ENGINE_COVERAGE_DATA_FILE
    return [
        (
            "scorecard-acceptance-engine focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *SCORECARD_ACCEPTANCE_ENGINE_COVERAGE_COHORT,
            ],
        ),
        (
            "scorecard-acceptance-engine exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/scorecard_acceptance_engine.py",
            ],
        ),
    ]


__all__ = [
    "SCORECARD_ACCEPTANCE_ENGINE_COVERAGE_COHORT",
    "SCORECARD_ACCEPTANCE_ENGINE_COVERAGE_DATA_FILE",
    "SCORECARD_ACCEPTANCE_ENGINE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
