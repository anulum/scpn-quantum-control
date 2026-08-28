# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio coupling-invariant quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
COUPLING_INVARIANT_SOURCE = "src/scpn_quantum_control/studio/coupling_invariant.py"
COUPLING_INVARIANT_COVERAGE_COHORT = [
    "tests/test_studio_coupling_invariant.py",
]
COUPLING_INVARIANT_TYPING_RATCHET = [
    COUPLING_INVARIANT_SOURCE,
    *COUPLING_INVARIANT_COVERAGE_COHORT,
    "tools/coupling_invariant_quality_gates.py",
    "tests/test_coupling_invariant_quality_gate.py",
]
COUPLING_INVARIANT_DOCSTRING_RATCHET = [
    *COUPLING_INVARIANT_TYPING_RATCHET,
]
COUPLING_INVARIANT_COVERAGE_DATA_FILE = "/tmp/scpn-qc-coupling-invariant-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-coupling-invariant-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *COUPLING_INVARIANT_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D coupling-invariant quality ratchet",
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
                *COUPLING_INVARIANT_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real module execution and exact source-coverage gates."""
    return [
        (
            "coupling-invariant focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={COUPLING_INVARIANT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *COUPLING_INVARIANT_COVERAGE_COHORT,
            ],
        ),
        (
            "coupling-invariant exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={COUPLING_INVARIANT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/studio/coupling_invariant.py",
            ],
        ),
    ]


__all__ = [
    "COUPLING_INVARIANT_COVERAGE_COHORT",
    "COUPLING_INVARIANT_COVERAGE_DATA_FILE",
    "COUPLING_INVARIANT_DOCSTRING_RATCHET",
    "COUPLING_INVARIANT_SOURCE",
    "COUPLING_INVARIANT_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
