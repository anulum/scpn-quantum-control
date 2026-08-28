# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Josephson magnitude-study quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
JOSEPHSON_MAGNITUDE_STUDY_SOURCE = (
    "src/scpn_quantum_control/applications/josephson_magnitude_study.py"
)
JOSEPHSON_MAGNITUDE_STUDY_COVERAGE_COHORT = [
    "tests/test_josephson_magnitude_study.py",
]
JOSEPHSON_MAGNITUDE_STUDY_TYPING_RATCHET = [
    JOSEPHSON_MAGNITUDE_STUDY_SOURCE,
    *JOSEPHSON_MAGNITUDE_STUDY_COVERAGE_COHORT,
    "tools/josephson_magnitude_study_quality_gates.py",
    "tests/test_josephson_magnitude_study_quality_gate.py",
]
JOSEPHSON_MAGNITUDE_STUDY_DOCSTRING_RATCHET = [
    *JOSEPHSON_MAGNITUDE_STUDY_TYPING_RATCHET,
]
JOSEPHSON_MAGNITUDE_STUDY_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-josephson-magnitude-study-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-josephson-magnitude-study-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *JOSEPHSON_MAGNITUDE_STUDY_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D Josephson magnitude-study quality ratchet",
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
                *JOSEPHSON_MAGNITUDE_STUDY_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real study execution and exact source-coverage gates."""
    return [
        (
            "Josephson magnitude-study focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={JOSEPHSON_MAGNITUDE_STUDY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *JOSEPHSON_MAGNITUDE_STUDY_COVERAGE_COHORT,
            ],
        ),
        (
            "Josephson magnitude-study exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={JOSEPHSON_MAGNITUDE_STUDY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/applications/josephson_magnitude_study.py",
            ],
        ),
    ]


__all__ = [
    "JOSEPHSON_MAGNITUDE_STUDY_COVERAGE_COHORT",
    "JOSEPHSON_MAGNITUDE_STUDY_COVERAGE_DATA_FILE",
    "JOSEPHSON_MAGNITUDE_STUDY_DOCSTRING_RATCHET",
    "JOSEPHSON_MAGNITUDE_STUDY_SOURCE",
    "JOSEPHSON_MAGNITUDE_STUDY_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
