# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — AVQDS quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
AVQDS_SOURCE = "src/scpn_quantum_control/phase/avqds.py"
AVQDS_PRIMARY_TEST = "tests/test_avqds.py"
AVQDS_COVERAGE_COHORT = [AVQDS_PRIMARY_TEST]
AVQDS_TYPING_RATCHET = [
    AVQDS_SOURCE,
    AVQDS_PRIMARY_TEST,
    "tools/avqds_quality_gates.py",
    "tests/test_avqds_quality_gate.py",
]
AVQDS_DOCSTRING_RATCHET = [*AVQDS_TYPING_RATCHET]
AVQDS_COVERAGE_DATA_FILE = "/tmp/scpn-qc-avqds-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-avqds-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *AVQDS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D AVQDS quality ratchet",
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
                *AVQDS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build bounded local AVQDS execution and exact source-coverage gates."""
    return [
        (
            "AVQDS focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={AVQDS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *AVQDS_COVERAGE_COHORT,
            ],
        ),
        (
            "AVQDS exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={AVQDS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/avqds.py",
            ],
        ),
    ]


__all__ = [
    "AVQDS_COVERAGE_COHORT",
    "AVQDS_COVERAGE_DATA_FILE",
    "AVQDS_DOCSTRING_RATCHET",
    "AVQDS_PRIMARY_TEST",
    "AVQDS_SOURCE",
    "AVQDS_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
