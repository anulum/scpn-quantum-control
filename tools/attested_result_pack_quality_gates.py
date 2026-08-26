# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — attested result-pack quality gates
"""Build strict documentation, typing, and coverage gates for BL-19."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
ATTESTED_RESULT_PACK_QUALITY_RATCHET = [
    "src/scpn_quantum_control/attested_result_pack.py",
    "tests/test_attested_result_pack.py",
    "tools/attested_result_pack_quality_gates.py",
    "tests/test_attested_result_pack_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
ATTESTED_RESULT_PACK_COVERAGE_COHORT = ["tests/test_attested_result_pack.py"]
"""Tests that own exact attested result-pack coverage."""
ATTESTED_RESULT_PACK_COVERAGE_DATA_FILE = ".coverage.attested-result-pack-quality"
"""Isolated coverage database for attested result packs."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-attested-result-pack-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ATTESTED_RESULT_PACK_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D attested-result-pack quality ratchet",
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
                *ATTESTED_RESULT_PACK_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    data = ATTESTED_RESULT_PACK_COVERAGE_DATA_FILE
    return [
        (
            "attested-result-pack focused coverage",
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
                *ATTESTED_RESULT_PACK_COVERAGE_COHORT,
            ],
        ),
        (
            "attested-result-pack exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/attested_result_pack.py",
            ],
        ),
    ]


__all__ = [
    "ATTESTED_RESULT_PACK_COVERAGE_COHORT",
    "ATTESTED_RESULT_PACK_COVERAGE_DATA_FILE",
    "ATTESTED_RESULT_PACK_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
