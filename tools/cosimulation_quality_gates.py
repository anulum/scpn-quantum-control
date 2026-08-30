# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum/classical cosimulation quality gates
"""Build strict documentation, typing, and exact cosimulation coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

COSIMULATION_TYPING_RATCHET = [
    "src/scpn_quantum_control/cosimulation/knm_partition.py",
    "src/scpn_quantum_control/cosimulation/quantum_classical.py",
    "tests/test_knm_partition_branches.py",
    "tools/cosimulation_quality_gates.py",
    "tests/test_cosimulation_quality_gate.py",
]
"""Ordered strict-typing cohort for the admitted partition owner."""
COSIMULATION_DOCSTRING_RATCHET = [
    "src/scpn_quantum_control/cosimulation/knm_partition.py",
    "src/scpn_quantum_control/cosimulation/quantum_classical.py",
    "tests/test_cosimulation.py",
    "tests/test_knm_partition_branches.py",
    "tools/cosimulation_quality_gates.py",
    "tests/test_cosimulation_quality_gate.py",
]
"""Ordered complete NumPy-docstring cohort."""
COSIMULATION_COVERAGE_COHORT = [
    "tests/test_cosimulation.py",
    "tests/test_knm_partition_branches.py",
]
"""Real tests that own exact K_nm partition coverage."""
COSIMULATION_COVERAGE_DATA_FILE = "/tmp/scpn-qc-cosimulation-quality.coverage"  # nosec B108
"""Isolated coverage database for quantum/classical cosimulation."""
COSIMULATION_COVERAGE_INCLUDE = (
    "*/cosimulation/knm_partition.py,*/cosimulation/quantum_classical.py"
)
"""Production source enforced at exact branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-cosimulation-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *COSIMULATION_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D cosimulation quality ratchet",
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
                *COSIMULATION_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "cosimulation focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={COSIMULATION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *COSIMULATION_COVERAGE_COHORT,
            ],
        ),
        (
            "cosimulation exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={COSIMULATION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={COSIMULATION_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "COSIMULATION_COVERAGE_COHORT",
    "COSIMULATION_COVERAGE_DATA_FILE",
    "COSIMULATION_COVERAGE_INCLUDE",
    "COSIMULATION_DOCSTRING_RATCHET",
    "COSIMULATION_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
