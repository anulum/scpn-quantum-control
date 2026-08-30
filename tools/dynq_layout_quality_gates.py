# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DynQ layout quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
QUBIT_MAPPER_SOURCE = "src/scpn_quantum_control/hardware/qubit_mapper.py"
DYNQ_LAYOUT_PASS_SOURCE = "src/scpn_quantum_control/hardware/dynq_layout_pass.py"
DYNQ_LAYOUT_COVERAGE_COHORT = [
    "tests/test_qubit_mapper.py",
    "tests/test_dynq_layout_pass.py",
]
DYNQ_LAYOUT_TYPING_RATCHET = [
    QUBIT_MAPPER_SOURCE,
    DYNQ_LAYOUT_PASS_SOURCE,
    *DYNQ_LAYOUT_COVERAGE_COHORT,
    "tools/dynq_layout_quality_gates.py",
    "tests/test_dynq_layout_quality_gate.py",
]
DYNQ_LAYOUT_DOCSTRING_RATCHET = [*DYNQ_LAYOUT_TYPING_RATCHET]
DYNQ_LAYOUT_COVERAGE_DATA_FILE = "/tmp/scpn-qc-dynq-layout-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-dynq-layout-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DYNQ_LAYOUT_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D DynQ layout quality ratchet",
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
                *DYNQ_LAYOUT_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build connected execution and exact source-coverage gates."""
    return [
        (
            "DynQ layout focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DYNQ_LAYOUT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DYNQ_LAYOUT_COVERAGE_COHORT,
            ],
        ),
        (
            "DynQ layout exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DYNQ_LAYOUT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hardware/qubit_mapper.py,*/hardware/dynq_layout_pass.py",
            ],
        ),
    ]


__all__ = [
    "DYNQ_LAYOUT_COVERAGE_COHORT",
    "DYNQ_LAYOUT_COVERAGE_DATA_FILE",
    "DYNQ_LAYOUT_DOCSTRING_RATCHET",
    "DYNQ_LAYOUT_PASS_SOURCE",
    "DYNQ_LAYOUT_TYPING_RATCHET",
    "QUBIT_MAPPER_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
