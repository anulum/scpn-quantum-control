# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — control StructuredAnsatz quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
CONTROL_STRUCTURED_ANSATZ_SOURCE = "src/scpn_quantum_control/control/structured_ansatz.py"
CONTROL_STRUCTURED_ANSATZ_COVERAGE_COHORT = [
    "tests/test_control_structured_ansatz.py",
    "tests/test_structured_ansatz_branches.py",
]
CONTROL_STRUCTURED_ANSATZ_TYPING_RATCHET = [
    CONTROL_STRUCTURED_ANSATZ_SOURCE,
    *CONTROL_STRUCTURED_ANSATZ_COVERAGE_COHORT,
    "tools/control_structured_ansatz_quality_gates.py",
    "tests/test_control_structured_ansatz_quality_gate.py",
]
CONTROL_STRUCTURED_ANSATZ_DOCSTRING_RATCHET = [*CONTROL_STRUCTURED_ANSATZ_TYPING_RATCHET]
CONTROL_STRUCTURED_ANSATZ_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-control-structured-ansatz-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-control-structured-ansatz-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *CONTROL_STRUCTURED_ANSATZ_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D control-StructuredAnsatz quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "D,D107,D413,D417,D420",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *CONTROL_STRUCTURED_ANSATZ_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real circuit execution and exact source-coverage gates."""
    return [
        (
            "control-StructuredAnsatz focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={CONTROL_STRUCTURED_ANSATZ_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *CONTROL_STRUCTURED_ANSATZ_COVERAGE_COHORT,
            ],
        ),
        (
            "control-StructuredAnsatz exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={CONTROL_STRUCTURED_ANSATZ_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/control/structured_ansatz.py",
            ],
        ),
    ]


__all__ = [
    "CONTROL_STRUCTURED_ANSATZ_COVERAGE_COHORT",
    "CONTROL_STRUCTURED_ANSATZ_COVERAGE_DATA_FILE",
    "CONTROL_STRUCTURED_ANSATZ_DOCSTRING_RATCHET",
    "CONTROL_STRUCTURED_ANSATZ_SOURCE",
    "CONTROL_STRUCTURED_ANSATZ_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
