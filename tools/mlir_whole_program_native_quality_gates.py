# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR whole-program native quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
MLIR_WHOLE_PROGRAM_NATIVE_SOURCE = "src/scpn_quantum_control/compiler/mlir_whole_program_native.py"
MLIR_WHOLE_PROGRAM_NATIVE_COVERAGE_COHORT = [
    "tests/test_mlir_whole_program_native.py",
    "tests/test_mlir_whole_program_native_quality_edges.py",
]
MLIR_WHOLE_PROGRAM_NATIVE_TYPING_RATCHET = [
    MLIR_WHOLE_PROGRAM_NATIVE_SOURCE,
    *MLIR_WHOLE_PROGRAM_NATIVE_COVERAGE_COHORT,
    "tools/mlir_whole_program_native_quality_gates.py",
    "tests/test_mlir_whole_program_native_quality_gate.py",
]
MLIR_WHOLE_PROGRAM_NATIVE_DOCSTRING_RATCHET = [
    *MLIR_WHOLE_PROGRAM_NATIVE_TYPING_RATCHET,
]
MLIR_WHOLE_PROGRAM_NATIVE_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-mlir-whole-program-native-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-mlir-whole-program-native-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *MLIR_WHOLE_PROGRAM_NATIVE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D MLIR whole-program native quality ratchet",
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
                *MLIR_WHOLE_PROGRAM_NATIVE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real native execution and exact source-coverage gates."""
    return [
        (
            "MLIR whole-program native focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={MLIR_WHOLE_PROGRAM_NATIVE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *MLIR_WHOLE_PROGRAM_NATIVE_COVERAGE_COHORT,
            ],
        ),
        (
            "MLIR whole-program native exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={MLIR_WHOLE_PROGRAM_NATIVE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/compiler/mlir_whole_program_native.py",
            ],
        ),
    ]


__all__ = [
    "MLIR_WHOLE_PROGRAM_NATIVE_COVERAGE_COHORT",
    "MLIR_WHOLE_PROGRAM_NATIVE_COVERAGE_DATA_FILE",
    "MLIR_WHOLE_PROGRAM_NATIVE_DOCSTRING_RATCHET",
    "MLIR_WHOLE_PROGRAM_NATIVE_SOURCE",
    "MLIR_WHOLE_PROGRAM_NATIVE_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
