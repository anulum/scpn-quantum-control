# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — XY compiler quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
XY_COMPILER_SOURCE = "src/scpn_quantum_control/phase/xy_compiler.py"
XY_COMPILER_OWNER_TEST = "tests/test_xy_compiler.py"
XY_COMPILER_CONTRACT_TEST = "tests/test_xy_compiler_contracts.py"
XY_COMPILER_TYPING_RATCHET = [
    XY_COMPILER_SOURCE,
    XY_COMPILER_OWNER_TEST,
    "tools/xy_compiler_quality_gates.py",
    "tests/test_xy_compiler_quality_gate.py",
]
XY_COMPILER_DOCSTRING_RATCHET = [*XY_COMPILER_TYPING_RATCHET]
XY_COMPILER_COVERAGE_COHORT = [XY_COMPILER_OWNER_TEST, XY_COMPILER_CONTRACT_TEST]
XY_COMPILER_COVERAGE_DATA_FILE = "/tmp/scpn-qc-xy-compiler-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-XY-compiler-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *XY_COMPILER_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D XY-compiler quality ratchet",
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
                *XY_COMPILER_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real compiler execution and exact source-coverage gates."""
    return [
        (
            "XY-compiler focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={XY_COMPILER_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *XY_COMPILER_COVERAGE_COHORT,
            ],
        ),
        (
            "XY-compiler exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={XY_COMPILER_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/xy_compiler.py",
            ],
        ),
    ]


__all__ = [
    "XY_COMPILER_CONTRACT_TEST",
    "XY_COMPILER_COVERAGE_COHORT",
    "XY_COMPILER_COVERAGE_DATA_FILE",
    "XY_COMPILER_DOCSTRING_RATCHET",
    "XY_COMPILER_OWNER_TEST",
    "XY_COMPILER_SOURCE",
    "XY_COMPILER_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
