# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Ansatz-benchmark quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
ANSATZ_BENCH_SOURCE = "src/scpn_quantum_control/phase/ansatz_bench.py"
ANSATZ_BENCH_PRIMARY_TEST = "tests/test_ansatz_bench.py"
APPQSIM_PROTOCOL_SOURCE = "src/scpn_quantum_control/benchmarks/appqsim_protocol.py"
APPQSIM_PROTOCOL_DIRECT_TEST = "tests/test_appqsim_protocol.py"
ANSATZ_BENCH_COVERAGE_COHORT = [
    ANSATZ_BENCH_PRIMARY_TEST,
    APPQSIM_PROTOCOL_DIRECT_TEST,
]
ANSATZ_BENCH_COVERAGE_INCLUDE = "*/phase/ansatz_bench.py,*/benchmarks/appqsim_protocol.py"
ANSATZ_BENCH_TYPING_RATCHET = [
    ANSATZ_BENCH_SOURCE,
    ANSATZ_BENCH_PRIMARY_TEST,
    APPQSIM_PROTOCOL_SOURCE,
    APPQSIM_PROTOCOL_DIRECT_TEST,
    "tools/ansatz_bench_quality_gates.py",
    "tests/test_ansatz_bench_quality_gate.py",
]
ANSATZ_BENCH_DOCSTRING_RATCHET = [*ANSATZ_BENCH_TYPING_RATCHET]
ANSATZ_BENCH_COVERAGE_DATA_FILE = "/tmp/scpn-qc-ansatz-benchmark-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-ansatz-benchmark-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ANSATZ_BENCH_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D ansatz-benchmark quality ratchet",
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
                *ANSATZ_BENCH_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build bounded local VQE execution and exact source-coverage gates."""
    return [
        (
            "ansatz-benchmark focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ANSATZ_BENCH_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ANSATZ_BENCH_COVERAGE_COHORT,
            ],
        ),
        (
            "ansatz-benchmark exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ANSATZ_BENCH_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={ANSATZ_BENCH_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "ANSATZ_BENCH_COVERAGE_COHORT",
    "ANSATZ_BENCH_COVERAGE_DATA_FILE",
    "ANSATZ_BENCH_COVERAGE_INCLUDE",
    "ANSATZ_BENCH_DOCSTRING_RATCHET",
    "ANSATZ_BENCH_PRIMARY_TEST",
    "ANSATZ_BENCH_SOURCE",
    "ANSATZ_BENCH_TYPING_RATCHET",
    "APPQSIM_PROTOCOL_DIRECT_TEST",
    "APPQSIM_PROTOCOL_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
