# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — benchmark CLI quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

BENCH_CLI_QUALITY_RATCHET = [
    "src/scpn_quantum_control/bench_cli.py",
    "tests/test_bench_cli.py",
    "tests/test_bench_cli_branches.py",
    "tests/test_execution_surface_policy.py",
    "tools/bench_cli_quality_gates.py",
    "tests/test_bench_cli_quality_gate.py",
]
"""Ordered strict-typing cohort."""

BENCH_CLI_DOCSTRING_RATCHET = [
    "src/scpn_quantum_control/bench_cli.py",
    "tests/test_bench_cli_branches.py",
    "tools/bench_cli_quality_gates.py",
    "tests/test_bench_cli_quality_gate.py",
]
"""Ordered NumPy-docstring cohort without inherited test-file debt."""

BENCH_CLI_COVERAGE_COHORT = [
    "tests/test_bench_cli.py",
    "tests/test_bench_cli_branches.py",
    "tests/test_execution_surface_policy.py",
]
"""Tests that own exact benchmark CLI coverage."""

BENCH_CLI_COVERAGE_DATA_FILE = "/tmp/scpn-qc-bench-cli-quality.coverage"  # nosec B108
"""Isolated coverage database for the benchmark CLI owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-bench-cli-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *BENCH_CLI_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D bench-cli quality ratchet",
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
                *BENCH_CLI_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "bench-cli focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={BENCH_CLI_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *BENCH_CLI_COVERAGE_COHORT,
            ],
        ),
        (
            "bench-cli exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={BENCH_CLI_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/bench_cli.py",
            ],
        ),
    ]


__all__ = [
    "BENCH_CLI_COVERAGE_COHORT",
    "BENCH_CLI_COVERAGE_DATA_FILE",
    "BENCH_CLI_DOCSTRING_RATCHET",
    "BENCH_CLI_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
