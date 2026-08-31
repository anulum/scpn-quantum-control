# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable-programming contract quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

DIFFERENTIABLE_PROGRAMMING_CONTRACTS_SOURCE = (
    "src/scpn_quantum_control/benchmarks/differentiable_programming_contracts.py"
)
"""Dependency-light differentiable benchmark result contracts."""
DIFFERENTIABLE_PROGRAMMING_CONTRACTS_TEST = "tests/test_differentiable_programming_contracts.py"
"""Public validation, facade-alias, and leaf-ownership tests."""
DIFFERENTIABLE_PROGRAMMING_CONTRACTS_QUALITY_RATCHET = [
    DIFFERENTIABLE_PROGRAMMING_CONTRACTS_SOURCE,
    DIFFERENTIABLE_PROGRAMMING_CONTRACTS_TEST,
    "tools/differentiable_programming_contracts_quality_gates.py",
    "tests/test_differentiable_programming_contracts_quality_gate.py",
]
"""Strict-typing and complete-preview-documentation owner."""
DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_COHORT = [
    DIFFERENTIABLE_PROGRAMMING_CONTRACTS_TEST,
]
"""Real public-surface suite that owns exact contract coverage."""
DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-differentiable-programming-contracts-quality.coverage"  # nosec B108
)
"""Isolated coverage database for the benchmark-contract owner."""
DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_INCLUDE = (
    "*/benchmarks/differentiable_programming_contracts.py"
)
"""Contract leaf enforced at exact statement and branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-documentation gates."""
    return [
        (
            "mypy-strict-differentiable-programming-contracts-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DIFFERENTIABLE_PROGRAMMING_CONTRACTS_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D differentiable-programming-contracts quality ratchet",
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
                "lint.explicit-preview-rules = true",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *DIFFERENTIABLE_PROGRAMMING_CONTRACTS_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-coverage gates."""
    return [
        (
            "differentiable-programming-contracts focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_COHORT,
            ],
        ),
        (
            "differentiable-programming-contracts exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_COHORT",
    "DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_DATA_FILE",
    "DIFFERENTIABLE_PROGRAMMING_CONTRACTS_COVERAGE_INCLUDE",
    "DIFFERENTIABLE_PROGRAMMING_CONTRACTS_QUALITY_RATCHET",
    "DIFFERENTIABLE_PROGRAMMING_CONTRACTS_SOURCE",
    "DIFFERENTIABLE_PROGRAMMING_CONTRACTS_TEST",
    "build_coverage_gates",
    "build_static_quality_gates",
]
