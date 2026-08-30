# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — error-aware chain quality gates
"""Build strict documentation and exact coverage gates for chain selection."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

ERROR_AWARE_CHAIN_QUALITY_RATCHET = [
    "src/scpn_quantum_control/hardware/error_aware_chain.py",
    "tests/test_error_aware_chain.py",
    "tools/error_aware_chain_quality_gates.py",
    "tests/test_error_aware_chain_quality_gate.py",
]
"""Complete strict-typing and preview-documentation owner."""

ERROR_AWARE_CHAIN_COVERAGE_COHORT = ["tests/test_error_aware_chain.py"]
"""Real calibrated-graph, greedy, and bounded-DFS execution suite."""

ERROR_AWARE_CHAIN_COVERAGE_DATA_FILE = "/tmp/scpn-qc-error-aware-chain-quality.coverage"
"""Isolated coverage database for the chain-selection owner."""

ERROR_AWARE_CHAIN_COVERAGE_INCLUDE = "*/hardware/error_aware_chain.py"
"""Production source enforced at exact branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and preview NumPy-documentation gates."""
    return [
        (
            "mypy-strict-error-aware-chain-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ERROR_AWARE_CHAIN_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D error-aware-chain quality ratchet",
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
                *ERROR_AWARE_CHAIN_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real execution and exact source-coverage gates."""
    return [
        (
            "error-aware-chain focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ERROR_AWARE_CHAIN_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ERROR_AWARE_CHAIN_COVERAGE_COHORT,
            ],
        ),
        (
            "error-aware-chain exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ERROR_AWARE_CHAIN_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={ERROR_AWARE_CHAIN_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "ERROR_AWARE_CHAIN_COVERAGE_COHORT",
    "ERROR_AWARE_CHAIN_COVERAGE_DATA_FILE",
    "ERROR_AWARE_CHAIN_COVERAGE_INCLUDE",
    "ERROR_AWARE_CHAIN_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
