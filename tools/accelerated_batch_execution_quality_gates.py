# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — accelerated batch execution quality gates
"""Build strict typing, documentation, execution and exact-coverage gates for batched evolution."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
ACCELERATED_BATCH_EXECUTION_SOURCE = "src/scpn_quantum_control/benchmarks/gpu_baseline.py"
ACCELERATED_BATCH_EXECUTION_COVERAGE_COHORT = [
    "tests/test_accelerated_batch_execution.py",
    "tests/test_gpu_baseline.py",
]
ACCELERATED_BATCH_EXECUTION_TYPING_RATCHET = [
    ACCELERATED_BATCH_EXECUTION_SOURCE,
    "tests/test_accelerated_batch_execution.py",
    "tools/accelerated_batch_execution_quality_gates.py",
    "tests/test_accelerated_batch_execution_quality_gate.py",
]
ACCELERATED_BATCH_EXECUTION_DOCSTRING_RATCHET = [*ACCELERATED_BATCH_EXECUTION_TYPING_RATCHET]
ACCELERATED_BATCH_EXECUTION_COVERAGE_DATA_FILE = (  # nosec B108
    "/tmp/scpn-qc-accelerated-batch-execution-quality.coverage"
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-accelerated-batch-execution-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ACCELERATED_BATCH_EXECUTION_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D accelerated-batch-execution quality ratchet",
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
                *ACCELERATED_BATCH_EXECUTION_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build batched execution and exact source-coverage gates."""
    return [
        (
            "accelerated-batch-execution focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ACCELERATED_BATCH_EXECUTION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ACCELERATED_BATCH_EXECUTION_COVERAGE_COHORT,
            ],
        ),
        (
            "accelerated-batch-execution exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ACCELERATED_BATCH_EXECUTION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/benchmarks/gpu_baseline.py",
            ],
        ),
    ]


__all__ = [
    "ACCELERATED_BATCH_EXECUTION_COVERAGE_COHORT",
    "ACCELERATED_BATCH_EXECUTION_COVERAGE_DATA_FILE",
    "ACCELERATED_BATCH_EXECUTION_DOCSTRING_RATCHET",
    "ACCELERATED_BATCH_EXECUTION_SOURCE",
    "ACCELERATED_BATCH_EXECUTION_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
