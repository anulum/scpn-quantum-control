# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio executive benchmark quality gates
"""Build strict documentation and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
STUDIO_EXECUTIVE_BENCHMARK_QUALITY_RATCHET = [
    "src/scpn_quantum_control/studio/executive_benchmark.py",
    "src/scpn_quantum_control/studio/benchmark_databank_bundle.py",
    "tests/test_studio_executive_benchmark.py",
    "tests/test_studio_benchmark_databank_bundle.py",
    "tools/studio_executive_benchmark_quality_gates.py",
    "tests/test_studio_executive_benchmark_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
STUDIO_EXECUTIVE_BENCHMARK_COVERAGE_COHORT = [
    "tests/test_studio_executive_benchmark.py",
    "tests/test_studio_benchmark_databank_bundle.py",
]
"""Tests that own exact Studio benchmark-handler coverage."""
STUDIO_EXECUTIVE_BENCHMARK_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-studio-executive-benchmark-quality.coverage"
)
"""Isolated coverage database for Studio benchmark diagnostics."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-studio-executive-benchmark-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *STUDIO_EXECUTIVE_BENCHMARK_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D studio-executive-benchmark quality ratchet",
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
                *STUDIO_EXECUTIVE_BENCHMARK_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    data = STUDIO_EXECUTIVE_BENCHMARK_COVERAGE_DATA_FILE
    return [
        (
            "studio-executive-benchmark focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *STUDIO_EXECUTIVE_BENCHMARK_COVERAGE_COHORT,
            ],
        ),
        (
            "studio-executive-benchmark exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/studio/executive_benchmark.py,*/studio/benchmark_databank_bundle.py",
            ],
        ),
    ]


__all__ = [
    "STUDIO_EXECUTIVE_BENCHMARK_COVERAGE_COHORT",
    "STUDIO_EXECUTIVE_BENCHMARK_COVERAGE_DATA_FILE",
    "STUDIO_EXECUTIVE_BENCHMARK_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
