# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Rust accelerator import-resilience quality gates
"""Build strict documentation and exact coverage gates for import resilience."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

RUST_ACCEL_IMPORT_RESILIENCE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/_rust_accel.py",
    "scripts/verify_hardware_result_packs.py",
    "tests/test_rust_accel_import_resilience.py",
    "tools/rust_accel_import_resilience_quality_gates.py",
    "tests/test_rust_accel_import_resilience_quality_gate.py",
]
"""Complete strict-typing and preview-documentation owner."""

RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_COHORT = ["tests/test_rust_accel_import_resilience.py"]
"""Real optional-accelerator and standalone-verifier import-resilience suite."""

RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-rust-accel-import-resilience-quality.coverage"
)
"""Isolated coverage database for the optional accelerator owner."""

RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_INCLUDE = "*/scpn_quantum_control/_rust_accel.py"
"""Production source enforced at exact branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and preview NumPy-documentation gates."""
    return [
        (
            "mypy-strict-rust-accel-import-resilience-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *RUST_ACCEL_IMPORT_RESILIENCE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D rust-accel-import-resilience quality ratchet",
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
                *RUST_ACCEL_IMPORT_RESILIENCE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real execution and exact source-coverage gates."""
    return [
        (
            "rust-accel-import-resilience focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_COHORT,
            ],
        ),
        (
            "rust-accel-import-resilience exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_COHORT",
    "RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_DATA_FILE",
    "RUST_ACCEL_IMPORT_RESILIENCE_COVERAGE_INCLUDE",
    "RUST_ACCEL_IMPORT_RESILIENCE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
