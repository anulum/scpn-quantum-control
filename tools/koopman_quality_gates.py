# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Koopman-analysis quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
KOOPMAN_SOURCE = "src/scpn_quantum_control/analysis/koopman.py"
KOOPMAN_COVERAGE_COHORT = [
    "tests/test_koopman.py",
    "tests/test_koopman_rust_generator_branches.py",
    "tests/test_rust_path_benchmarks.py",
]
KOOPMAN_TYPING_RATCHET = [
    KOOPMAN_SOURCE,
    "tools/koopman_quality_gates.py",
    "tests/test_koopman_quality_gate.py",
]
KOOPMAN_DOCSTRING_RATCHET = [
    KOOPMAN_SOURCE,
    "tests/test_koopman.py",
    "tests/test_koopman_rust_generator_branches.py",
    "tools/koopman_quality_gates.py",
    "tests/test_koopman_quality_gate.py",
]
KOOPMAN_COVERAGE_DATA_FILE = "/tmp/scpn-qc-koopman-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-koopman-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *KOOPMAN_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D Koopman quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "D,D413,D417",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *KOOPMAN_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline Koopman execution and exact coverage gates."""
    return [
        (
            "Koopman focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={KOOPMAN_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *KOOPMAN_COVERAGE_COHORT,
            ],
        ),
        (
            "Koopman exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={KOOPMAN_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/analysis/koopman.py",
            ],
        ),
    ]


__all__ = [
    "KOOPMAN_COVERAGE_COHORT",
    "KOOPMAN_COVERAGE_DATA_FILE",
    "KOOPMAN_DOCSTRING_RATCHET",
    "KOOPMAN_SOURCE",
    "KOOPMAN_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
