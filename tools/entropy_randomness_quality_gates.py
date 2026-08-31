# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — entropy and randomness quality-gate specification
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
ENTROPY_RANDOMNESS_SOURCES = [
    "src/scpn_quantum_control/entropy/nist_sp800_22.py",
    "src/scpn_quantum_control/entropy/qrng_stream.py",
]
"""NIST and buffered QRNG production owner."""
ENTROPY_RANDOMNESS_TESTS = [
    "tests/test_entropy_qrng.py",
    "tests/test_nist_sp800_22_guards.py",
    "tests/test_qrng_stream.py",
]
"""Worked-example, guard, fallback, and stream behavioral cohort."""
ENTROPY_RANDOMNESS_TYPING_RATCHET = [
    *ENTROPY_RANDOMNESS_SOURCES,
    *ENTROPY_RANDOMNESS_TESTS,
    "tools/entropy_randomness_quality_gates.py",
    "tests/test_entropy_randomness_quality_gate.py",
    "tools/preflight.py",
    "tests/test_preflight_tool.py",
]
"""Production, tests, and gate surfaces held to strict MyPy."""
ENTROPY_RANDOMNESS_DOCSTRING_RATCHET = [
    *ENTROPY_RANDOMNESS_SOURCES,
    *ENTROPY_RANDOMNESS_TESTS,
    "tools/entropy_randomness_quality_gates.py",
    "tests/test_entropy_randomness_quality_gate.py",
    "tests/test_preflight_tool.py",
]
"""Whole owner cohort held to complete NumPy docstrings."""
ENTROPY_RANDOMNESS_COVERAGE_DATA_FILE = "/tmp/scpn-qc-entropy-randomness-quality.coverage"  # nosec B108
ENTROPY_RANDOMNESS_COVERAGE_INCLUDE = "*/entropy/nist_sp800_22.py,*/entropy/qrng_stream.py"


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-entropy-randomness-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ENTROPY_RANDOMNESS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D entropy-randomness quality ratchet",
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
                *ENTROPY_RANDOMNESS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real entropy execution and exact source-coverage gates."""
    return [
        (
            "entropy-randomness focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ENTROPY_RANDOMNESS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ENTROPY_RANDOMNESS_TESTS,
            ],
        ),
        (
            "entropy-randomness exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ENTROPY_RANDOMNESS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={ENTROPY_RANDOMNESS_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "ENTROPY_RANDOMNESS_COVERAGE_DATA_FILE",
    "ENTROPY_RANDOMNESS_COVERAGE_INCLUDE",
    "ENTROPY_RANDOMNESS_DOCSTRING_RATCHET",
    "ENTROPY_RANDOMNESS_SOURCES",
    "ENTROPY_RANDOMNESS_TESTS",
    "ENTROPY_RANDOMNESS_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
