# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — pulse-shaping quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PULSE_SHAPING_SOURCE = "src/scpn_quantum_control/phase/pulse_shaping.py"
PULSE_SHAPING_COVERAGE_COHORT = [
    "tests/test_pulse_shaping.py",
    "tests/test_pulse_shaping_python_fallback.py",
]
PULSE_SHAPING_TYPING_RATCHET = [
    PULSE_SHAPING_SOURCE,
    "tools/pulse_shaping_quality_gates.py",
    "tests/test_pulse_shaping_quality_gate.py",
]
PULSE_SHAPING_DOCSTRING_RATCHET = [
    PULSE_SHAPING_SOURCE,
    *PULSE_SHAPING_COVERAGE_COHORT,
    "tools/pulse_shaping_quality_gates.py",
    "tests/test_pulse_shaping_quality_gate.py",
]
PULSE_SHAPING_COVERAGE_DATA_FILE = "/tmp/scpn-qc-pulse-shaping-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-pulse-shaping-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PULSE_SHAPING_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D pulse-shaping quality ratchet",
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
                *PULSE_SHAPING_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline pulse execution and exact source coverage gates."""
    return [
        (
            "pulse-shaping focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PULSE_SHAPING_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PULSE_SHAPING_COVERAGE_COHORT,
            ],
        ),
        (
            "pulse-shaping exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PULSE_SHAPING_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/pulse_shaping.py",
            ],
        ),
    ]


__all__ = [
    "PULSE_SHAPING_COVERAGE_COHORT",
    "PULSE_SHAPING_COVERAGE_DATA_FILE",
    "PULSE_SHAPING_DOCSTRING_RATCHET",
    "PULSE_SHAPING_SOURCE",
    "PULSE_SHAPING_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
