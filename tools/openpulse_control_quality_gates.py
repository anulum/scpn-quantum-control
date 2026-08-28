# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — OpenPulse-control quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
OPENPULSE_CONTROL_SOURCE = "src/scpn_quantum_control/hardware/openpulse_control.py"
OPENPULSE_CONTROL_COVERAGE_COHORT = ["tests/test_openpulse_control.py"]
OPENPULSE_CONTROL_TYPING_RATCHET = [
    OPENPULSE_CONTROL_SOURCE,
    "tools/openpulse_control_quality_gates.py",
    "tests/test_openpulse_control_quality_gate.py",
]
OPENPULSE_CONTROL_DOCSTRING_RATCHET = [
    OPENPULSE_CONTROL_SOURCE,
    *OPENPULSE_CONTROL_COVERAGE_COHORT,
    "tools/openpulse_control_quality_gates.py",
    "tests/test_openpulse_control_quality_gate.py",
]
OPENPULSE_CONTROL_COVERAGE_DATA_FILE = "/tmp/scpn-qc-openpulse-control-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-openpulse-control-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *OPENPULSE_CONTROL_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D openpulse-control quality ratchet",
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
                *OPENPULSE_CONTROL_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline OpenPulse execution and exact source coverage gates."""
    return [
        (
            "openpulse-control focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={OPENPULSE_CONTROL_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *OPENPULSE_CONTROL_COVERAGE_COHORT,
            ],
        ),
        (
            "openpulse-control exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={OPENPULSE_CONTROL_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hardware/openpulse_control.py",
            ],
        ),
    ]


__all__ = [
    "OPENPULSE_CONTROL_COVERAGE_COHORT",
    "OPENPULSE_CONTROL_COVERAGE_DATA_FILE",
    "OPENPULSE_CONTROL_DOCSTRING_RATCHET",
    "OPENPULSE_CONTROL_SOURCE",
    "OPENPULSE_CONTROL_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
