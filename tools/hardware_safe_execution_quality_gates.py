# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware-safe execution quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
HARDWARE_SAFE_EXECUTION_TYPING_RATCHET = [
    "src/scpn_quantum_control/hardware_safe_execution.py",
    "src/scpn_quantum_control/active_sensing_product.py",
    "src/scpn_quantum_control/hardware/feedback_capability_probe.py",
    "src/scpn_quantum_control/hardware/feedback_dryrun.py",
    "tests/test_hardware_safe_execution.py",
    "tests/test_active_sensing_product.py",
    "tools/hardware_safe_execution_quality_gates.py",
    "tests/test_hardware_safe_execution_quality_gate.py",
]
"""Ordered strict-typing owner."""
HARDWARE_SAFE_EXECUTION_QUALITY_RATCHET = [
    "src/scpn_quantum_control/hardware_safe_execution.py",
    "src/scpn_quantum_control/active_sensing_product.py",
    "src/scpn_quantum_control/hardware/feedback_capability_probe.py",
    "src/scpn_quantum_control/hardware/feedback_dryrun.py",
    "tests/test_hardware_safe_execution.py",
    "tests/test_active_sensing_product.py",
    "tests/test_feedback_capability_probe.py",
    "tests/test_feedback_capability_probe_branch.py",
    "tests/test_feedback_dryrun.py",
    "tests/test_feedback_dryrun_branch.py",
    "tools/hardware_safe_execution_quality_gates.py",
    "tests/test_hardware_safe_execution_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
HARDWARE_SAFE_EXECUTION_COVERAGE_COHORT = [
    "tests/test_hardware_safe_execution.py",
    "tests/test_active_sensing_product.py",
    "tests/test_feedback_capability_probe.py",
    "tests/test_feedback_capability_probe_branch.py",
    "tests/test_feedback_dryrun.py",
    "tests/test_feedback_dryrun_branch.py",
]
"""Tests that own hardware-safe, active-sensing, and capability coverage."""
HARDWARE_SAFE_EXECUTION_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-hardware-safe-execution-quality.coverage"
)
"""Isolated coverage database for the hardware-safe execution owner."""
HARDWARE_SAFE_EXECUTION_COVERAGE_INCLUDE = (
    "*/hardware_safe_execution.py,*/active_sensing_product.py,"
    "*/hardware/feedback_capability_probe.py,*/hardware/feedback_dryrun.py"
)
"""Production policy, active-sensing, and capability sources at exact coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-hardware-safe-execution-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *HARDWARE_SAFE_EXECUTION_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D hardware-safe-execution quality ratchet",
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
                *HARDWARE_SAFE_EXECUTION_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "hardware-safe-execution focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={HARDWARE_SAFE_EXECUTION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *HARDWARE_SAFE_EXECUTION_COVERAGE_COHORT,
            ],
        ),
        (
            "hardware-safe-execution exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={HARDWARE_SAFE_EXECUTION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={HARDWARE_SAFE_EXECUTION_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "HARDWARE_SAFE_EXECUTION_COVERAGE_COHORT",
    "HARDWARE_SAFE_EXECUTION_COVERAGE_DATA_FILE",
    "HARDWARE_SAFE_EXECUTION_COVERAGE_INCLUDE",
    "HARDWARE_SAFE_EXECUTION_QUALITY_RATCHET",
    "HARDWARE_SAFE_EXECUTION_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
