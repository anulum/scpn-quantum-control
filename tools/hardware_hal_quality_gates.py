# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware HAL quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
HARDWARE_HAL_SOURCE = "src/scpn_quantum_control/hardware/hal.py"
"""Production source owned by the provider-neutral HAL."""
HARDWARE_HAL_COVERAGE_COHORT = [
    "tests/test_hardware_hal.py",
    "tests/test_hardware_hal_contract_guards.py",
    "tests/test_hardware_hal_count_integrity_contract.py",
    "tests/test_hardware_hal_provider_id_contract.py",
    "tests/test_hardware_hal_status_normalisation_contract.py",
]
"""Offline and fake-adapter tests that own exact HAL coverage."""
HARDWARE_HAL_TYPING_RATCHET = [
    HARDWARE_HAL_SOURCE,
    "tools/hardware_hal_quality_gates.py",
    "tests/test_hardware_hal_quality_gate.py",
]
"""Strict-typing cohort for production and gate contracts."""
HARDWARE_HAL_DOCSTRING_RATCHET = [
    HARDWARE_HAL_SOURCE,
    "tests/test_hardware_hal.py",
    "tools/hardware_hal_quality_gates.py",
    "tests/test_hardware_hal_quality_gate.py",
]
"""Complete HAL and gate-contract docstring cohort."""
HARDWARE_HAL_COVERAGE_DATA_FILE = "/tmp/scpn-qc-hardware-hal-quality.coverage"  # nosec B108
"""Isolated coverage database for the hardware HAL owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-hardware-hal-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *HARDWARE_HAL_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D hardware-hal quality ratchet",
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
                *HARDWARE_HAL_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    return [
        (
            "hardware-hal focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={HARDWARE_HAL_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *HARDWARE_HAL_COVERAGE_COHORT,
            ],
        ),
        (
            "hardware-hal exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={HARDWARE_HAL_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hardware/hal.py",
            ],
        ),
    ]


__all__ = [
    "HARDWARE_HAL_COVERAGE_COHORT",
    "HARDWARE_HAL_COVERAGE_DATA_FILE",
    "HARDWARE_HAL_DOCSTRING_RATCHET",
    "HARDWARE_HAL_SOURCE",
    "HARDWARE_HAL_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
