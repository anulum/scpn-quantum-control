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
ASYNC_HARDWARE_RUNNER_SOURCE = "src/scpn_quantum_control/hardware/async_runner.py"
"""Bounded asynchronous orchestration over hardware runners."""
ASYNC_HARDWARE_RUNNER_TEST = "tests/test_async_runner.py"
"""Offline fake-adapter orchestration and provenance tests."""
HARDWARE_CIRCUIT_CUTTING_SOURCE = "src/scpn_quantum_control/hardware/circuit_cutting.py"
"""Bounded circuit-partition and reconstruction-overhead planner."""
HARDWARE_CIRCUIT_CUTTING_TEST = "tests/test_circuit_cutting.py"
"""Real coupling-matrix partition and scaling tests."""
HARDWARE_CIRCUIT_EXPORT_SOURCE = "src/scpn_quantum_control/hardware/circuit_export.py"
"""Multi-platform circuit construction and serialization surface."""
HARDWARE_CIRCUIT_EXPORT_TEST = "tests/test_circuit_export.py"
"""Real circuit construction plus offline conversion-path tests."""
HARDWARE_HAL_COVERAGE_COHORT = [
    "tests/test_hardware_hal.py",
    "tests/test_hardware_hal_contract_guards.py",
    "tests/test_hardware_hal_count_integrity_contract.py",
    "tests/test_hardware_hal_provider_id_contract.py",
    "tests/test_hardware_hal_status_normalisation_contract.py",
    ASYNC_HARDWARE_RUNNER_TEST,
    HARDWARE_CIRCUIT_CUTTING_TEST,
    HARDWARE_CIRCUIT_EXPORT_TEST,
]
"""Offline and fake-adapter tests that own exact HAL coverage."""
HARDWARE_HAL_TYPING_RATCHET = [
    HARDWARE_HAL_SOURCE,
    ASYNC_HARDWARE_RUNNER_SOURCE,
    ASYNC_HARDWARE_RUNNER_TEST,
    HARDWARE_CIRCUIT_CUTTING_SOURCE,
    HARDWARE_CIRCUIT_CUTTING_TEST,
    HARDWARE_CIRCUIT_EXPORT_SOURCE,
    HARDWARE_CIRCUIT_EXPORT_TEST,
    "tools/hardware_hal_quality_gates.py",
    "tests/test_hardware_hal_quality_gate.py",
]
"""Strict-typing cohort for production and gate contracts."""
HARDWARE_HAL_DOCSTRING_RATCHET = [
    HARDWARE_HAL_SOURCE,
    ASYNC_HARDWARE_RUNNER_SOURCE,
    HARDWARE_CIRCUIT_CUTTING_SOURCE,
    HARDWARE_CIRCUIT_EXPORT_SOURCE,
    "tests/test_hardware_hal.py",
    ASYNC_HARDWARE_RUNNER_TEST,
    HARDWARE_CIRCUIT_CUTTING_TEST,
    HARDWARE_CIRCUIT_EXPORT_TEST,
    "tools/hardware_hal_quality_gates.py",
    "tests/test_hardware_hal_quality_gate.py",
]
"""Complete HAL and gate-contract docstring cohort."""
HARDWARE_HAL_COVERAGE_DATA_FILE = "/tmp/scpn-qc-hardware-hal-quality.coverage"  # nosec B108
"""Isolated coverage database for the hardware HAL owner."""
HARDWARE_HAL_COVERAGE_INCLUDE = (
    "*/hardware/hal.py,*/hardware/async_runner.py,*/hardware/circuit_cutting.py,"
    "*/hardware/circuit_export.py"
)
"""Provider-neutral and asynchronous hardware sources under exact coverage."""


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
                f"--include={HARDWARE_HAL_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "ASYNC_HARDWARE_RUNNER_SOURCE",
    "ASYNC_HARDWARE_RUNNER_TEST",
    "HARDWARE_HAL_COVERAGE_COHORT",
    "HARDWARE_HAL_COVERAGE_DATA_FILE",
    "HARDWARE_HAL_COVERAGE_INCLUDE",
    "HARDWARE_HAL_DOCSTRING_RATCHET",
    "HARDWARE_CIRCUIT_CUTTING_SOURCE",
    "HARDWARE_CIRCUIT_CUTTING_TEST",
    "HARDWARE_CIRCUIT_EXPORT_SOURCE",
    "HARDWARE_CIRCUIT_EXPORT_TEST",
    "HARDWARE_HAL_SOURCE",
    "HARDWARE_HAL_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
