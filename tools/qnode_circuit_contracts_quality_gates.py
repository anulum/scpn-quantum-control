# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QNode circuit-contract quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

QNODE_CIRCUIT_CONTRACTS_SOURCE = "src/scpn_quantum_control/phase/qnode_circuit_contracts.py"
"""Production source owned by the QNode circuit contracts."""

QNODE_CIRCUIT_CONTRACTS_COVERAGE_COHORT = [
    "tests/test_phase_qnode_circuit.py",
    "tests/test_phase_qnode_circuit_builders.py",
    "tests/test_phase_qnode_circuit_builders_integration.py",
    "tests/test_phase_qnode_circuit_contracts.py",
    "tests/test_phase_qnode_circuit_differentiation.py",
    "tests/test_phase_qnode_circuit_differentiation_integration.py",
    "tests/test_phase_qnode_circuit_execution.py",
    "tests/test_phase_qnode_circuit_execution_integration.py",
    "tests/test_phase_qnode_circuit_support.py",
    "tests/test_phase_qnode_circuit_support_integration.py",
]
"""Real tests that own exact QNode circuit-contract coverage."""

QNODE_CIRCUIT_CONTRACTS_TYPING_RATCHET = [
    QNODE_CIRCUIT_CONTRACTS_SOURCE,
    "tools/qnode_circuit_contracts_quality_gates.py",
    "tests/test_qnode_circuit_contracts_quality_gate.py",
]
"""Strict-typing cohort for production and gate contracts."""

QNODE_CIRCUIT_CONTRACTS_DOCSTRING_RATCHET = [
    QNODE_CIRCUIT_CONTRACTS_SOURCE,
    "tests/test_phase_qnode_circuit_contracts.py",
    "tools/qnode_circuit_contracts_quality_gates.py",
    "tests/test_qnode_circuit_contracts_quality_gate.py",
]
"""Complete contract-leaf and gate-contract docstring cohort."""

QNODE_CIRCUIT_CONTRACTS_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-qnode-circuit-contracts-quality.coverage"  # nosec B108
)
"""Isolated coverage database for the QNode circuit-contract owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-qnode-circuit-contracts-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *QNODE_CIRCUIT_CONTRACTS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D qnode-circuit-contracts quality ratchet",
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
                *QNODE_CIRCUIT_CONTRACTS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    return [
        (
            "qnode-circuit-contracts focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={QNODE_CIRCUIT_CONTRACTS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *QNODE_CIRCUIT_CONTRACTS_COVERAGE_COHORT,
            ],
        ),
        (
            "qnode-circuit-contracts exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={QNODE_CIRCUIT_CONTRACTS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/qnode_circuit_contracts.py",
            ],
        ),
    ]


__all__ = [
    "QNODE_CIRCUIT_CONTRACTS_COVERAGE_COHORT",
    "QNODE_CIRCUIT_CONTRACTS_COVERAGE_DATA_FILE",
    "QNODE_CIRCUIT_CONTRACTS_DOCSTRING_RATCHET",
    "QNODE_CIRCUIT_CONTRACTS_SOURCE",
    "QNODE_CIRCUIT_CONTRACTS_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
