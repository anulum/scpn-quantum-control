# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — gradient-backend planner quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
GRADIENT_BACKEND_SOURCE = "src/scpn_quantum_control/phase/gradient_backend.py"
PROVIDER_GRADIENT_SOURCE = "src/scpn_quantum_control/phase/provider_gradient.py"
QISKIT_RUNTIME_SOURCE = "src/scpn_quantum_control/phase/qiskit_runtime.py"
QISKIT_BRIDGE_CONTRACTS_SOURCE = "src/scpn_quantum_control/phase/qiskit_bridge_contracts.py"
GRADIENT_BACKEND_SOURCES = [
    GRADIENT_BACKEND_SOURCE,
    PROVIDER_GRADIENT_SOURCE,
    QISKIT_RUNTIME_SOURCE,
    QISKIT_BRIDGE_CONTRACTS_SOURCE,
]
GRADIENT_BACKEND_COVERAGE_COHORT = [
    "tests/test_phase_gradient_backend.py",
    "tests/test_phase_provider_gradient.py",
    "tests/test_phase_provider_gradient_branches.py",
    "tests/test_phase_qiskit_bridge_contracts.py",
    "tests/test_phase_qiskit_bridge_contract_edges.py",
    "tests/test_phase_qiskit_gradients.py",
    "tests/test_phase_qiskit_runtime.py",
]
GRADIENT_BACKEND_TYPING_RATCHET = [
    *GRADIENT_BACKEND_SOURCES,
    "tests/test_phase_provider_gradient.py",
    "tests/test_phase_provider_gradient_branches.py",
    "tests/test_phase_qiskit_bridge_contracts.py",
    "tests/test_phase_qiskit_bridge_contract_edges.py",
    "tools/gradient_backend_quality_gates.py",
    "tests/test_gradient_backend_quality_gate.py",
]
GRADIENT_BACKEND_DOCSTRING_RATCHET = [
    *GRADIENT_BACKEND_SOURCES,
    "tests/test_phase_gradient_backend.py",
    "tests/test_phase_provider_gradient.py",
    "tests/test_phase_provider_gradient_branches.py",
    "tests/test_phase_qiskit_bridge_contracts.py",
    "tests/test_phase_qiskit_bridge_contract_edges.py",
    "tools/gradient_backend_quality_gates.py",
    "tests/test_gradient_backend_quality_gate.py",
]
GRADIENT_BACKEND_COVERAGE_DATA_FILE = "/tmp/scpn-qc-gradient-backend-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-gradient-backend-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *GRADIENT_BACKEND_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D gradient-backend quality ratchet",
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
                'lint.pydocstyle.convention = "numpy"',
                *GRADIENT_BACKEND_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build planner execution and exact coverage gates."""
    return [
        (
            "gradient-backend focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={GRADIENT_BACKEND_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *GRADIENT_BACKEND_COVERAGE_COHORT,
            ],
        ),
        (
            "gradient-backend exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={GRADIENT_BACKEND_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/gradient_backend.py,*/phase/provider_gradient.py,"
                "*/phase/qiskit_runtime.py,*/phase/qiskit_bridge_contracts.py",
            ],
        ),
    ]


__all__ = [
    "GRADIENT_BACKEND_COVERAGE_COHORT",
    "GRADIENT_BACKEND_COVERAGE_DATA_FILE",
    "GRADIENT_BACKEND_DOCSTRING_RATCHET",
    "GRADIENT_BACKEND_SOURCE",
    "GRADIENT_BACKEND_SOURCES",
    "GRADIENT_BACKEND_TYPING_RATCHET",
    "PROVIDER_GRADIENT_SOURCE",
    "QISKIT_BRIDGE_CONTRACTS_SOURCE",
    "QISKIT_RUNTIME_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
