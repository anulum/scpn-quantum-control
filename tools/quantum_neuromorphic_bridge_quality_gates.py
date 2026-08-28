# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-neuromorphic bridge quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
QUANTUM_NEUROMORPHIC_BRIDGE_SOURCE = "src/scpn_quantum_control/qsnn/quantum_neuromorphic_bridge.py"
QUANTUM_NEUROMORPHIC_BRIDGE_COVERAGE_COHORT = [
    "tests/test_quantum_neuromorphic_bridge.py",
    "tests/test_quantum_neuromorphic_bridge_branches.py",
    "tests/test_qsnn_topology_policy_wiring.py",
    "tests/test_topology_control_integration.py",
]
QUANTUM_NEUROMORPHIC_BRIDGE_TYPING_RATCHET = [
    QUANTUM_NEUROMORPHIC_BRIDGE_SOURCE,
    "tools/quantum_neuromorphic_bridge_quality_gates.py",
    "tests/test_quantum_neuromorphic_bridge_quality_gate.py",
]
QUANTUM_NEUROMORPHIC_BRIDGE_DOCSTRING_RATCHET = [
    QUANTUM_NEUROMORPHIC_BRIDGE_SOURCE,
    *QUANTUM_NEUROMORPHIC_BRIDGE_COVERAGE_COHORT,
    "tools/quantum_neuromorphic_bridge_quality_gates.py",
    "tests/test_quantum_neuromorphic_bridge_quality_gate.py",
]
QUANTUM_NEUROMORPHIC_BRIDGE_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-quantum-neuromorphic-bridge-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-quantum-neuromorphic-bridge-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *QUANTUM_NEUROMORPHIC_BRIDGE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D quantum-neuromorphic-bridge quality ratchet",
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
                *QUANTUM_NEUROMORPHIC_BRIDGE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline bridge execution and exact coverage gates."""
    return [
        (
            "quantum-neuromorphic-bridge focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={QUANTUM_NEUROMORPHIC_BRIDGE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *QUANTUM_NEUROMORPHIC_BRIDGE_COVERAGE_COHORT,
            ],
        ),
        (
            "quantum-neuromorphic-bridge exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={QUANTUM_NEUROMORPHIC_BRIDGE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/qsnn/quantum_neuromorphic_bridge.py",
            ],
        ),
    ]


__all__ = [
    "QUANTUM_NEUROMORPHIC_BRIDGE_COVERAGE_COHORT",
    "QUANTUM_NEUROMORPHIC_BRIDGE_COVERAGE_DATA_FILE",
    "QUANTUM_NEUROMORPHIC_BRIDGE_DOCSTRING_RATCHET",
    "QUANTUM_NEUROMORPHIC_BRIDGE_SOURCE",
    "QUANTUM_NEUROMORPHIC_BRIDGE_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
