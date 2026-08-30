# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-QMI compatibility quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
QUANTUM_PHI_SOURCE = "src/scpn_quantum_control/analysis/quantum_phi.py"
INTEGRATED_INFORMATION_PHI_SOURCE = (
    "src/scpn_quantum_control/analysis/integrated_information_phi.py"
)
QUANTUM_PHI_COVERAGE_SOURCES = [
    QUANTUM_PHI_SOURCE,
    INTEGRATED_INFORMATION_PHI_SOURCE,
]
QUANTUM_PHI_COVERAGE_INCLUDE = "*/analysis/quantum_phi.py,*/analysis/integrated_information_phi.py"
QUANTUM_PHI_COVERAGE_COHORT = [
    "tests/test_quantum_phi.py",
    "tests/test_analysis_topology_contracts.py",
    "tests/test_observables.py",
]
QUANTUM_PHI_TYPING_RATCHET = [
    QUANTUM_PHI_SOURCE,
    INTEGRATED_INFORMATION_PHI_SOURCE,
    "tests/test_observables.py",
    "tools/quantum_phi_quality_gates.py",
    "tests/test_quantum_phi_quality_gate.py",
]
QUANTUM_PHI_DOCSTRING_RATCHET = [
    QUANTUM_PHI_SOURCE,
    INTEGRATED_INFORMATION_PHI_SOURCE,
    "tests/test_quantum_phi.py",
    "tests/test_observables.py",
    "tools/quantum_phi_quality_gates.py",
    "tests/test_quantum_phi_quality_gate.py",
]
QUANTUM_PHI_COVERAGE_DATA_FILE = "/tmp/scpn-qc-quantum-phi-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-quantum-phi-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *QUANTUM_PHI_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D quantum-Phi quality ratchet",
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
                *QUANTUM_PHI_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline QMI execution and exact coverage gates."""
    return [
        (
            "quantum-Phi focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={QUANTUM_PHI_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *QUANTUM_PHI_COVERAGE_COHORT,
            ],
        ),
        (
            "quantum-Phi exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={QUANTUM_PHI_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={QUANTUM_PHI_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "INTEGRATED_INFORMATION_PHI_SOURCE",
    "QUANTUM_PHI_COVERAGE_COHORT",
    "QUANTUM_PHI_COVERAGE_DATA_FILE",
    "QUANTUM_PHI_COVERAGE_INCLUDE",
    "QUANTUM_PHI_COVERAGE_SOURCES",
    "QUANTUM_PHI_DOCSTRING_RATCHET",
    "QUANTUM_PHI_SOURCE",
    "QUANTUM_PHI_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
