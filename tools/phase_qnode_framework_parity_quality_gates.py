# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QNode framework-parity quality gates
"""Build static and exact-coverage gates for local QNode framework parity."""

from __future__ import annotations

from os import devnull
from tempfile import gettempdir

Gate = tuple[str, list[str]]

PHASE_QNODE_FRAMEWORK_PARITY_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/qnode_framework_parity.py",
    "tests/test_phase_qnode_framework_parity.py",
    "tools/phase_qnode_framework_parity_quality_gates.py",
    "tests/test_phase_qnode_framework_parity_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring owner cohort."""

PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_COHORT = [
    "tests/test_phase_qnode_framework_parity.py",
]
"""Real local-framework and adapter-boundary coverage owner."""

PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_DATA_FILE = (
    f"{gettempdir()}/scpn-qc-phase-qnode-framework-parity.coverage"
)
"""Isolated coverage database for framework parity."""

PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_INCLUDE = "*/qnode_framework_parity.py"
"""Production source enforced at exact branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-documentation gates."""
    return [
        (
            "mypy-strict-phase-qnode-framework-parity",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PHASE_QNODE_FRAMEWORK_PARITY_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D phase-qnode-framework-parity quality ratchet",
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
                *PHASE_QNODE_FRAMEWORK_PARITY_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real local-framework execution and exact coverage gates."""
    return [
        (
            "phase-qnode-framework-parity focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_COHORT,
            ],
        ),
        (
            "phase-qnode-framework-parity exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_COHORT",
    "PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_DATA_FILE",
    "PHASE_QNODE_FRAMEWORK_PARITY_COVERAGE_INCLUDE",
    "PHASE_QNODE_FRAMEWORK_PARITY_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
