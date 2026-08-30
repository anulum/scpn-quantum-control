# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — focused Phase-QNode Torch quality-gate specification
"""Build static and exact-coverage gates for the Phase-QNode Torch facade."""

from __future__ import annotations

from os import devnull
from tempfile import gettempdir

Gate = tuple[str, list[str]]

PHASE_TORCH_BRIDGE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/torch_bridge.py",
    "src/scpn_quantum_control/phase/torch_maturity.py",
    "tools/phase_torch_bridge_quality_gates.py",
    "tests/test_phase_torch_bridge_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring owner cohort."""

PHASE_TORCH_BRIDGE_COVERAGE_COHORT = [
    "tests/test_phase_torch_bridge.py",
    "tests/test_phase_torch_bridge_contracts.py",
    "tests/test_phase_torch_bridge_runtime_boundaries.py",
    "tests/test_phase_torch_bridge_validation_edges.py",
    "tests/test_phase_torch_compatibility.py",
    "tests/test_phase_torch_compatibility_integration.py",
    "tests/test_phase_torch_gradients.py",
    "tests/test_phase_torch_gradients_integration.py",
    "tests/test_phase_torch_maturity.py",
    "tests/test_phase_torch_maturity_integration.py",
    "tests/test_phase_torch_qnode_transforms.py",
    "tests/test_phase_torch_qnode_transforms_integration.py",
]
"""Real facade and leaf tests that own exact Torch bridge coverage."""

PHASE_TORCH_BRIDGE_COVERAGE_DATA_FILE = f"{gettempdir()}/scpn-qc-phase-torch-bridge.coverage"
"""Isolated coverage database for the Phase-QNode Torch facade."""

PHASE_TORCH_BRIDGE_COVERAGE_INCLUDE = "*/torch_bridge.py,*/torch_maturity.py"
"""Production source enforced at exact branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered static gates for the Torch facade owner.

    """
    return [
        (
            "mypy-strict-phase-torch-bridge",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PHASE_TORCH_BRIDGE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D phase-torch-bridge quality ratchet",
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
                *PHASE_TORCH_BRIDGE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real Torch execution and exact facade-coverage gates.

    Parameters
    ----------
    python
        Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Focused execution followed by exact source-only coverage.

    """
    return [
        (
            "phase-torch-bridge focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_TORCH_BRIDGE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PHASE_TORCH_BRIDGE_COVERAGE_COHORT,
            ],
        ),
        (
            "phase-torch-bridge exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_TORCH_BRIDGE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={PHASE_TORCH_BRIDGE_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "PHASE_TORCH_BRIDGE_COVERAGE_COHORT",
    "PHASE_TORCH_BRIDGE_COVERAGE_DATA_FILE",
    "PHASE_TORCH_BRIDGE_COVERAGE_INCLUDE",
    "PHASE_TORCH_BRIDGE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
