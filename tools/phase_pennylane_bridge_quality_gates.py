# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — focused Phase-QNode PennyLane quality-gate specification
"""Build static and exact-coverage gates for the Phase-QNode PennyLane bridge."""

from __future__ import annotations

from os import devnull
from tempfile import gettempdir

Gate = tuple[str, list[str]]

PHASE_PENNYLANE_BRIDGE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/pennylane_bridge.py",
    "tests/_phase_pennylane_bridge_test_helpers.py",
    "tests/test_phase_pennylane_bridge.py",
    "tests/test_phase_pennylane_bridge_validation_edges.py",
    "tools/phase_pennylane_bridge_quality_gates.py",
    "tests/test_phase_pennylane_bridge_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring owner cohort."""

PHASE_PENNYLANE_BRIDGE_COVERAGE_COHORT = [
    "tests/test_phase_pennylane_bridge.py",
    "tests/test_phase_pennylane_bridge_validation_edges.py",
    "tests/test_phase_pennylane_import.py",
    "tests/test_phase_pennylane_provider_plugin.py",
    "tests/test_phase_pennylane_provider_plugin_integration.py",
]
"""Real bridge, import, provider and maturity tests owning exact coverage."""

PHASE_PENNYLANE_BRIDGE_COVERAGE_DATA_FILE = (
    f"{gettempdir()}/scpn-qc-phase-pennylane-bridge.coverage"
)
"""Isolated coverage database for the Phase-QNode PennyLane bridge."""

PHASE_PENNYLANE_BRIDGE_COVERAGE_INCLUDE = "*/pennylane_bridge.py"
"""Production bridge source enforced at exact branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Python interpreter path admitted by the caller.

    Returns
    -------
    list[Gate]
        Ordered static gates for the PennyLane bridge owner.

    """
    return [
        (
            "mypy-strict-phase-pennylane-bridge",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PHASE_PENNYLANE_BRIDGE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D phase-pennylane-bridge quality ratchet",
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
                *PHASE_PENNYLANE_BRIDGE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real PennyLane execution and exact bridge-coverage gates.

    Parameters
    ----------
    python
        Python interpreter path admitted by the caller.

    Returns
    -------
    list[Gate]
        Focused execution followed by exact source-only coverage.

    """
    return [
        (
            "phase-pennylane-bridge focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_PENNYLANE_BRIDGE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PHASE_PENNYLANE_BRIDGE_COVERAGE_COHORT,
            ],
        ),
        (
            "phase-pennylane-bridge exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_PENNYLANE_BRIDGE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--show-missing",
                "--fail-under=100",
                f"--include={PHASE_PENNYLANE_BRIDGE_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "PHASE_PENNYLANE_BRIDGE_COVERAGE_COHORT",
    "PHASE_PENNYLANE_BRIDGE_COVERAGE_DATA_FILE",
    "PHASE_PENNYLANE_BRIDGE_COVERAGE_INCLUDE",
    "PHASE_PENNYLANE_BRIDGE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
