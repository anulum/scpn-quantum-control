# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — focused Phase-QNode JAX quality-gate specification
"""Build local static and exact-coverage gates for the Phase-QNode JAX owner."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

PHASE_JAX_QNODE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/phase/jax_qnode_transforms.py",
    "src/scpn_quantum_control/phase/jax_compatibility.py",
    "tests/_phase_jax_bridge_test_helpers.py",
    "tests/_phase_jax_qnode_test_helpers.py",
    "tests/test_phase_jax_bridge_aot_export.py",
    "tests/test_phase_jax_qnode_transforms.py",
    "tests/test_phase_jax_qnode_transforms_integration.py",
    "tests/test_phase_jax_qnode_input_validation.py",
    "tests/test_phase_jax_qnode_pytree_validation.py",
    "tests/test_phase_jax_qnode_aot_validation.py",
    "tests/test_phase_jax_qnode_statevector_edges.py",
    "tools/phase_jax_qnode_quality_gates.py",
    "tests/test_phase_jax_qnode_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring owner cohort."""

PHASE_JAX_QNODE_COVERAGE_COHORT = [
    "tests/test_phase_jax_qnode_transforms.py",
    "tests/test_phase_jax_qnode_transforms_integration.py",
    "tests/test_phase_jax_bridge_aot_export.py",
    "tests/test_phase_jax_qnode_input_validation.py",
    "tests/test_phase_jax_qnode_pytree_validation.py",
    "tests/test_phase_jax_qnode_aot_validation.py",
    "tests/test_phase_jax_qnode_statevector_edges.py",
]
"""Public-path tests that own exact transform-leaf coverage."""

PHASE_JAX_QNODE_COVERAGE_DATA_FILE = ".coverage.phase-jax-qnode"
"""Isolated coverage database for the Phase-QNode JAX owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered static quality gates for the runtime and policy owners.

    """
    return [
        (
            "mypy-strict-phase-jax-qnode",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PHASE_JAX_QNODE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D phase-jax-qnode quality ratchet",
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
                *PHASE_JAX_QNODE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build isolated exact statement and branch coverage gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Coverage execution followed by its exact owner-only report.

    """
    return [
        (
            "phase-jax-qnode focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_JAX_QNODE_COVERAGE_DATA_FILE}",
                "--branch",
                "--source=src/scpn_quantum_control/phase",
                "-m",
                "pytest",
                "-q",
                *PHASE_JAX_QNODE_COVERAGE_COHORT,
            ],
        ),
        (
            "phase-jax-qnode exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_JAX_QNODE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/jax_qnode_transforms.py",
            ],
        ),
    ]


__all__ = [
    "PHASE_JAX_QNODE_COVERAGE_COHORT",
    "PHASE_JAX_QNODE_COVERAGE_DATA_FILE",
    "PHASE_JAX_QNODE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
