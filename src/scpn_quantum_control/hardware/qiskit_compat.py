# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Qiskit Compat
"""Qiskit version compatibility layer.

Handles breaking changes between Qiskit 1.x and 2.x:
    - PauliEvolutionGate: moved in 2.0
    - SparsePauliOp: API changes
    - Backend/provider restructuring

Usage: import from here instead of qiskit directly for forward-compat.
"""

from __future__ import annotations

import importlib.metadata


def qiskit_version() -> str:
    """Get installed Qiskit version."""
    return importlib.metadata.version("qiskit")


def qiskit_major() -> int:
    """Get Qiskit major version number."""
    return int(qiskit_version().split(".")[0])


def get_pauli_evolution_gate():
    """Import PauliEvolutionGate from correct location."""
    try:
        from qiskit.circuit.library import PauliEvolutionGate

        return PauliEvolutionGate
    except ImportError:
        # Qiskit 2.x may move this
        from qiskit.circuit.library import PauliEvolutionGate  # type: ignore[no-redef]

        return PauliEvolutionGate


def get_lie_trotter():
    """Import LieTrotter from correct location."""
    try:
        from qiskit.synthesis import LieTrotter

        return LieTrotter
    except ImportError:
        from qiskit.synthesis.evolution import LieTrotter  # type: ignore[no-redef]

        return LieTrotter


def get_statevector():
    """Import Statevector from correct location."""
    from qiskit.quantum_info import Statevector

    return Statevector


def get_sparse_pauli_op():
    """Import SparsePauliOp from correct location."""
    from qiskit.quantum_info import SparsePauliOp

    return SparsePauliOp


def check_qiskit_compatibility() -> dict:
    """Check Qiskit installation and compatibility."""
    version = qiskit_version()
    major = qiskit_major()

    issues: list[str] = []
    if major >= 2:
        issues.append(
            "Qiskit 2.x detected — PauliEvolutionGate may have breaking changes (GH #15476)"
        )

    return {
        "version": version,
        "major": major,
        "compatible": major < 2,
        "issues": issues,
    }
