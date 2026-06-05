# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Compiler Lowering
"""Tests for compiler/mlir.py phase-QNode lowering reports."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.compiler.mlir import lower_phase_qnode_circuit_to_mlir
from scpn_quantum_control.phase.qnode_circuit import PauliTerm, PhaseQNodeCircuit


def test_phase_qnode_compiler_lowering_reports_registered_subset() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("ry", (0,), 0), ("cnot", (0, 1)), ("rzz", (0, 1), 1)),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )

    module = lower_phase_qnode_circuit_to_mlir(circuit, np.array([0.2, -0.3], dtype=float))

    assert module.dialect == "scpn_phase_qnode"
    assert module.metadata["supported"] is True
    assert module.metadata["primitive_support"]["gates"] == ["ry", "cnot", "rzz"]
    assert module.metadata["shape_limits"]["max_qubits"] >= 2
    assert module.metadata["rust_pyo3_parity"] == "blocked: no Rust phase-QNode lowering backend"
    assert "scpn_phase_qnode.ry" in module.text
    assert "scpn_phase_qnode.expectation" in module.text


def test_phase_qnode_compiler_lowering_fails_closed_for_unsupported_circuit() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("u3", (0,), 0),),
        observable="pauli_z",
    )

    with pytest.raises(ValueError, match="phase-QNode lowering failed closed"):
        lower_phase_qnode_circuit_to_mlir(circuit, np.array([0.2], dtype=float))
