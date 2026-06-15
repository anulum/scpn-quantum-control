# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Differentiable API workflow demo."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control import differentiable_compile_report, explain_differentiability
from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
    train_parameter_shift_qnn_classifier,
    verify_parameter_shift_qnn_classifier_gradient,
)


def main() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    params = np.array([0.4], dtype=float)

    value = execute_phase_qnode_circuit(circuit, params)
    gradient = parameter_shift_phase_qnode_gradient(circuit, params)
    print("minimal qnode")
    print(f"  value: {value.value:.6f}")
    print(f"  gradient: {gradient.gradient.tolist()}")
    print(f"  support: {gradient.support_report.supported}")
    print(f"  unsupported gates: {list(gradient.support_report.unsupported_gates)}")

    diagnostic = explain_differentiability(
        gate="arbitrary_unitary",
        observable="pauli_expectation",
        backend="hardware",
        shots=1024,
    )
    print("\nwhy this blocked route cannot differentiate")
    print(f"  supported: {diagnostic.supported}")
    print(f"  blocked reasons: {list(diagnostic.blocked_reasons)}")
    print(f"  alternatives: {list(diagnostic.suggested_alternatives)}")
    print(f"  framework rows: {[row['framework'] for row in diagnostic.dependency_matrix]}")

    compile_report = differentiable_compile_report(
        primitive_identities=("scpn.program_ad.array:getitem@1",)
    )
    print("\ncompiler report")
    print(f"  primitive count: {compile_report.payload['primitive_count']}")
    print(f"  method: {compile_report.method}")

    training = train_parameter_shift_qnn_classifier(
        np.array([[0.0], [np.pi]], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        initial_params=np.array([0.8], dtype=float),
        learning_rate=0.7,
        max_steps=80,
        target_loss=0.0,
        target_loss_tolerance=1.0e-4,
    )
    verification = verify_parameter_shift_qnn_classifier_gradient(
        np.array([[0.0], [np.pi]], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        training.best_params,
    )
    print("\nbounded qnn training")
    print(f"  best loss: {training.training.best_value:.8f}")
    print(f"  accuracy: {training.prediction.accuracy:.3f}")
    print(f"  gradient check: {verification.passed}")
    print(f"  method: {training.method}")


if __name__ == "__main__":
    main()
