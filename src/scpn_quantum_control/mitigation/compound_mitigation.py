# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Compound Mitigation
"""Compound Error Mitigation: CPDR + Z2 Symmetry Verification.

Combines Clifford Perturbation Data Regression (CPDR) with Z2 Symmetry
Verification for an order-of-magnitude reduction in noise.

Symmetry Verification projects out states that violate the known parity
of the system (e.g., total Z parity in XY Hamiltonians). Applying this
to the CPDR training circuits reduces the noise floor variance before
the regression model is fit, leading to vastly improved mitigation accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit

from scpn_quantum_control.mitigation.cpdr import (
    compute_ideal_values,
    compute_noisy_values_from_counts,
    cpdr_mitigate,
    generate_training_circuits,
)
from scpn_quantum_control.mitigation.symmetry_verification import parity_postselect


@dataclass
class CompoundMitigationResult:
    """Result from compound mitigation."""

    raw_value: float
    mitigated_value: float
    n_training_circuits: int
    regression_r_squared: float
    regression_slope: float
    regression_intercept: float
    mean_rejection_rate: float


def compound_mitigate_pipeline(
    target_circuit: QuantumCircuit,
    target_counts: dict[str, int],
    run_on_backend,
    expected_parity: int,
    n_training: int = 20,
    perturbation_scale: float = 0.1,
    observable_qubits: list[int] | None = None,
    seed: int = 42,
) -> CompoundMitigationResult:
    """End-to-end CPDR with Z2 symmetry verification.

    Args:
        target_circuit: The circuit to mitigate.
        target_counts: Noisy measurement counts from the target circuit.
        run_on_backend: Callable(list[QuantumCircuit]) -> list[dict[str,int]]
            that runs circuits on the noisy backend and returns counts.
        expected_parity: 0 for even, 1 for odd.
        n_training: Number of near-Clifford training circuits.
        perturbation_scale: σ for Gaussian angle perturbation.
        observable_qubits: Which qubits to measure ⟨Z⟩ on.
        seed: RNG seed for reproducibility.
    """
    n_qubits = target_circuit.num_qubits

    training_circuits = generate_training_circuits(
        target_circuit, n_training, perturbation_scale, seed
    )
    ideal_values = compute_ideal_values(training_circuits, observable_qubits)

    training_counts = run_on_backend(training_circuits)

    verified_training_counts = []
    rejection_rates = []
    for counts in training_counts:
        sym_res = parity_postselect(counts, expected_parity=expected_parity)
        verified_training_counts.append(sym_res.verified_counts)
        rejection_rates.append(sym_res.rejection_rate)

    mean_rejection = sum(rejection_rates) / max(len(rejection_rates), 1)

    noisy_values = compute_noisy_values_from_counts(
        verified_training_counts, n_qubits, observable_qubits
    )

    sym_target = parity_postselect(target_counts, expected_parity=expected_parity)
    raw_verified_value = compute_noisy_values_from_counts(
        [sym_target.verified_counts], n_qubits, observable_qubits
    )[0]

    cpdr_res = cpdr_mitigate(raw_verified_value, ideal_values, noisy_values)

    return CompoundMitigationResult(
        raw_value=raw_verified_value,
        mitigated_value=cpdr_res.mitigated_value,
        n_training_circuits=cpdr_res.n_training_circuits,
        regression_r_squared=cpdr_res.regression_r_squared,
        regression_slope=cpdr_res.regression_slope,
        regression_intercept=cpdr_res.regression_intercept,
        mean_rejection_rate=mean_rejection,
    )
