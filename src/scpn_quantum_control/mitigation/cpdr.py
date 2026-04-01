# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Cpdr
"""Clifford Perturbation Data Regression (CPDR) error mitigation.

Constructs near-Clifford training circuits by replacing non-Clifford
rotation gates (RZ, RY, RX) with their nearest Clifford equivalent
plus small perturbations. Ideal values are computed via Clifford
simulation (polynomial time). Noisy values are measured on hardware.
A regression model trained on (ideal, noisy) pairs corrects the
target circuit's noisy output.

Outperforms ZNE on IBM Eagle/Heron for deep circuits. Unlike ZNE,
does not require circuit depth amplification.

Reference:
    Zhang et al., "Clifford Perturbation Data Regression for Quantum
    Error Mitigation", arXiv:2412.09518 (Dec 2024).

Combined with Z₂ symmetry verification (symmetry_verification.py),
this provides compound mitigation: CPDR corrects coherent + incoherent
errors, parity post-selection removes remaining symmetry-violating noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class CPDRResult:
    """CPDR mitigation output."""

    raw_value: float
    mitigated_value: float
    n_training_circuits: int
    regression_r_squared: float
    regression_slope: float
    regression_intercept: float


# Clifford angles: multiples of π/2
_CLIFFORD_ANGLES = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])


def _nearest_clifford_angle(theta: float) -> float:
    """Snap a rotation angle to the nearest Clifford angle (multiple of π/2)."""
    theta_mod = theta % (2 * np.pi)
    idx = int(np.argmin(np.abs(_CLIFFORD_ANGLES - theta_mod)))
    return float(_CLIFFORD_ANGLES[idx])


def generate_training_circuits(
    target_circuit: QuantumCircuit,
    n_training: int = 20,
    perturbation_scale: float = 0.1,
    seed: int = 42,
) -> list[QuantumCircuit]:
    """Generate near-Clifford training circuits from the target.

    For each non-Clifford rotation gate in the target:
    1. Snap to nearest Clifford angle
    2. Add a small random perturbation (Gaussian, σ = perturbation_scale)

    This creates circuits that are structurally similar to the target
    but close enough to Clifford for efficient ideal simulation.
    """
    rng = np.random.default_rng(seed)
    base = target_circuit.remove_final_measurements(inplace=False)

    rotation_gates = {"rz", "ry", "rx", "p", "u1", "rxx", "ryy", "rzz"}
    rotation_indices = []
    for i, instruction in enumerate(base.data):
        if instruction.operation.name in rotation_gates:
            rotation_indices.append(i)

    training_circuits = []
    for _t in range(n_training):
        qc = base.copy()
        data = list(qc.data)
        for idx in rotation_indices:
            inst = data[idx]
            params = list(inst.operation.params)
            new_params = []
            for p in params:
                cliff_angle = _nearest_clifford_angle(float(p))
                perturbed = cliff_angle + rng.normal(0, perturbation_scale)
                new_params.append(perturbed)
            inst.operation.params = new_params
            data[idx] = inst
        qc.data = data
        if target_circuit.num_clbits > 0:
            qc.measure_all()
        training_circuits.append(qc)

    return training_circuits


def compute_ideal_values(
    circuits: list[QuantumCircuit],
    observable_qubits: list[int] | None = None,
) -> list[float]:
    """Compute ideal (noiseless) expectation values via statevector simulation.

    For near-Clifford circuits this is efficient. Returns ⟨Z⟩ averaged
    over observable_qubits, or all qubits if not specified.
    """
    ideal_values = []
    for circuit in circuits:
        base = circuit.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(base)
        n = base.num_qubits
        qubits = observable_qubits or list(range(n))

        exp_z = 0.0
        probs = sv.probabilities()
        for bitval in range(2**n):
            p = probs[bitval]
            for q in qubits:
                bit = (bitval >> q) & 1
                exp_z += p * (1 - 2 * bit)
        exp_z /= len(qubits)
        ideal_values.append(float(exp_z))

    return ideal_values


def compute_noisy_values_from_counts(
    counts_list: list[dict[str, int]],
    n_qubits: int,
    observable_qubits: list[int] | None = None,
) -> list[float]:
    """Extract ⟨Z⟩ from measurement counts (hardware or noisy simulator).

    Each entry in counts_list corresponds to one training circuit.
    """
    qubits = observable_qubits or list(range(n_qubits))
    noisy_values = []

    for counts in counts_list:
        total = sum(counts.values())
        if total == 0:
            noisy_values.append(0.0)
            continue

        exp_z = 0.0
        for bitstring, count in counts.items():
            bits = bitstring.replace(" ", "")
            for q in qubits:
                if q < len(bits):
                    bit = int(bits[-(q + 1)])
                    exp_z += (1 - 2 * bit) * count
        exp_z /= total * len(qubits)
        noisy_values.append(float(exp_z))

    return noisy_values


def fit_regression(
    ideal_values: list[float],
    noisy_values: list[float],
) -> tuple[float, float, float]:
    """Fit linear regression: noisy = slope * ideal + intercept.

    Returns (slope, intercept, r_squared).
    """
    x = np.array(ideal_values)
    y = np.array(noisy_values)

    if len(x) < 2:
        return 1.0, 0.0, 0.0

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res: float = float(np.sum((y - y_pred) ** 2))
    ss_tot: float = float(np.sum((y - np.mean(y)) ** 2))
    r_sq = 1.0 - ss_res / max(ss_tot, 1e-15)

    return float(slope), float(intercept), float(r_sq)


def cpdr_mitigate(
    raw_noisy_value: float,
    ideal_training: list[float],
    noisy_training: list[float],
) -> CPDRResult:
    """Apply CPDR correction to a noisy measurement.

    Given the linear model noisy = slope * ideal + intercept,
    the corrected value is: ideal_est = (noisy - intercept) / slope.
    """
    slope, intercept, r_sq = fit_regression(ideal_training, noisy_training)

    if abs(slope) < 1e-12:
        mitigated = raw_noisy_value
    else:
        mitigated = (raw_noisy_value - intercept) / slope

    return CPDRResult(
        raw_value=raw_noisy_value,
        mitigated_value=float(mitigated),
        n_training_circuits=len(ideal_training),
        regression_r_squared=r_sq,
        regression_slope=slope,
        regression_intercept=intercept,
    )


def cpdr_full_pipeline(
    target_circuit: QuantumCircuit,
    target_counts: dict[str, int],
    run_on_backend,
    n_training: int = 20,
    perturbation_scale: float = 0.1,
    observable_qubits: list[int] | None = None,
    seed: int = 42,
) -> CPDRResult:
    """End-to-end CPDR: generate training circuits, run, regress, correct.

    Args:
        target_circuit: The circuit to mitigate.
        target_counts: Noisy measurement counts from the target circuit.
        run_on_backend: Callable(list[QuantumCircuit]) -> list[dict[str,int]]
            that runs circuits on the noisy backend and returns counts.
        n_training: Number of near-Clifford training circuits.
        perturbation_scale: σ for Gaussian angle perturbation.
        observable_qubits: Which qubits to measure ⟨Z⟩ on.
        seed: RNG seed for reproducibility.
    """
    n_qubits = target_circuit.num_qubits

    # Generate and simulate training circuits
    training_circuits = generate_training_circuits(
        target_circuit, n_training, perturbation_scale, seed
    )
    ideal_values = compute_ideal_values(training_circuits, observable_qubits)

    # Run training circuits on noisy backend
    training_counts = run_on_backend(training_circuits)
    noisy_values = compute_noisy_values_from_counts(training_counts, n_qubits, observable_qubits)

    # Extract raw target value
    raw_value = compute_noisy_values_from_counts([target_counts], n_qubits, observable_qubits)[0]

    return cpdr_mitigate(raw_value, ideal_values, noisy_values)
