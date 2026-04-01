# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Snn Backward
"""SNN backward pass: parameter-shift gradient through quantum layer.

The forward pass (snn_adapter.py) maps spike trains → quantum rotation
angles → quantum evolution → measurement → SNN currents.

The backward pass computes dL/dθ (gradient of a loss function w.r.t.
quantum circuit parameters) via the parameter-shift rule:

    dL/dθ_k = [L(θ_k + π/2) - L(θ_k - π/2)] / 2

For the SNN-quantum hybrid:
    1. SNN forward → spike rates → θ (rotation angles)
    2. Quantum forward → measurements → y (output)
    3. Loss L(y, target)
    4. dL/dy (from loss)
    5. dy/dθ via parameter-shift (this module)
    6. dθ/d(spike_rates) = π (linear mapping)
    7. Chain: dL/d(spike_rates) = dL/dy × dy/dθ × π

This enables end-to-end training of the SNN-quantum hybrid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..qsnn.qlayer import QuantumDenseLayer


@dataclass
class BackwardResult:
    """SNN-quantum backward pass result."""

    grad_params: np.ndarray  # dL/dθ for quantum circuit parameters
    grad_spikes: np.ndarray  # dL/d(spike_rates) for SNN
    loss: float
    n_evaluations: int  # 2 per parameter (parameter-shift)


def _quantum_forward(
    layer: QuantumDenseLayer,
    input_values: np.ndarray,
) -> np.ndarray:
    """Run quantum layer forward and return neuron P(|1>) probabilities.

    Rebuilds the circuit internally to get continuous probabilities
    (not thresholded spikes).
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector as SV

    qc = QuantumCircuit(layer.n_qubits)
    for i, val in enumerate(input_values):
        theta = np.pi * float(np.clip(val, 0.0, 1.0))
        qc.ry(theta, i)
    for n_idx in range(layer.n_neurons):
        neuron_qubit = layer.n_inputs + n_idx
        for i in range(layer.n_inputs):
            layer.synapses[n_idx][i].apply(qc, i, neuron_qubit)
    for n_idx in range(layer.n_neurons - 1):
        qc.cx(layer.n_inputs + n_idx, layer.n_inputs + n_idx + 1)

    sv = SV.from_instruction(qc)
    probs = np.zeros(layer.n_neurons)
    for n_idx in range(layer.n_neurons):
        marginal = sv.probabilities([layer.n_inputs + n_idx])
        probs[n_idx] = marginal[1]
    result: np.ndarray = probs
    return result


def _mse_loss(y: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error loss."""
    return float(np.mean((y - target) ** 2))


def _mse_grad(y: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gradient of MSE loss w.r.t. y."""
    result: np.ndarray = 2.0 * (y - target) / len(y)
    return result


def parameter_shift_gradient(
    layer: QuantumDenseLayer,
    input_values: np.ndarray,
    target: np.ndarray,
    shift: float = 0.25,
) -> BackwardResult:
    """Compute dL/d(input) via parameter-shift rule.

    Args:
        layer: quantum dense layer
        input_values: input values in [0, 1] from SNN spike rates
        target: target output for MSE loss
        shift: value-space shift (0.25 → π/4 angle shift for Ry)
    """
    y_forward = _quantum_forward(layer, input_values)
    loss = _mse_loss(y_forward, target)
    dl_dy = _mse_grad(y_forward, target)

    n_params = len(input_values)
    grad_params = np.zeros(n_params)
    n_evals = 0

    for k in range(n_params):
        vals_plus = input_values.copy()
        vals_plus[k] = min(vals_plus[k] + shift, 1.0)
        vals_minus = input_values.copy()
        vals_minus[k] = max(vals_minus[k] - shift, 0.0)

        y_plus = _quantum_forward(layer, vals_plus)
        y_minus = _quantum_forward(layer, vals_minus)

        actual_shift = vals_plus[k] - vals_minus[k]
        if actual_shift > 1e-10:
            dy_dv = (y_plus - y_minus) / actual_shift
        else:
            dy_dv = np.zeros_like(y_plus)

        grad_params[k] = float(np.dot(dl_dy, dy_dv))
        n_evals += 2

    # Gradient w.r.t. spike rates: input_values = spike_rates directly
    grad_spikes = grad_params * np.pi

    return BackwardResult(
        grad_params=grad_params,
        grad_spikes=grad_spikes,
        loss=loss,
        n_evaluations=n_evals,
    )
