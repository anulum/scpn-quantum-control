# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Snn Backward
"""SNN backward pass: Ry parameter-shift gradient through the quantum layer.

The forward pass (snn_adapter.py) maps spike trains → quantum rotation
angles → quantum evolution → measurement → SNN currents.

The backward pass computes dL/dtheta (gradient of a loss function with respect
to input Ry angles) via the exact Ry parameter-shift rule:

    dy/dtheta_k = [y(theta_k + pi/2) - y(theta_k - pi/2)] / 2

For the SNN-quantum hybrid:
    1. SNN forward -> spike rates -> theta (Ry rotation angles)
    2. Quantum forward -> measurements -> y (output)
    3. Loss L(y, target)
    4. dL/dy (from loss)
    5. dy/dθ via parameter-shift (this module)
    6. dtheta/d(spike_rates) = pi (linear mapping)
    7. Chain: dL/d(spike_rates) = dL/dy * dy/dtheta * pi

This enables end-to-end training of the SNN-quantum hybrid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..qsnn.qlayer import QuantumDenseLayer


@dataclass
class BackwardResult:
    """SNN-quantum backward pass result."""

    grad_params: NDArray[np.float64]  # dL/dtheta for Ry input angles
    grad_spikes: NDArray[np.float64]  # dL/d(spike_rates) for the SNN bridge
    loss: float
    n_evaluations: int  # two shifted quantum evaluations per input angle


def _quantum_forward_angles(
    layer: QuantumDenseLayer,
    angles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Run the quantum layer for raw Ry angles and return P(|1>) probabilities."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector as SV

    qc = QuantumCircuit(layer.n_qubits)
    for i, angle in enumerate(angles):
        qc.ry(float(angle), i)
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
    result: NDArray[np.float64] = probs
    return result


def _quantum_forward(
    layer: QuantumDenseLayer,
    input_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Run clamped value-domain quantum forward and return probabilities.

    This is the ordinary bridge forward path: SNN spike-rate values are clamped
    to ``[0, 1]`` and converted to Ry angles. Parameter-shift evaluations use
    :func:`_quantum_forward_angles` so shifted angles are not clamped.
    """
    angles = np.pi * np.clip(input_values, 0.0, 1.0)
    return _quantum_forward_angles(layer, angles.astype(np.float64, copy=False))


def _mse_loss(y: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    """Mean squared error loss."""
    return float(np.mean((y - target) ** 2))


def _mse_grad(y: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gradient of MSE loss w.r.t. y."""
    result: NDArray[np.float64] = 2.0 * (y - target) / len(y)
    return result


def parameter_shift_gradient(
    layer: QuantumDenseLayer,
    input_values: NDArray[np.float64],
    target: NDArray[np.float64],
    shift: float = np.pi / 2.0,
) -> BackwardResult:
    """Compute SNN bridge gradients via the exact Ry parameter-shift rule.

    Args:
        layer: quantum dense layer
        input_values: input values in [0, 1] from SNN spike rates
        target: target output for MSE loss
        shift: angle-domain shift in radians; ``pi / 2`` is the exact Ry rule
    """
    y_forward = _quantum_forward(layer, input_values)
    loss = _mse_loss(y_forward, target)
    dl_dy = _mse_grad(y_forward, target)

    n_params = len(input_values)
    grad_params = np.zeros(n_params)
    n_evals = 0
    base_angles = np.pi * np.clip(input_values, 0.0, 1.0)
    coefficient = 0.0 if abs(np.sin(shift)) <= 1e-12 else 1.0 / (2.0 * np.sin(shift))

    for k in range(n_params):
        angles_plus = base_angles.copy()
        angles_plus[k] += shift
        angles_minus = base_angles.copy()
        angles_minus[k] -= shift

        y_plus = _quantum_forward_angles(layer, angles_plus.astype(np.float64, copy=False))
        y_minus = _quantum_forward_angles(layer, angles_minus.astype(np.float64, copy=False))

        dy_dtheta = coefficient * (y_plus - y_minus)
        grad_params[k] = float(np.dot(dl_dy, dy_dtheta))
        n_evals += 2

    # input_values are spike rates and theta = pi * spike_rate.
    grad_spikes = grad_params * np.pi

    return BackwardResult(
        grad_params=grad_params,
        grad_spikes=grad_spikes,
        loss=loss,
        n_evaluations=n_evals,
    )
