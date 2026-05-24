# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Zne
"""Zero-Noise Extrapolation via global unitary folding.

Reference: Giurgica-Tiron et al., "Digital zero noise extrapolation for
quantum error mitigation", IEEE QCE 2020.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Any

import numpy as np
from qiskit import QuantumCircuit


@dataclass
class ZNEResult:
    """Richardson extrapolation result: scales, raw values, and zero-noise estimate."""

    noise_scales: list[int]
    expectation_values: list[float]
    zero_noise_estimate: float
    fit_residual: float


def gate_fold_circuit(circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Global unitary folding: G -> G (G^dag G)^((scale-1)/2).

    ``scale`` must be an odd positive integer. scale=1 returns the original
    circuit. Measurement gates are stripped before folding and re-appended.
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"scale must be odd positive integer, got {scale}")
    if scale == 1:
        return circuit.copy()

    data = list(circuit.data)
    measurement_removed = False
    while data and data[-1].operation.name in {"barrier", "measure"}:
        measurement_removed = measurement_removed or data[-1].operation.name == "measure"
        data.pop()

    base = QuantumCircuit(circuit.num_qubits)
    for instruction in data:
        if instruction.clbits:
            raise ValueError("cannot fold circuits with mid-circuit classical operations")
        qubits = [base.qubits[circuit.find_bit(qubit).index] for qubit in instruction.qubits]
        base._append(instruction.operation.copy(), qubits)

    folded = base.copy()
    n_folds = (scale - 1) // 2
    base_inv = base.inverse()
    for _ in range(n_folds):
        folded.compose(base_inv, inplace=True)
        folded.compose(base, inplace=True)

    if measurement_removed:
        folded.measure_all()

    return folded


def zne_extrapolate(
    noise_scales: list[int],
    expectation_values: list[float],
    order: int = 1,
) -> ZNEResult:
    """Richardson extrapolation to zero noise.

    ``order`` controls polynomial degree: 1=linear, 2=quadratic.
    """
    order = _validate_order(order)
    x = np.array(noise_scales, dtype=float)
    y = np.array(expectation_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("noise_scales and expectation_values must be one-dimensional")
    if len(x) != len(y):
        raise ValueError("noise_scales and expectation_values must have the same length")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("noise_scales and expectation_values must be finite")
    if any(scale < 1 or int(scale) != scale or int(scale) % 2 == 0 for scale in x):
        raise ValueError("noise_scales must be odd positive integers")
    if len(set(int(scale) for scale in x)) != len(x):
        raise ValueError("noise_scales must be distinct")
    if len(x) < order + 1:
        raise ValueError(f"Need >= {order + 1} data points for order-{order} fit, got {len(x)}")

    coeffs = np.polyfit(x, y, deg=min(order, len(x) - 1))
    poly = np.poly1d(coeffs)
    zero_est = float(poly(0.0))
    residual = float(np.sqrt(np.mean((poly(x) - y) ** 2)))

    return ZNEResult(
        noise_scales=list(noise_scales),
        expectation_values=list(expectation_values),
        zero_noise_estimate=zero_est,
        fit_residual=residual,
    )


def _validate_order(order: Any) -> int:
    if isinstance(order, bool):
        raise ValueError("order must be a non-negative integer")
    try:
        order_value = index(order)
    except TypeError as exc:
        raise ValueError("order must be a non-negative integer") from exc
    if order_value < 0:
        raise ValueError("order must be a non-negative integer")
    return int(order_value)
