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
    """Amplify noise by global unitary folding: ``G -> G (G^dag G)^((scale-1)/2)``.

    The trailing measurement and barrier instructions are detached before
    folding and re-attached afterwards **at their original qubit and clbit
    positions**, so a partial, permuted or multi-register readout keeps the
    observable it started with. The returned circuit reuses the input's
    registers, bits, name, metadata and global phase; only the unitary body is
    repeated.

    Parameters
    ----------
    circuit
        Circuit whose unitary body is folded. Classical operations may appear
        only in the trailing measurement/barrier block; a mid-circuit classical
        operation is rejected rather than silently dropped.
    scale
        Odd positive noise-scale factor. ``scale=1`` returns an unchanged copy;
        ``scale=2k+1`` appends ``k`` inverse-forward pairs.

    Returns
    -------
    qiskit.QuantumCircuit
        Folded circuit with the same measurement mapping as ``circuit``.

    Raises
    ------
    ValueError
        If ``scale`` is not an odd positive integer, or if the circuit carries a
        classical operation outside its trailing measurement block.

    References
    ----------
    Giurgica-Tiron et al., "Digital zero noise extrapolation for quantum error
    mitigation", IEEE QCE 2020.

    """
    scale = _validate_scale(scale)
    if scale == 1:
        return circuit.copy()

    data = list(circuit.data)
    trailing: list[Any] = []
    while data and data[-1].operation.name in {"barrier", "measure"}:
        trailing.append(data.pop())
    trailing.reverse()

    base = QuantumCircuit(circuit.num_qubits)
    for instruction in data:
        if instruction.clbits:
            raise ValueError("cannot fold circuits with mid-circuit classical operations")
        qubits = [base.qubits[circuit.find_bit(qubit).index] for qubit in instruction.qubits]
        base._append(instruction.operation.copy(), qubits)

    body = base.copy()
    n_folds = (scale - 1) // 2
    base_inv = base.inverse()
    for _ in range(n_folds):
        body.compose(base_inv, inplace=True)
        body.compose(base, inplace=True)

    folded = circuit.copy_empty_like()
    folded.compose(body, qubits=folded.qubits[: circuit.num_qubits], inplace=True)
    for instruction in trailing:
        folded.append(
            instruction.operation.copy(),
            [folded.qubits[circuit.find_bit(qubit).index] for qubit in instruction.qubits],
            [folded.clbits[circuit.find_bit(clbit).index] for clbit in instruction.clbits],
        )

    return folded


def zne_extrapolate(
    noise_scales: list[int],
    expectation_values: list[float],
    order: int = 1,
) -> ZNEResult:
    """Richardson extrapolation to zero noise.

    ``order`` controls polynomial degree: 1=linear, 2=quadratic.
    Scales must be distinct odd positive integers; booleans, floats and strings
    are rejected with ValueError without coercion, as in gate_fold_circuit.
    """
    order = _validate_order(order)
    raw_scales = np.asarray(noise_scales, dtype=object)
    y = np.array(expectation_values, dtype=float)
    if raw_scales.ndim != 1 or y.ndim != 1:
        raise ValueError("noise_scales and expectation_values must be one-dimensional")
    if len(raw_scales) != len(y):
        raise ValueError("noise_scales and expectation_values must have the same length")
    validated_scales = [_validate_scale(scale) for scale in raw_scales]
    x = np.array(validated_scales, dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("noise_scales and expectation_values must be finite")
    if len(set(validated_scales)) != len(x):
        raise ValueError("noise_scales must be distinct")
    if len(x) < order + 1:
        raise ValueError(f"Need >= {order + 1} data points for order-{order} fit, got {len(x)}")

    coeffs = np.polyfit(x, y, deg=min(order, len(x) - 1))
    poly = np.poly1d(coeffs)
    zero_est = float(poly(0.0))
    residual = float(np.sqrt(np.mean((poly(x) - y) ** 2)))

    return ZNEResult(
        noise_scales=validated_scales,
        expectation_values=list(expectation_values),
        zero_noise_estimate=zero_est,
        fit_residual=residual,
    )


def _validate_scale(value: object) -> int:
    """Return an odd positive integer without accepting bool or lossy casts."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("scale must be odd positive integer, excluding booleans")
    scale = int(value)
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"scale must be odd positive integer, got {scale}")
    return scale


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
