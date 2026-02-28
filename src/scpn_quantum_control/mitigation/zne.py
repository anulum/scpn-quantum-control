"""Zero-Noise Extrapolation via global unitary folding.

Reference: Giurgica-Tiron et al., "Digital zero noise extrapolation for
quantum error mitigation", IEEE QCE 2020.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    base = circuit.remove_final_measurements(inplace=False)

    folded = base.copy()
    n_folds = (scale - 1) // 2
    base_inv = base.inverse()
    for _ in range(n_folds):
        folded.compose(base_inv, inplace=True)
        folded.compose(base, inplace=True)

    if circuit.num_clbits > 0:
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
    x = np.array(noise_scales, dtype=float)
    y = np.array(expectation_values, dtype=float)

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
