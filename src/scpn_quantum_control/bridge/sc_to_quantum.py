"""Bitstream probability <-> quantum rotation angle converters.

Maps stochastic computing probabilities to qubit rotation angles and back.
The core identity: P(|1>) = sin^2(theta/2) for Ry(theta)|0>.
"""

from __future__ import annotations

import numpy as np


def probability_to_angle(p: float) -> float:
    """Convert probability p in [0,1] to Ry rotation angle theta.

    P(|1>) = sin^2(theta/2)  =>  theta = 2*arcsin(sqrt(p))
    """
    p = np.clip(p, 0.0, 1.0)
    return float(2.0 * np.arcsin(np.sqrt(p)))


def angle_to_probability(theta: float) -> float:
    """Convert Ry rotation angle to measurement probability P(|1>).

    P(|1>) = sin^2(theta/2)
    """
    return float(np.sin(theta / 2.0) ** 2)


def bitstream_to_statevector(bits: np.ndarray) -> np.ndarray:
    """Decode bitstream mean probability, return single-qubit statevector [alpha, beta].

    |psi> = cos(theta/2)|0> + sin(theta/2)|1>  where  sin^2(theta/2) = mean(bits)
    """
    p = float(np.mean(bits))
    theta = probability_to_angle(p)
    return np.array([np.cos(theta / 2.0), np.sin(theta / 2.0)])


def measurement_to_bitstream(
    counts: dict, length: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Convert shot counts {'0': n0, '1': n1} to Bernoulli bitstream of given length."""
    total = sum(counts.values())
    p_one = counts.get("1", 0) / total if total > 0 else 0.0
    if rng is None:
        rng = np.random.default_rng()
    return rng.binomial(1, p_one, size=length).astype(np.uint8)
