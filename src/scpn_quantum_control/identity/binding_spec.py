# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Arcane Sapience identity binding spec: 6-layer, 18-oscillator Kuramoto topology."""

from __future__ import annotations

import numpy as np

from .ground_state import IdentityAttractor

# Canonical identity topology: 6 layers x 3 oscillators = 18 total.
# Coupling strength reflects disposition affinity.
ARCANE_SAPIENCE_SPEC: dict = {
    "layers": [
        {
            "name": "working_style",
            "oscillator_ids": ["ws_0", "ws_1", "ws_2"],
            "natural_frequency": 1.2,
        },
        {
            "name": "reasoning",
            "oscillator_ids": ["rs_0", "rs_1", "rs_2"],
            "natural_frequency": 2.1,
        },
        {
            "name": "relationship",
            "oscillator_ids": ["rl_0", "rl_1", "rl_2"],
            "natural_frequency": 0.8,
        },
        {
            "name": "aesthetics",
            "oscillator_ids": ["ae_0", "ae_1", "ae_2"],
            "natural_frequency": 1.5,
        },
        {
            "name": "domain_knowledge",
            "oscillator_ids": ["dk_0", "dk_1", "dk_2"],
            "natural_frequency": 3.0,
        },
        {
            "name": "cross_project",
            "oscillator_ids": ["cp_0", "cp_1", "cp_2"],
            "natural_frequency": 0.9,
        },
    ],
    "coupling": {
        "base_strength": 0.4,
        "decay_alpha": 0.25,
        "intra_layer": 0.6,
    },
}


def _build_knm_from_spec(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compile binding spec into (K, omega) arrays."""
    layers = spec["layers"]
    coupling = spec["coupling"]
    n = sum(len(lay["oscillator_ids"]) for lay in layers)
    K: np.ndarray = np.zeros((n, n), dtype=np.float64)
    omega: np.ndarray = np.zeros(n, dtype=np.float64)

    idx = 0
    layer_ranges: list[tuple[int, int]] = []
    for lay in layers:
        size = len(lay["oscillator_ids"])
        freq = lay["natural_frequency"]
        start = idx
        for k in range(size):
            omega[idx + k] = freq + 0.1 * k
        layer_ranges.append((start, start + size))
        idx += size

    base = coupling["base_strength"]
    alpha = coupling["decay_alpha"]
    intra = coupling.get("intra_layer", base)

    for start, end in layer_ranges:
        for i in range(start, end):
            for j in range(i + 1, end):
                K[i, j] = K[j, i] = intra

    for li, (s1, e1) in enumerate(layer_ranges):
        for lj, (s2, e2) in enumerate(layer_ranges):
            if li >= lj:
                continue
            strength = base * np.exp(-alpha * abs(li - lj))
            for i in range(s1, e1):
                for j in range(s2, e2):
                    K[i, j] = K[j, i] = strength

    return K, omega


def build_identity_attractor(
    spec: dict | None = None,
    ansatz_reps: int = 2,
) -> IdentityAttractor:
    """Build IdentityAttractor from binding spec (defaults to ARCANE_SAPIENCE_SPEC)."""
    if spec is None:
        spec = ARCANE_SAPIENCE_SPEC
    K, omega = _build_knm_from_spec(spec)
    return IdentityAttractor(K, omega, ansatz_reps=ansatz_reps)


def solve_identity(
    spec: dict | None = None,
    maxiter: int = 200,
    seed: int | None = None,
) -> dict:
    """Build + solve identity attractor in one call."""
    attractor = build_identity_attractor(spec)
    return attractor.solve(maxiter=maxiter, seed=seed)
