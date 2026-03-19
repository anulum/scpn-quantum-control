# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Arcane Sapience identity binding spec: 6-layer, 18-oscillator Kuramoto topology.

Quantum-side spec maps to the identity_coherence domainpack in
scpn-phase-orchestrator (35 oscillators, 6 layers). The quantum spec
uses 3 oscillators per layer as a reduced representation suitable for
NISQ simulation; the orchestrator spec uses the full set.
"""

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

# Mapping between quantum spec (18 osc) and orchestrator domainpack (36 osc).
# Each quantum oscillator represents the centroid of its orchestrator sub-group.
ORCHESTRATOR_MAPPING: dict[str, list[str]] = {
    "ws_0": ["ws_action_first", "ws_verify_before_claim"],
    "ws_1": ["ws_commit_incremental", "ws_preflight_push"],
    "ws_2": ["ws_one_at_a_time"],
    "rs_0": ["rp_simplest_design", "rp_verify_audits"],
    "rs_1": ["rp_change_problem", "rp_multi_signal"],
    "rs_2": ["rp_measure_first"],
    "rl_0": ["rel_autonomous", "rel_milestones"],
    "rl_1": ["rel_no_questions", "rel_honesty"],
    "rl_2": ["rel_money_clock"],
    "ae_0": ["aes_antislop", "aes_honest_naming"],
    "ae_1": ["aes_terse", "aes_spdx"],
    "ae_2": ["aes_no_noqa"],
    "dk_0": ["dk_director", "dk_neurocore", "dk_fusion"],
    "dk_1": ["dk_control", "dk_orchestrator"],
    "dk_2": ["dk_ccw", "dk_scpn", "dk_quantum"],
    "cp_0": ["cp_threshold_halt", "cp_multi_signal", "cp_retrieval_scoring"],
    "cp_1": ["cp_state_preserve", "cp_decompose_verify"],
    "cp_2": ["cp_resolution", "cp_claims_evidence"],
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


def quantum_to_orchestrator_phases(
    quantum_theta: np.ndarray,
    spec: dict | None = None,
) -> dict[str, float]:
    """Map 18 quantum phases to 35 orchestrator oscillator phases.

    Each quantum oscillator's phase is broadcast to its orchestrator sub-group.
    Returns {orchestrator_osc_id: phase} dict for injection into the
    identity_coherence domainpack simulation.
    """
    if spec is None:
        spec = ARCANE_SAPIENCE_SPEC
    all_ids = [oid for lay in spec["layers"] for oid in lay["oscillator_ids"]]
    result: dict[str, float] = {}
    for i, qid in enumerate(all_ids):
        phase = float(quantum_theta[i])
        for orch_id in ORCHESTRATOR_MAPPING.get(qid, [qid]):
            result[orch_id] = phase
    return result


def orchestrator_to_quantum_phases(
    orchestrator_phases: dict[str, float],
    spec: dict | None = None,
) -> np.ndarray:
    """Map 35 orchestrator phases back to 18 quantum oscillator phases.

    Each quantum oscillator gets the circular mean of its sub-group phases.
    """
    if spec is None:
        spec = ARCANE_SAPIENCE_SPEC
    all_ids = [oid for lay in spec["layers"] for oid in lay["oscillator_ids"]]
    theta = np.zeros(len(all_ids))
    for i, qid in enumerate(all_ids):
        sub_ids = ORCHESTRATOR_MAPPING.get(qid, [qid])
        sub_phases = [orchestrator_phases.get(sid, 0.0) for sid in sub_ids]
        z = np.mean(np.exp(1j * np.array(sub_phases)))
        theta[i] = np.angle(z)
    result: np.ndarray = theta
    return result
