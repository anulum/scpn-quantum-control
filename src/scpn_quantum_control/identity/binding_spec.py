# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Identity Binding Spec
"""Six-layer, 18-oscillator Arcane Sapience identity binding topology.

Quantum-side spec maps to the identity_coherence domainpack in
scpn-phase-orchestrator (35 oscillators, 6 layers). The quantum spec
uses 3 oscillators per layer as a reduced representation suitable for
NISQ simulation; the orchestrator spec uses the full set.
"""

from __future__ import annotations

import re
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .ground_state import IdentityAttractor

# Canonical identity topology: 6 layers x 3 oscillators = 18 total.
# Coupling strength reflects disposition affinity.
ARCANE_SAPIENCE_SPEC: dict[str, Any] = {
    "layers": [
        {
            "name": "working_style",
            "oscillator_ids": [
                "working_style_action_verification",
                "working_style_delivery_preflight",
                "working_style_single_task",
            ],
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

# Mapping between quantum spec (18 osc) and orchestrator domainpack (35 osc).
# Each quantum oscillator represents the centroid of its orchestrator sub-group.
ORCHESTRATOR_MAPPING: dict[str, list[str]] = {
    "working_style_action_verification": ["ws_action_first", "ws_verify_before_claim"],
    "working_style_delivery_preflight": ["ws_commit_incremental", "ws_preflight_push"],
    "working_style_single_task": ["ws_one_at_a_time"],
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

_STALE_WORKING_STYLE_ID: Final[re.Pattern[str]] = re.compile(r"^ws_[0-2]$")


def _oscillator_ids(spec: dict[str, Any]) -> list[str]:
    """Return ordered oscillator IDs and reject the stale abbreviated contract."""
    oscillator_ids = [oid for layer in spec["layers"] for oid in layer["oscillator_ids"]]
    if any(_STALE_WORKING_STYLE_ID.fullmatch(oscillator_id) for oscillator_id in oscillator_ids):
        raise ValueError("stale abbreviated working-style oscillator IDs are not supported")
    return oscillator_ids


def _build_knm_from_spec(
    spec: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compile binding spec into (K, omega) arrays."""
    layers = spec["layers"]
    coupling = spec["coupling"]
    n = len(_oscillator_ids(spec))
    K: NDArray[np.float64] = np.zeros((n, n), dtype=np.float64)
    omega: NDArray[np.float64] = np.zeros(n, dtype=np.float64)

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
    spec: dict[str, Any] | None = None,
    ansatz_reps: int = 2,
) -> IdentityAttractor:
    """Build an identity attractor from a binding specification.

    Parameters
    ----------
    spec : dict[str, Any] or None
        Binding specification. ``None`` selects ``ARCANE_SAPIENCE_SPEC``.
    ansatz_reps : int
        Repetition count for the attractor ansatz.

    Returns
    -------
    IdentityAttractor
        Attractor compiled from the specification's coupling and frequency data.

    Raises
    ------
    ValueError
        If the specification uses the stale abbreviated working-style IDs.

    """
    if spec is None:
        spec = ARCANE_SAPIENCE_SPEC
    K, omega = _build_knm_from_spec(spec)
    return IdentityAttractor(K, omega, ansatz_reps=ansatz_reps)


def solve_identity(
    spec: dict[str, Any] | None = None,
    maxiter: int = 200,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build and solve an identity attractor.

    Parameters
    ----------
    spec : dict[str, Any] or None
        Binding specification. ``None`` selects ``ARCANE_SAPIENCE_SPEC``.
    maxiter : int
        Maximum optimiser iterations.
    seed : int or None
        Optional deterministic optimiser seed.

    Returns
    -------
    dict[str, Any]
        Identity-attractor solve result.

    """
    attractor = build_identity_attractor(spec)
    return attractor.solve(maxiter=maxiter, seed=seed)


def quantum_to_orchestrator_phases(
    quantum_theta: NDArray[np.float64],
    spec: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Map reduced quantum phases to orchestrator oscillator phases.

    Each quantum oscillator's phase is broadcast to its orchestrator sub-group.

    Parameters
    ----------
    quantum_theta : NDArray[np.float64]
        One phase per reduced oscillator, in specification order.
    spec : dict[str, Any] or None
        Binding specification. ``None`` selects ``ARCANE_SAPIENCE_SPEC``.

    Returns
    -------
    dict[str, float]
        Orchestrator oscillator ID to phase mapping for domainpack injection.

    Raises
    ------
    ValueError
        If the specification uses stale abbreviated working-style IDs or the
        phase vector length differs from the reduced oscillator count.

    """
    if spec is None:
        spec = ARCANE_SAPIENCE_SPEC
    all_ids = _oscillator_ids(spec)
    if quantum_theta.shape != (len(all_ids),):
        raise ValueError(f"expected {len(all_ids)} quantum phases, got {quantum_theta.shape}")
    result: dict[str, float] = {}
    for i, qid in enumerate(all_ids):
        phase = float(quantum_theta[i])
        for orch_id in ORCHESTRATOR_MAPPING.get(qid, [qid]):
            result[orch_id] = phase
    return result


def orchestrator_to_quantum_phases(
    orchestrator_phases: dict[str, float],
    spec: dict[str, Any] | None = None,
) -> NDArray[np.float64]:
    """Map orchestrator phases back to reduced quantum oscillator phases.

    Each quantum oscillator gets the circular mean of its sub-group phases.

    Parameters
    ----------
    orchestrator_phases : dict[str, float]
        Orchestrator oscillator ID to phase mapping.
    spec : dict[str, Any] or None
        Binding specification. ``None`` selects ``ARCANE_SAPIENCE_SPEC``.

    Returns
    -------
    NDArray[np.float64]
        Circular-mean phase for each reduced oscillator in specification order.

    Raises
    ------
    ValueError
        If the specification uses stale abbreviated working-style IDs.

    """
    if spec is None:
        spec = ARCANE_SAPIENCE_SPEC
    all_ids = _oscillator_ids(spec)
    theta = np.zeros(len(all_ids))
    for i, qid in enumerate(all_ids):
        sub_ids = ORCHESTRATOR_MAPPING.get(qid, [qid])
        sub_phases = [orchestrator_phases.get(sid, 0.0) for sid in sub_ids]
        z = np.mean(np.exp(1j * np.array(sub_phases)))
        theta[i] = np.angle(z)
    result: NDArray[np.float64] = theta
    return result
