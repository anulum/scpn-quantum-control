# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Identity Key
"""Quantum identity fingerprint from coupling topology.

Generates a cryptographic fingerprint from the identity K_nm matrix
via VQE ground state correlations. The K_nm encodes the full history
of disposition co-activation — different session histories produce
different K_nm, therefore different quantum keys.
"""

from __future__ import annotations

import numpy as np

from ..bridge.orchestrator_adapter import PhaseOrchestratorAdapter
from ..crypto.knm_key import prepare_key_state
from ..crypto.topology_auth import (
    challenge_response_prove,
    challenge_response_verify,
    spectral_fingerprint,
    topology_commitment,
)


def identity_fingerprint(
    K: np.ndarray,
    omega: np.ndarray,
    *,
    ansatz_reps: int = 2,
    maxiter: int = 200,
) -> dict:
    """Generate a quantum identity fingerprint from coupling topology.

    Combines spectral fingerprint (public, topology-derived) with
    VQE ground state energy (quantum-derived). The spectral fingerprint
    can be published without revealing K_nm; the ground state energy
    serves as an additional consistency check.

    Returns dict with spectral (public fingerprint), ground_energy,
    commitment (SHA-256 hash binding K_nm), and n_parameters (security).
    """
    n = K.shape[0]
    spectral = spectral_fingerprint(K)
    key_state = prepare_key_state(K, omega, ansatz_reps=ansatz_reps, maxiter=maxiter)

    commitment = topology_commitment(K)

    return {
        "spectral": spectral,
        "ground_energy": key_state["energy"],
        "commitment": commitment.hex(),
        "n_parameters": n * (n - 1) // 2,
        "n_qubits": n,
    }


def identity_fingerprint_from_binding_spec(
    binding_spec: dict,
    **kwargs,
) -> dict:
    """Generate fingerprint from an scpn-phase-orchestrator binding spec."""
    K = PhaseOrchestratorAdapter.build_knm_from_binding_spec(
        binding_spec,
        zero_diagonal=True,
    )
    omega = PhaseOrchestratorAdapter.build_omega_from_binding_spec(binding_spec)
    return identity_fingerprint(K, omega, **kwargs)


def verify_identity(
    K: np.ndarray,
    challenge: bytes,
    response: bytes,
) -> bool:
    """Verify that a claimant holds the correct K_nm.

    The verifier sends a random challenge; the claimant responds with
    HMAC(K_nm, challenge). Returns True iff the response matches.
    """
    return challenge_response_verify(K, challenge, response)


def prove_identity(K: np.ndarray, challenge: bytes) -> bytes:
    """Prove knowledge of K_nm without transmitting it."""
    return challenge_response_prove(K, challenge)
