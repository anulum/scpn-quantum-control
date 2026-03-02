"""Quantum cryptography research module.

Topology-authenticated QKD using SCPN coupling matrix K_nm as shared
secret. The Kuramoto-XY isomorphism converts K_nm into an entangled
ground state whose measurement statistics serve as correlated key material.

Research status: scaffolding only — no production crypto.
"""

from .entanglement_qkd import bell_inequality_test, correlator_matrix, scpn_qkd_protocol
from .hierarchical_keys import (
    derive_layer_key,
    derive_master_key,
    evolve_key_phases,
    group_key,
    hmac_sign,
    hmac_verify_key,
    key_hierarchy,
    rotating_key_schedule,
    verify_key_chain,
)
from .knm_key import estimate_qber, extract_raw_key, prepare_key_state, privacy_amplification
from .noise_analysis import (
    amplitude_damping_single,
    depolarizing_channel,
    devetak_winter_rate,
    intercept_resend_qber,
    noisy_concurrence,
    security_analysis,
)
from .percolation import (
    active_channel_graph,
    best_entanglement_path,
    concurrence_map,
    key_rate_per_channel,
    percolation_threshold,
    robustness_random_removal,
    robustness_targeted_removal,
)
from .topology_auth import (
    EIGENVALUE_ZERO_ATOL,
    EIGENVALUE_ZERO_RTOL,
    challenge_response_prove,
    challenge_response_verify,
    fingerprint_noise_tolerance,
    normalized_laplacian_fingerprint,
    row_hash_fingerprint,
    spectral_fingerprint,
    topology_commitment,
    topology_distance,
    verify_commitment,
    verify_fingerprint,
    verify_row_hash,
)

__all__ = [
    "prepare_key_state",
    "extract_raw_key",
    "estimate_qber",
    "privacy_amplification",
    "spectral_fingerprint",
    "normalized_laplacian_fingerprint",
    "verify_fingerprint",
    "topology_distance",
    "topology_commitment",
    "verify_commitment",
    "challenge_response_prove",
    "challenge_response_verify",
    "fingerprint_noise_tolerance",
    "row_hash_fingerprint",
    "verify_row_hash",
    "EIGENVALUE_ZERO_ATOL",
    "EIGENVALUE_ZERO_RTOL",
    "scpn_qkd_protocol",
    "correlator_matrix",
    "bell_inequality_test",
    "concurrence_map",
    "percolation_threshold",
    "active_channel_graph",
    "key_rate_per_channel",
    "robustness_random_removal",
    "robustness_targeted_removal",
    "best_entanglement_path",
    "derive_master_key",
    "derive_layer_key",
    "key_hierarchy",
    "verify_key_chain",
    "evolve_key_phases",
    "rotating_key_schedule",
    "group_key",
    "hmac_verify_key",
    "hmac_sign",
    "depolarizing_channel",
    "amplitude_damping_single",
    "noisy_concurrence",
    "intercept_resend_qber",
    "devetak_winter_rate",
    "security_analysis",
]
