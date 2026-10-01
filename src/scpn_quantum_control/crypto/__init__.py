# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Cryptography
"""Quantum cryptography research module.

Topology-authenticated QKD using SCPN coupling matrix K_nm as shared
secret. The Kuramoto-XY isomorphism converts K_nm into an entangled
ground state whose measurement statistics serve as correlated key material.

Research status: scaffolding only — no production crypto.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    from .ml_dsa_seal import MLDSASigner, MLDSAVerifier
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
    from .pqc_trigger import PqcTriggerSigner, PrivateKey, PublicKey, Signature
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

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "bell_inequality_test": (
        "scpn_quantum_control.crypto.entanglement_qkd",
        "bell_inequality_test",
    ),
    "correlator_matrix": ("scpn_quantum_control.crypto.entanglement_qkd", "correlator_matrix"),
    "scpn_qkd_protocol": ("scpn_quantum_control.crypto.entanglement_qkd", "scpn_qkd_protocol"),
    "derive_layer_key": ("scpn_quantum_control.crypto.hierarchical_keys", "derive_layer_key"),
    "derive_master_key": ("scpn_quantum_control.crypto.hierarchical_keys", "derive_master_key"),
    "evolve_key_phases": ("scpn_quantum_control.crypto.hierarchical_keys", "evolve_key_phases"),
    "group_key": ("scpn_quantum_control.crypto.hierarchical_keys", "group_key"),
    "hmac_sign": ("scpn_quantum_control.crypto.hierarchical_keys", "hmac_sign"),
    "hmac_verify_key": ("scpn_quantum_control.crypto.hierarchical_keys", "hmac_verify_key"),
    "key_hierarchy": ("scpn_quantum_control.crypto.hierarchical_keys", "key_hierarchy"),
    "rotating_key_schedule": (
        "scpn_quantum_control.crypto.hierarchical_keys",
        "rotating_key_schedule",
    ),
    "verify_key_chain": ("scpn_quantum_control.crypto.hierarchical_keys", "verify_key_chain"),
    "estimate_qber": ("scpn_quantum_control.crypto.knm_key", "estimate_qber"),
    "extract_raw_key": ("scpn_quantum_control.crypto.knm_key", "extract_raw_key"),
    "prepare_key_state": ("scpn_quantum_control.crypto.knm_key", "prepare_key_state"),
    "privacy_amplification": ("scpn_quantum_control.crypto.knm_key", "privacy_amplification"),
    "MLDSASigner": ("scpn_quantum_control.crypto.ml_dsa_seal", "MLDSASigner"),
    "MLDSAVerifier": ("scpn_quantum_control.crypto.ml_dsa_seal", "MLDSAVerifier"),
    "amplitude_damping_single": (
        "scpn_quantum_control.crypto.noise_analysis",
        "amplitude_damping_single",
    ),
    "depolarizing_channel": ("scpn_quantum_control.crypto.noise_analysis", "depolarizing_channel"),
    "devetak_winter_rate": ("scpn_quantum_control.crypto.noise_analysis", "devetak_winter_rate"),
    "intercept_resend_qber": (
        "scpn_quantum_control.crypto.noise_analysis",
        "intercept_resend_qber",
    ),
    "noisy_concurrence": ("scpn_quantum_control.crypto.noise_analysis", "noisy_concurrence"),
    "security_analysis": ("scpn_quantum_control.crypto.noise_analysis", "security_analysis"),
    "active_channel_graph": ("scpn_quantum_control.crypto.percolation", "active_channel_graph"),
    "best_entanglement_path": (
        "scpn_quantum_control.crypto.percolation",
        "best_entanglement_path",
    ),
    "concurrence_map": ("scpn_quantum_control.crypto.percolation", "concurrence_map"),
    "key_rate_per_channel": ("scpn_quantum_control.crypto.percolation", "key_rate_per_channel"),
    "percolation_threshold": ("scpn_quantum_control.crypto.percolation", "percolation_threshold"),
    "robustness_random_removal": (
        "scpn_quantum_control.crypto.percolation",
        "robustness_random_removal",
    ),
    "robustness_targeted_removal": (
        "scpn_quantum_control.crypto.percolation",
        "robustness_targeted_removal",
    ),
    "PqcTriggerSigner": ("scpn_quantum_control.crypto.pqc_trigger", "PqcTriggerSigner"),
    "PrivateKey": ("scpn_quantum_control.crypto.pqc_trigger", "PrivateKey"),
    "PublicKey": ("scpn_quantum_control.crypto.pqc_trigger", "PublicKey"),
    "Signature": ("scpn_quantum_control.crypto.pqc_trigger", "Signature"),
    "EIGENVALUE_ZERO_ATOL": ("scpn_quantum_control.crypto.topology_auth", "EIGENVALUE_ZERO_ATOL"),
    "EIGENVALUE_ZERO_RTOL": ("scpn_quantum_control.crypto.topology_auth", "EIGENVALUE_ZERO_RTOL"),
    "challenge_response_prove": (
        "scpn_quantum_control.crypto.topology_auth",
        "challenge_response_prove",
    ),
    "challenge_response_verify": (
        "scpn_quantum_control.crypto.topology_auth",
        "challenge_response_verify",
    ),
    "fingerprint_noise_tolerance": (
        "scpn_quantum_control.crypto.topology_auth",
        "fingerprint_noise_tolerance",
    ),
    "normalized_laplacian_fingerprint": (
        "scpn_quantum_control.crypto.topology_auth",
        "normalized_laplacian_fingerprint",
    ),
    "row_hash_fingerprint": ("scpn_quantum_control.crypto.topology_auth", "row_hash_fingerprint"),
    "spectral_fingerprint": ("scpn_quantum_control.crypto.topology_auth", "spectral_fingerprint"),
    "topology_commitment": ("scpn_quantum_control.crypto.topology_auth", "topology_commitment"),
    "topology_distance": ("scpn_quantum_control.crypto.topology_auth", "topology_distance"),
    "verify_commitment": ("scpn_quantum_control.crypto.topology_auth", "verify_commitment"),
    "verify_fingerprint": ("scpn_quantum_control.crypto.topology_auth", "verify_fingerprint"),
    "verify_row_hash": ("scpn_quantum_control.crypto.topology_auth", "verify_row_hash"),
}


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


__all__ = [
    "PqcTriggerSigner",
    "PrivateKey",
    "PublicKey",
    "Signature",
    "MLDSASigner",
    "MLDSAVerifier",
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
