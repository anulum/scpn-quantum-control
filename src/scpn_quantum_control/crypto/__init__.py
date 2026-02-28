"""Quantum cryptography research module.

Topology-authenticated QKD using SCPN coupling matrix K_nm as shared
secret. The Kuramoto-XY isomorphism converts K_nm into an entangled
ground state whose measurement statistics serve as correlated key material.

Research status: scaffolding only — no production crypto.
"""

__all__ = [
    "knm_key",
    "topology_auth",
    "entanglement_qkd",
    "percolation",
    "hierarchical_keys",
    "noise_analysis",
]
