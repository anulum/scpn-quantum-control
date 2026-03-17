"""Identity continuity analysis for coupled oscillator networks.

Quantitative tools for characterizing identity attractor basins,
coherence budgets, entanglement structure, and cryptographic
fingerprinting of coupling topologies.
"""

from .coherence_budget import coherence_budget, fidelity_at_depth
from .entanglement_witness import chsh_from_statevector, disposition_entanglement_map
from .ground_state import IdentityAttractor
from .identity_key import identity_fingerprint, verify_identity

__all__ = [
    "IdentityAttractor",
    "coherence_budget",
    "fidelity_at_depth",
    "disposition_entanglement_map",
    "chsh_from_statevector",
    "identity_fingerprint",
    "verify_identity",
]
