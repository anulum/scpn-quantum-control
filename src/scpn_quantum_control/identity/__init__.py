# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
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
