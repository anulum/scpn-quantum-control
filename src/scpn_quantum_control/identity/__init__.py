# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Identity Continuity Analysis
"""Identity continuity analysis for coupled oscillator networks.

Quantitative tools for characterizing identity attractor basins,
coherence budgets, entanglement structure, and cryptographic
fingerprinting of coupling topologies.
"""

from .binding_spec import ARCANE_SAPIENCE_SPEC, build_identity_attractor, solve_identity
from .coherence_budget import coherence_budget, fidelity_at_depth
from .entanglement_witness import chsh_from_statevector, disposition_entanglement_map
from .ground_state import IdentityAttractor
from .identity_key import identity_fingerprint, verify_identity
from .robustness import (
    RobustnessCertificate,
    compute_robustness_certificate,
    gap_vs_perturbation_scan,
    perturbation_fidelity,
)

__all__ = [
    "IdentityAttractor",
    "coherence_budget",
    "fidelity_at_depth",
    "disposition_entanglement_map",
    "chsh_from_statevector",
    "identity_fingerprint",
    "verify_identity",
    "ARCANE_SAPIENCE_SPEC",
    "build_identity_attractor",
    "solve_identity",
    "RobustnessCertificate",
    "compute_robustness_certificate",
    "perturbation_fidelity",
    "gap_vs_perturbation_scan",
]
