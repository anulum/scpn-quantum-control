# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Free Energy Principle
"""Free Energy Principle (FEP) on the SCPN quantum substrate.

Implements Friston's variational free energy framework mapped onto
the SCPN hierarchy: oscillator phases serve as sufficient statistics,
K_nm couplings as precision parameters, and the UPDE as belief dynamics.

Modules:
    variational_free_energy — F, KL divergence, ELBO
    predictive_coding — hierarchical message-passing across SCPN layers
    quantum_belief — quantum state as belief, measurement as update
"""

from .predictive_coding import (
    PredictiveCodingResult,
    hierarchical_prediction_error,
    predictive_coding_step,
)
from .variational_free_energy import (
    FreeEnergyResult,
    evidence_lower_bound,
    kl_divergence_gaussian,
    variational_free_energy,
)

__all__ = [
    "FreeEnergyResult",
    "variational_free_energy",
    "kl_divergence_gaussian",
    "evidence_lower_bound",
    "PredictiveCodingResult",
    "hierarchical_prediction_error",
    "predictive_coding_step",
]
