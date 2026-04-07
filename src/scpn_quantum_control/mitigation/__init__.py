# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Error Mitigation
"""Error mitigation techniques: ZNE, PEC, dynamical decoupling, CPDR,
and Z2 symmetry verification with parity post-selection.
"""

from .cpdr import CPDRResult, cpdr_full_pipeline, cpdr_mitigate, generate_training_circuits
from .dd import DDSequence, insert_dd_sequence
from .mitiq_integration import (
    ddd_mitigated_expectation,
    is_mitiq_available,
    zne_mitigated_expectation,
)
from .pec import PECResult, pauli_twirl_decompose, pec_sample
from .symmetry_decay import (
    GUESSResult,
    SymmetryDecayModel,
    guess_extrapolate,
    learn_symmetry_decay,
    xy_magnetisation_ideal,
)
from .symmetry_verification import (
    SymmetryVerificationResult,
    bitstring_parity,
    initial_state_parity,
    parity_postselect,
    parity_verified_expectation,
    parity_verified_R,
    symmetry_expand,
)
from .zne import ZNEResult, gate_fold_circuit, zne_extrapolate

__all__ = [
    "gate_fold_circuit",
    "zne_extrapolate",
    "ZNEResult",
    "DDSequence",
    "insert_dd_sequence",
    "PECResult",
    "pauli_twirl_decompose",
    "pec_sample",
    "CPDRResult",
    "cpdr_mitigate",
    "cpdr_full_pipeline",
    "generate_training_circuits",
    "SymmetryVerificationResult",
    "bitstring_parity",
    "initial_state_parity",
    "parity_postselect",
    "parity_verified_expectation",
    "parity_verified_R",
    "symmetry_expand",
    "is_mitiq_available",
    "zne_mitigated_expectation",
    "ddd_mitigated_expectation",
    "SymmetryDecayModel",
    "GUESSResult",
    "learn_symmetry_decay",
    "guess_extrapolate",
    "xy_magnetisation_ideal",
]
