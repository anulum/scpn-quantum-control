# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Error Mitigation
"""Expose error-mitigation techniques and their public contracts.

The facade re-exports ZNE, PEC, dynamical decoupling, CPDR, readout
correction, symmetry-sector replay, and Z2 parity post-selection helpers.
"""

from .cpdr import CPDRResult, cpdr_full_pipeline, cpdr_mitigate, generate_training_circuits
from .dd import DDSequence, insert_dd_sequence
from .mitiq_integration import (
    ddd_mitigated_expectation,
    is_mitiq_available,
    zne_mitigated_expectation,
)
from .pec import PECResult, pauli_twirl_decompose, pec_sample
from .readout_matrix import (
    ReadoutConfusionMatrix,
    build_readout_confusion_matrix,
    computational_basis_labels,
    counts_to_probabilities,
    mitigate_counts,
    mitigate_probabilities,
    probability_magnetisation_leakage,
    probability_mean_magnetisation,
    probability_parity_leakage,
    probability_state_retention,
)
from .symmetry_decay import (
    GUESSResult,
    SymmetryDecayModel,
    guess_extrapolate,
    learn_symmetry_decay,
    xy_magnetisation_ideal,
)
from .symmetry_sector_compiler import (
    SymmetrySectorPlan,
    SymmetrySectorProblem,
    plan_symmetry_sector_mitigation,
)
from .symmetry_sector_replay import (
    SymmetrySectorReplayResult,
    replay_symmetry_sector_counts,
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
from .zne_uncertainty import ZNEUncertaintyResult, zne_extrapolate_with_uncertainty

__all__ = [
    "gate_fold_circuit",
    "zne_extrapolate",
    "ZNEResult",
    "ZNEUncertaintyResult",
    "zne_extrapolate_with_uncertainty",
    "DDSequence",
    "insert_dd_sequence",
    "PECResult",
    "pauli_twirl_decompose",
    "pec_sample",
    "ReadoutConfusionMatrix",
    "computational_basis_labels",
    "counts_to_probabilities",
    "build_readout_confusion_matrix",
    "mitigate_counts",
    "mitigate_probabilities",
    "probability_state_retention",
    "probability_parity_leakage",
    "probability_magnetisation_leakage",
    "probability_mean_magnetisation",
    "CPDRResult",
    "cpdr_mitigate",
    "cpdr_full_pipeline",
    "generate_training_circuits",
    "SymmetrySectorPlan",
    "SymmetrySectorProblem",
    "SymmetrySectorReplayResult",
    "SymmetryVerificationResult",
    "bitstring_parity",
    "initial_state_parity",
    "parity_postselect",
    "parity_verified_expectation",
    "parity_verified_R",
    "symmetry_expand",
    "plan_symmetry_sector_mitigation",
    "replay_symmetry_sector_counts",
    "is_mitiq_available",
    "zne_mitigated_expectation",
    "ddd_mitigated_expectation",
    "SymmetryDecayModel",
    "GUESSResult",
    "learn_symmetry_decay",
    "guess_extrapolate",
    "xy_magnetisation_ideal",
]
