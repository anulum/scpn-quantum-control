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

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "CPDRResult": ("scpn_quantum_control.mitigation.cpdr", "CPDRResult"),
    "cpdr_full_pipeline": ("scpn_quantum_control.mitigation.cpdr", "cpdr_full_pipeline"),
    "cpdr_mitigate": ("scpn_quantum_control.mitigation.cpdr", "cpdr_mitigate"),
    "generate_training_circuits": (
        "scpn_quantum_control.mitigation.cpdr",
        "generate_training_circuits",
    ),
    "DDSequence": ("scpn_quantum_control.mitigation.dd", "DDSequence"),
    "insert_dd_sequence": ("scpn_quantum_control.mitigation.dd", "insert_dd_sequence"),
    "ddd_mitigated_expectation": (
        "scpn_quantum_control.mitigation.mitiq_integration",
        "ddd_mitigated_expectation",
    ),
    "is_mitiq_available": (
        "scpn_quantum_control.mitigation.mitiq_integration",
        "is_mitiq_available",
    ),
    "zne_mitigated_expectation": (
        "scpn_quantum_control.mitigation.mitiq_integration",
        "zne_mitigated_expectation",
    ),
    "PECResult": ("scpn_quantum_control.mitigation.pec", "PECResult"),
    "pauli_twirl_decompose": ("scpn_quantum_control.mitigation.pec", "pauli_twirl_decompose"),
    "pec_sample": ("scpn_quantum_control.mitigation.pec", "pec_sample"),
    "ReadoutConfusionMatrix": (
        "scpn_quantum_control.mitigation.readout_matrix",
        "ReadoutConfusionMatrix",
    ),
    "build_readout_confusion_matrix": (
        "scpn_quantum_control.mitigation.readout_matrix",
        "build_readout_confusion_matrix",
    ),
    "computational_basis_labels": (
        "scpn_quantum_control.mitigation.readout_matrix",
        "computational_basis_labels",
    ),
    "counts_to_probabilities": (
        "scpn_quantum_control.mitigation.readout_matrix",
        "counts_to_probabilities",
    ),
    "mitigate_counts": ("scpn_quantum_control.mitigation.readout_matrix", "mitigate_counts"),
    "mitigate_probabilities": (
        "scpn_quantum_control.mitigation.readout_matrix",
        "mitigate_probabilities",
    ),
    "probability_magnetisation_leakage": (
        "scpn_quantum_control.mitigation.readout_matrix",
        "probability_magnetisation_leakage",
    ),
    "probability_mean_magnetisation": (
        "scpn_quantum_control.mitigation.readout_matrix",
        "probability_mean_magnetisation",
    ),
    "probability_parity_leakage": (
        "scpn_quantum_control.mitigation.readout_matrix",
        "probability_parity_leakage",
    ),
    "probability_state_retention": (
        "scpn_quantum_control.mitigation.readout_matrix",
        "probability_state_retention",
    ),
    "GUESSResult": ("scpn_quantum_control.mitigation.symmetry_decay", "GUESSResult"),
    "SymmetryDecayModel": ("scpn_quantum_control.mitigation.symmetry_decay", "SymmetryDecayModel"),
    "guess_extrapolate": ("scpn_quantum_control.mitigation.symmetry_decay", "guess_extrapolate"),
    "learn_symmetry_decay": (
        "scpn_quantum_control.mitigation.symmetry_decay",
        "learn_symmetry_decay",
    ),
    "xy_magnetisation_ideal": (
        "scpn_quantum_control.mitigation.symmetry_decay",
        "xy_magnetisation_ideal",
    ),
    "SymmetrySectorPlan": (
        "scpn_quantum_control.mitigation.symmetry_sector_compiler",
        "SymmetrySectorPlan",
    ),
    "SymmetrySectorProblem": (
        "scpn_quantum_control.mitigation.symmetry_sector_compiler",
        "SymmetrySectorProblem",
    ),
    "plan_symmetry_sector_mitigation": (
        "scpn_quantum_control.mitigation.symmetry_sector_compiler",
        "plan_symmetry_sector_mitigation",
    ),
    "SymmetrySectorReplayResult": (
        "scpn_quantum_control.mitigation.symmetry_sector_replay",
        "SymmetrySectorReplayResult",
    ),
    "replay_symmetry_sector_counts": (
        "scpn_quantum_control.mitigation.symmetry_sector_replay",
        "replay_symmetry_sector_counts",
    ),
    "SymmetryVerificationResult": (
        "scpn_quantum_control.mitigation.symmetry_verification",
        "SymmetryVerificationResult",
    ),
    "bitstring_parity": (
        "scpn_quantum_control.mitigation.symmetry_verification",
        "bitstring_parity",
    ),
    "initial_state_parity": (
        "scpn_quantum_control.mitigation.symmetry_verification",
        "initial_state_parity",
    ),
    "parity_postselect": (
        "scpn_quantum_control.mitigation.symmetry_verification",
        "parity_postselect",
    ),
    "parity_verified_expectation": (
        "scpn_quantum_control.mitigation.symmetry_verification",
        "parity_verified_expectation",
    ),
    "parity_verified_R": (
        "scpn_quantum_control.mitigation.symmetry_verification",
        "parity_verified_R",
    ),
    "symmetry_expand": (
        "scpn_quantum_control.mitigation.symmetry_verification",
        "symmetry_expand",
    ),
    "ZNEResult": ("scpn_quantum_control.mitigation.zne", "ZNEResult"),
    "gate_fold_circuit": ("scpn_quantum_control.mitigation.zne", "gate_fold_circuit"),
    "zne_extrapolate": ("scpn_quantum_control.mitigation.zne", "zne_extrapolate"),
    "ZNEUncertaintyResult": (
        "scpn_quantum_control.mitigation.zne_uncertainty",
        "ZNEUncertaintyResult",
    ),
    "zne_extrapolate_with_uncertainty": (
        "scpn_quantum_control.mitigation.zne_uncertainty",
        "zne_extrapolate_with_uncertainty",
    ),
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
