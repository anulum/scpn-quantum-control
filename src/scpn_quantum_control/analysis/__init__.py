# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Quantum analysis toolkit for the Kuramoto-XY system."""

from .bkt_analysis import (
    BKTResult,
    bkt_analysis,
    coupling_laplacian,
    estimate_t_bkt,
    fiedler_eigenvalue,
    scan_synchronization_transition,
)
from .bkt_universals import BKTUniversalsSummary, check_all_candidates
from .dynamical_lie_algebra import DLAResult, compute_dla
from .enaqt import ENAQTResult, enaqt_scan
from .entanglement_spectrum import (
    EntanglementResult,
    entanglement_analysis,
    entropy_vs_coupling_scan,
)
from .h1_persistence import H1PersistenceResult, scan_h1_persistence
from .hamiltonian_learning import HamiltonianLearningResult, learn_hamiltonian
from .koopman import KoopmanResult, koopman_analysis, koopman_to_hamiltonian
from .otoc import OTOCResult, compute_otoc
from .p_h1_derivation import P_H1_Derivation, derive_p_h1
from .phase_diagram import (
    PhaseBoundary,
    PhaseDiagramResult,
    compute_phase_diagram,
    critical_coupling_finite_graph,
    critical_coupling_mean_field,
    decoherence_temperature,
    effective_temperature,
    order_parameter_steady_state,
)
from .qfi import QFIResult, compute_qfi, qfi_gap_tradeoff
from .quantum_phi import PhiResult, compute_quantum_phi, phi_vs_coupling_scan, von_neumann_entropy
from .shadow_tomography import ShadowResult, classical_shadow_estimation
from .vortex_binding import VortexBindingResult, compute_vortex_binding

__all__ = [
    "BKTResult",
    "bkt_analysis",
    "coupling_laplacian",
    "estimate_t_bkt",
    "fiedler_eigenvalue",
    "scan_synchronization_transition",
    "BKTUniversalsSummary",
    "check_all_candidates",
    "DLAResult",
    "compute_dla",
    "ENAQTResult",
    "enaqt_scan",
    "EntanglementResult",
    "entanglement_analysis",
    "entropy_vs_coupling_scan",
    "H1PersistenceResult",
    "scan_h1_persistence",
    "HamiltonianLearningResult",
    "learn_hamiltonian",
    "KoopmanResult",
    "koopman_analysis",
    "koopman_to_hamiltonian",
    "OTOCResult",
    "compute_otoc",
    "P_H1_Derivation",
    "derive_p_h1",
    "PhaseBoundary",
    "PhaseDiagramResult",
    "compute_phase_diagram",
    "critical_coupling_finite_graph",
    "critical_coupling_mean_field",
    "decoherence_temperature",
    "effective_temperature",
    "order_parameter_steady_state",
    "QFIResult",
    "compute_qfi",
    "qfi_gap_tradeoff",
    "PhiResult",
    "compute_quantum_phi",
    "phi_vs_coupling_scan",
    "von_neumann_entropy",
    "ShadowResult",
    "classical_shadow_estimation",
    "VortexBindingResult",
    "compute_vortex_binding",
]
