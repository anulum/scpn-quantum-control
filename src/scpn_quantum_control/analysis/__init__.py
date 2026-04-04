# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Analysis Toolkit
"""Quantum analysis toolkit for the Kuramoto-XY system."""

from .berry_phase import BerryPhaseResult, berry_phase_scan
from .bkt_analysis import (
    BKTResult,
    bkt_analysis,
    coupling_laplacian,
    estimate_t_bkt,
    fiedler_eigenvalue,
    scan_synchronization_transition,
)
from .bkt_universals import BKTUniversalsSummary, check_all_candidates
from .critical_concordance import ConcordanceResult
from .critical_concordance import critical_concordance as compute_critical_concordance
from .dla_parity_theorem import DLAParityTheoremResult
from .dla_parity_theorem import verify_theorem as verify_z2_parity
from .dynamical_lie_algebra import DLAResult, compute_dla
from .enaqt import ENAQTResult, enaqt_scan
from .entanglement_enhanced_sync import entanglement_advantage
from .entanglement_entropy import entanglement_vs_coupling
from .entanglement_percolation import PercolationScanResult, percolation_scan
from .entanglement_spectrum import (
    EntanglementResult,
    entanglement_analysis,
    entropy_vs_coupling_scan,
)
from .finite_size_scaling import FSSResult
from .finite_size_scaling import finite_size_scaling as compute_finite_size_scaling
from .h1_persistence import H1PersistenceResult, scan_h1_persistence
from .hamiltonian_learning import HamiltonianLearningResult, learn_hamiltonian
from .hamiltonian_self_consistency import SelfConsistencyResult, self_consistency_from_exact
from .koopman import KoopmanResult, koopman_analysis, koopman_to_hamiltonian
from .krylov_complexity import KrylovResult, krylov_vs_coupling
from .lindblad_ness import NESSResult, ness_vs_coupling
from .loschmidt_echo import LoschmidtResult, loschmidt_quench
from .magic_nonstabilizerness import MagicResult, magic_vs_coupling
from .magnetisation_sectors import basis_by_magnetisation, eigh_by_magnetisation
from .monte_carlo_xy import MCResult, mc_simulate
from .otoc import OTOCResult, compute_otoc
from .otoc_sync_probe import OTOCSyncScanResult, otoc_sync_scan
from .p_h1_derivation import P_H1_Derivation, derive_p_h1
from .pairing_correlator import PairingResult, pairing_vs_anisotropy
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
from .qfi_criticality import QFICriticalityResult, qfi_vs_coupling
from .qrc_phase_detector import QRCPhaseResult, qrc_phase_detection
from .quantum_mpemba import MpembaResult, mpemba_experiment
from .quantum_persistent_homology import QuantumPHResult, ph_sync_scan
from .quantum_phi import PhiResult, compute_quantum_phi, phi_vs_coupling_scan, von_neumann_entropy
from .shadow_tomography import ShadowResult, classical_shadow_estimation
from .spectral_form_factor import SFFResult, sff_vs_coupling
from .sync_entanglement_witness import (
    EntanglementWitnessResult,
    R_from_statevector,
    R_separable_bound,
)
from .sync_witness import WitnessResult, evaluate_all_witnesses
from .vortex_binding import VortexBindingResult, compute_vortex_binding
from .xxz_phase_diagram import AnisotropyScanResult, anisotropy_phase_diagram

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
    "BerryPhaseResult",
    "berry_phase_scan",
    "ConcordanceResult",
    "compute_critical_concordance",
    "DLAParityTheoremResult",
    "verify_z2_parity",
    "entanglement_advantage",
    "entanglement_vs_coupling",
    "PercolationScanResult",
    "percolation_scan",
    "FSSResult",
    "compute_finite_size_scaling",
    "SelfConsistencyResult",
    "self_consistency_from_exact",
    "KrylovResult",
    "krylov_vs_coupling",
    "NESSResult",
    "ness_vs_coupling",
    "LoschmidtResult",
    "loschmidt_quench",
    "MagicResult",
    "magic_vs_coupling",
    "basis_by_magnetisation",
    "eigh_by_magnetisation",
    "MCResult",
    "mc_simulate",
    "OTOCSyncScanResult",
    "otoc_sync_scan",
    "PairingResult",
    "pairing_vs_anisotropy",
    "QFICriticalityResult",
    "qfi_vs_coupling",
    "MpembaResult",
    "mpemba_experiment",
    "QuantumPHResult",
    "ph_sync_scan",
    "QRCPhaseResult",
    "qrc_phase_detection",
    "SFFResult",
    "sff_vs_coupling",
    "EntanglementWitnessResult",
    "R_from_statevector",
    "R_separable_bound",
    "WitnessResult",
    "evaluate_all_witnesses",
    "AnisotropyScanResult",
    "anisotropy_phase_diagram",
]
