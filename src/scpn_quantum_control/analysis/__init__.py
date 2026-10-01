# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Analysis Toolkit
"""Quantum analysis toolkit for the Kuramoto-XY system."""

from __future__ import annotations

import sys as _sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adaptive_fim_feedback import (
        ADAPTIVE_FIM_CLAIM_BOUNDARY,
        ADAPTIVE_FIM_SCHEMA,
        AdaptiveFIMConfig,
        AdaptiveFIMObserverRecord,
        AdaptiveFIMPlan,
        AdaptiveFIMStep,
        BinomialInterval,
        FIMWitness,
        adaptive_count_aware_schedule,
        adaptive_lambda_schedule,
        observer_record_from_step,
        plan_adaptive_fim_schedule,
        propose_count_aware_lambda,
        propose_next_lambda,
        wilson_score_interval,
    )
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
    from .dla_parity_witness import DLAParityWitness
    from .dla_truncated_tn import dla_truncated_tn
    from .dynamical_lie_algebra import DLAResult, compute_dla
    from .enaqt import ENAQTResult, enaqt_scan
    from .entanglement_enhanced_sync import (
        InitialState,
        InitialStateControlComparison,
        SyncTrajectory,
        compare_all_initial_states,
        compare_initial_states_with_dephased_controls,
        entanglement_advantage,
        local_phase_observables,
        mean_single_qubit_linear_entropy,
        prepare_initial_state,
        simulate_sync_trajectory,
        transverse_exchange_coherence,
    )
    from .entanglement_entropy import entanglement_vs_coupling
    from .entanglement_percolation import PercolationScanResult, percolation_scan
    from .entanglement_spectrum import (
        EntanglementResult,
        entanglement_analysis,
        entropy_vs_coupling_scan,
    )
    from .fim_hamiltonian import (
        SpectrumSummary,
        add_fim_feedback,
        adjacent_gap_ratio,
        bipartite_entropy_from_statevector,
        commutator_frobenius_norm_with_diagonal,
        computational_magnetisations,
        fim_diagonal,
        magnetisation_operator_diagonal,
        magnetisation_sector_indices,
        sector_coupling_rows,
        sector_spectrum_rows,
        summarise_spectrum,
    )
    from .finite_size_scaling import FSSFitDiagnostics, FSSResult
    from .finite_size_scaling import finite_size_scaling as compute_finite_size_scaling
    from .h1_persistence import H1PersistenceResult, scan_h1_persistence
    from .hamiltonian_learning import HamiltonianLearningResult, learn_hamiltonian
    from .hamiltonian_self_consistency import SelfConsistencyResult, self_consistency_from_exact
    from .integrated_information_phi import IntegratedInformationPhi
    from .koopman import KoopmanResult, koopman_analysis, koopman_to_hamiltonian
    from .krylov_complexity import KrylovResult, krylov_vs_coupling
    from .lindblad_ness import NESSResult, ness_vs_coupling
    from .logical_sync_witness import LogicalSyncWitness
    from .loschmidt_echo import LoschmidtResult, loschmidt_quench
    from .magic_nonstabilizerness import MagicResult, magic_vs_coupling
    from .magnetisation_sectors import basis_by_magnetisation, eigh_by_magnetisation
    from .monte_carlo_xy import MCResult, mc_simulate
    from .otoc import OTOC, OTOCResult, compute_otoc
    from .otoc_sync_probe import OTOCSyncScanResult, otoc_sync_scan
    from .p_h1_derivation import P_H1_Derivation, derive_p_h1
    from .p_h1_open_guard import (
        P_H1_OPEN_CLAIM_BOUNDARY,
        P_H1_OPEN_GUARD_SCHEMA,
        P_H1OpenGuardReport,
        P_H1OpenGuardViolation,
        public_markdown_paths,
        run_p_h1_open_guard,
        validate_p_h1_open_claim_text,
    )
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
    from .qfi_geometric_crosscheck import QFIGeometricCrosscheck, crosscheck_qfi_geometric
    from .qrc_phase_detector import QRCPhaseResult, qrc_phase_detection
    from .quantum_fisher_information import QuantumFisherInformation
    from .quantum_mpemba import MpembaResult, mpemba_experiment
    from .quantum_persistent_homology import QuantumPHResult, ph_sync_scan
    from .quantum_phi import (
        PhiResult,
        compute_quantum_phi,
        phi_vs_coupling_scan,
        von_neumann_entropy,
    )
    from .research_lane_registry import (
        RESEARCH_LANE_REGISTRY_BOUNDARY,
        RESEARCH_LANE_REGISTRY_SCHEMA,
        ResearchLaneClaimStatus,
        ResearchLaneDiffHook,
        ResearchLaneInventoryReport,
        ResearchLaneMaturity,
        ResearchLaneRecord,
        ResearchLaneRegistryReport,
        assert_research_lane_inventory,
        build_research_lane_registry_report,
        discover_research_lane_modules,
        get_research_lane,
        list_research_lanes,
        render_research_lane_registry_markdown,
        validate_research_lane_inventory,
    )
    from .rl_discovery_agent import RLDiscoveryAgent
    from .rl_pulse_optimizer import RLPulseOptimizer
    from .rl_research_governance import (
        DEFAULT_RL_RESEARCH_SEEDS,
        RL_DENSE_REWARD_CONTRACT,
        RL_ENVIRONMENT_API,
        RL_RESEARCH_CLAIM_BOUNDARY,
        RL_RESEARCH_GOVERNANCE_SCHEMA,
        RLResearchDecision,
        RLResearchGovernanceError,
        RLResearchLane,
        RLResearchPolicy,
        RLSeedEvaluation,
        RLSeedSuiteReport,
        assert_rl_research_allowed,
        assess_rl_research,
        build_rl_research_evidence_report,
        build_witness_seed_suite,
        estimate_witness_evaluation_budget,
        render_rl_research_evidence_markdown,
        run_governed_witness_seed_suite,
    )
    from .sensing import (
        CRITICALITY_TAIL_SCHEMA,
        QUANTUM_SENSING_SCHEMA,
        CriticalitySensingTail,
        QuantumSensingReadinessConfig,
        SensingGainRow,
        SensingGainScan,
        metrological_gain_vs_k,
        optimal_sensing_k,
        qfi_criticality_sensing_tail,
        quantum_sensing_markdown,
        quantum_sensing_payload,
    )
    from .shadow_tomography import ShadowResult, classical_shadow_estimation
    from .spectral_form_factor import SFFResult, sff_vs_coupling
    from .sync_entanglement_witness import (
        EntanglementWitnessResult,
        R_from_statevector,
        R_separable_bound,
    )
    from .sync_order_parameter import SyncOrderParameter
    from .sync_uncertainty import (
        UncertaintyInterval,
        metric_bootstrap,
        order_parameter_bootstrap,
        order_parameter_estimate,
        order_parameter_shot_noise,
    )
    from .sync_witness import WitnessResult, evaluate_all_witnesses
    from .tcbo_weighted_complex import (
        TCBOWeightedComplexResult,
        TCBOWeightedThresholdScan,
        coupling_weighted_edge_matrix,
        tcbo_weighted_complex,
        tcbo_weighted_threshold_scan,
    )
    from .theory_hook_promotion import (
        THEORY_HOOK_PROMOTION_BOUNDARY,
        THEORY_HOOK_PROMOTION_SCHEMA,
        TheoryHookEvidenceRecord,
        TheoryHookPromotionRecord,
        TheoryHookPromotionReport,
        TheoryHookRole,
        TheoryHookStatus,
        TheoryHookTier,
        build_theory_hook_promotion_report,
        get_theory_hook_promotion,
        list_theory_hook_promotions,
        render_theory_hook_promotion_markdown,
        run_theory_hook_evidence,
    )
    from .thermodynamic_witness import ThermodynamicWitness
    from .vortex_binding import VortexBindingResult, compute_vortex_binding
    from .witness_discovery import (
        WitnessCandidate,
        WitnessDiscoveryEvaluation,
        WitnessDiscoveryResult,
        WitnessDiscoverySpec,
        WitnessSearchMode,
        discover_kuramoto_witnesses,
        score_witness_candidates,
    )
    from .xxz_phase_diagram import AnisotropyScanResult, anisotropy_phase_diagram

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ADAPTIVE_FIM_CLAIM_BOUNDARY": (
        "scpn_quantum_control.analysis.adaptive_fim_feedback",
        "ADAPTIVE_FIM_CLAIM_BOUNDARY",
    ),
    "ADAPTIVE_FIM_SCHEMA": (
        "scpn_quantum_control.analysis.adaptive_fim_feedback",
        "ADAPTIVE_FIM_SCHEMA",
    ),
    "AdaptiveFIMConfig": (
        "scpn_quantum_control.analysis.adaptive_fim_feedback",
        "AdaptiveFIMConfig",
    ),
    "AdaptiveFIMObserverRecord": (
        "scpn_quantum_control.analysis.adaptive_fim_feedback",
        "AdaptiveFIMObserverRecord",
    ),
    "AdaptiveFIMPlan": ("scpn_quantum_control.analysis.adaptive_fim_feedback", "AdaptiveFIMPlan"),
    "AdaptiveFIMStep": ("scpn_quantum_control.analysis.adaptive_fim_feedback", "AdaptiveFIMStep"),
    "BinomialInterval": (
        "scpn_quantum_control.analysis.adaptive_fim_feedback",
        "BinomialInterval",
    ),
    "FIMWitness": ("scpn_quantum_control.analysis.adaptive_fim_feedback", "FIMWitness"),
    "adaptive_count_aware_schedule": (
        "scpn_quantum_control.analysis.adaptive_fim_feedback",
        "adaptive_count_aware_schedule",
    ),
    "adaptive_lambda_schedule": (
        "scpn_quantum_control.analysis.adaptive_fim_feedback",
        "adaptive_lambda_schedule",
    ),
    "observer_record_from_step": (
        "scpn_quantum_control.analysis.adaptive_fim_feedback",
        "observer_record_from_step",
    ),
    "plan_adaptive_fim_schedule": (
        "scpn_quantum_control.analysis.adaptive_fim_feedback",
        "plan_adaptive_fim_schedule",
    ),
    "propose_count_aware_lambda": (
        "scpn_quantum_control.analysis.adaptive_fim_feedback",
        "propose_count_aware_lambda",
    ),
    "propose_next_lambda": (
        "scpn_quantum_control.analysis.adaptive_fim_feedback",
        "propose_next_lambda",
    ),
    "wilson_score_interval": (
        "scpn_quantum_control.analysis.adaptive_fim_feedback",
        "wilson_score_interval",
    ),
    "BerryPhaseResult": ("scpn_quantum_control.analysis.berry_phase", "BerryPhaseResult"),
    "berry_phase_scan": ("scpn_quantum_control.analysis.berry_phase", "berry_phase_scan"),
    "BKTResult": ("scpn_quantum_control.analysis.bkt_analysis", "BKTResult"),
    "bkt_analysis": ("scpn_quantum_control.analysis.bkt_analysis", "bkt_analysis"),
    "coupling_laplacian": ("scpn_quantum_control.analysis.bkt_analysis", "coupling_laplacian"),
    "estimate_t_bkt": ("scpn_quantum_control.analysis.bkt_analysis", "estimate_t_bkt"),
    "fiedler_eigenvalue": ("scpn_quantum_control.analysis.bkt_analysis", "fiedler_eigenvalue"),
    "scan_synchronization_transition": (
        "scpn_quantum_control.analysis.bkt_analysis",
        "scan_synchronization_transition",
    ),
    "BKTUniversalsSummary": (
        "scpn_quantum_control.analysis.bkt_universals",
        "BKTUniversalsSummary",
    ),
    "check_all_candidates": (
        "scpn_quantum_control.analysis.bkt_universals",
        "check_all_candidates",
    ),
    "ConcordanceResult": (
        "scpn_quantum_control.analysis.critical_concordance",
        "ConcordanceResult",
    ),
    "compute_critical_concordance": (
        "scpn_quantum_control.analysis.critical_concordance",
        "critical_concordance",
    ),
    "DLAParityTheoremResult": (
        "scpn_quantum_control.analysis.dla_parity_theorem",
        "DLAParityTheoremResult",
    ),
    "verify_z2_parity": ("scpn_quantum_control.analysis.dla_parity_theorem", "verify_theorem"),
    "DLAParityWitness": ("scpn_quantum_control.analysis.dla_parity_witness", "DLAParityWitness"),
    "dla_truncated_tn": ("scpn_quantum_control.analysis.dla_truncated_tn", "dla_truncated_tn"),
    "DLAResult": ("scpn_quantum_control.analysis.dynamical_lie_algebra", "DLAResult"),
    "compute_dla": ("scpn_quantum_control.analysis.dynamical_lie_algebra", "compute_dla"),
    "ENAQTResult": ("scpn_quantum_control.analysis.enaqt", "ENAQTResult"),
    "enaqt_scan": ("scpn_quantum_control.analysis.enaqt", "enaqt_scan"),
    "InitialState": ("scpn_quantum_control.analysis.entanglement_enhanced_sync", "InitialState"),
    "InitialStateControlComparison": (
        "scpn_quantum_control.analysis.entanglement_enhanced_sync",
        "InitialStateControlComparison",
    ),
    "SyncTrajectory": (
        "scpn_quantum_control.analysis.entanglement_enhanced_sync",
        "SyncTrajectory",
    ),
    "compare_all_initial_states": (
        "scpn_quantum_control.analysis.entanglement_enhanced_sync",
        "compare_all_initial_states",
    ),
    "compare_initial_states_with_dephased_controls": (
        "scpn_quantum_control.analysis.entanglement_enhanced_sync",
        "compare_initial_states_with_dephased_controls",
    ),
    "entanglement_advantage": (
        "scpn_quantum_control.analysis.entanglement_enhanced_sync",
        "entanglement_advantage",
    ),
    "local_phase_observables": (
        "scpn_quantum_control.analysis.entanglement_enhanced_sync",
        "local_phase_observables",
    ),
    "mean_single_qubit_linear_entropy": (
        "scpn_quantum_control.analysis.entanglement_enhanced_sync",
        "mean_single_qubit_linear_entropy",
    ),
    "prepare_initial_state": (
        "scpn_quantum_control.analysis.entanglement_enhanced_sync",
        "prepare_initial_state",
    ),
    "simulate_sync_trajectory": (
        "scpn_quantum_control.analysis.entanglement_enhanced_sync",
        "simulate_sync_trajectory",
    ),
    "transverse_exchange_coherence": (
        "scpn_quantum_control.analysis.entanglement_enhanced_sync",
        "transverse_exchange_coherence",
    ),
    "entanglement_vs_coupling": (
        "scpn_quantum_control.analysis.entanglement_entropy",
        "entanglement_vs_coupling",
    ),
    "PercolationScanResult": (
        "scpn_quantum_control.analysis.entanglement_percolation",
        "PercolationScanResult",
    ),
    "percolation_scan": (
        "scpn_quantum_control.analysis.entanglement_percolation",
        "percolation_scan",
    ),
    "EntanglementResult": (
        "scpn_quantum_control.analysis.entanglement_spectrum",
        "EntanglementResult",
    ),
    "entanglement_analysis": (
        "scpn_quantum_control.analysis.entanglement_spectrum",
        "entanglement_analysis",
    ),
    "entropy_vs_coupling_scan": (
        "scpn_quantum_control.analysis.entanglement_spectrum",
        "entropy_vs_coupling_scan",
    ),
    "SpectrumSummary": ("scpn_quantum_control.analysis.fim_hamiltonian", "SpectrumSummary"),
    "add_fim_feedback": ("scpn_quantum_control.analysis.fim_hamiltonian", "add_fim_feedback"),
    "adjacent_gap_ratio": ("scpn_quantum_control.analysis.fim_hamiltonian", "adjacent_gap_ratio"),
    "bipartite_entropy_from_statevector": (
        "scpn_quantum_control.analysis.fim_hamiltonian",
        "bipartite_entropy_from_statevector",
    ),
    "commutator_frobenius_norm_with_diagonal": (
        "scpn_quantum_control.analysis.fim_hamiltonian",
        "commutator_frobenius_norm_with_diagonal",
    ),
    "computational_magnetisations": (
        "scpn_quantum_control.analysis.fim_hamiltonian",
        "computational_magnetisations",
    ),
    "fim_diagonal": ("scpn_quantum_control.analysis.fim_hamiltonian", "fim_diagonal"),
    "magnetisation_operator_diagonal": (
        "scpn_quantum_control.analysis.fim_hamiltonian",
        "magnetisation_operator_diagonal",
    ),
    "magnetisation_sector_indices": (
        "scpn_quantum_control.analysis.fim_hamiltonian",
        "magnetisation_sector_indices",
    ),
    "sector_coupling_rows": (
        "scpn_quantum_control.analysis.fim_hamiltonian",
        "sector_coupling_rows",
    ),
    "sector_spectrum_rows": (
        "scpn_quantum_control.analysis.fim_hamiltonian",
        "sector_spectrum_rows",
    ),
    "summarise_spectrum": ("scpn_quantum_control.analysis.fim_hamiltonian", "summarise_spectrum"),
    "FSSFitDiagnostics": (
        "scpn_quantum_control.analysis.finite_size_scaling",
        "FSSFitDiagnostics",
    ),
    "FSSResult": ("scpn_quantum_control.analysis.finite_size_scaling", "FSSResult"),
    "compute_finite_size_scaling": (
        "scpn_quantum_control.analysis.finite_size_scaling",
        "finite_size_scaling",
    ),
    "H1PersistenceResult": ("scpn_quantum_control.analysis.h1_persistence", "H1PersistenceResult"),
    "scan_h1_persistence": ("scpn_quantum_control.analysis.h1_persistence", "scan_h1_persistence"),
    "HamiltonianLearningResult": (
        "scpn_quantum_control.analysis.hamiltonian_learning",
        "HamiltonianLearningResult",
    ),
    "learn_hamiltonian": (
        "scpn_quantum_control.analysis.hamiltonian_learning",
        "learn_hamiltonian",
    ),
    "SelfConsistencyResult": (
        "scpn_quantum_control.analysis.hamiltonian_self_consistency",
        "SelfConsistencyResult",
    ),
    "self_consistency_from_exact": (
        "scpn_quantum_control.analysis.hamiltonian_self_consistency",
        "self_consistency_from_exact",
    ),
    "IntegratedInformationPhi": (
        "scpn_quantum_control.analysis.integrated_information_phi",
        "IntegratedInformationPhi",
    ),
    "KoopmanResult": ("scpn_quantum_control.analysis.koopman", "KoopmanResult"),
    "koopman_analysis": ("scpn_quantum_control.analysis.koopman", "koopman_analysis"),
    "koopman_to_hamiltonian": ("scpn_quantum_control.analysis.koopman", "koopman_to_hamiltonian"),
    "KrylovResult": ("scpn_quantum_control.analysis.krylov_complexity", "KrylovResult"),
    "krylov_vs_coupling": (
        "scpn_quantum_control.analysis.krylov_complexity",
        "krylov_vs_coupling",
    ),
    "NESSResult": ("scpn_quantum_control.analysis.lindblad_ness", "NESSResult"),
    "ness_vs_coupling": ("scpn_quantum_control.analysis.lindblad_ness", "ness_vs_coupling"),
    "LogicalSyncWitness": (
        "scpn_quantum_control.analysis.logical_sync_witness",
        "LogicalSyncWitness",
    ),
    "LoschmidtResult": ("scpn_quantum_control.analysis.loschmidt_echo", "LoschmidtResult"),
    "loschmidt_quench": ("scpn_quantum_control.analysis.loschmidt_echo", "loschmidt_quench"),
    "MagicResult": ("scpn_quantum_control.analysis.magic_nonstabilizerness", "MagicResult"),
    "magic_vs_coupling": (
        "scpn_quantum_control.analysis.magic_nonstabilizerness",
        "magic_vs_coupling",
    ),
    "basis_by_magnetisation": (
        "scpn_quantum_control.analysis.magnetisation_sectors",
        "basis_by_magnetisation",
    ),
    "eigh_by_magnetisation": (
        "scpn_quantum_control.analysis.magnetisation_sectors",
        "eigh_by_magnetisation",
    ),
    "MCResult": ("scpn_quantum_control.analysis.monte_carlo_xy", "MCResult"),
    "mc_simulate": ("scpn_quantum_control.analysis.monte_carlo_xy", "mc_simulate"),
    "OTOC": ("scpn_quantum_control.analysis.otoc", "OTOC"),
    "OTOCResult": ("scpn_quantum_control.analysis.otoc", "OTOCResult"),
    "compute_otoc": ("scpn_quantum_control.analysis.otoc", "compute_otoc"),
    "OTOCSyncScanResult": ("scpn_quantum_control.analysis.otoc_sync_probe", "OTOCSyncScanResult"),
    "otoc_sync_scan": ("scpn_quantum_control.analysis.otoc_sync_probe", "otoc_sync_scan"),
    "P_H1_Derivation": ("scpn_quantum_control.analysis.p_h1_derivation", "P_H1_Derivation"),
    "derive_p_h1": ("scpn_quantum_control.analysis.p_h1_derivation", "derive_p_h1"),
    "P_H1_OPEN_CLAIM_BOUNDARY": (
        "scpn_quantum_control.analysis.p_h1_open_guard",
        "P_H1_OPEN_CLAIM_BOUNDARY",
    ),
    "P_H1_OPEN_GUARD_SCHEMA": (
        "scpn_quantum_control.analysis.p_h1_open_guard",
        "P_H1_OPEN_GUARD_SCHEMA",
    ),
    "P_H1OpenGuardReport": (
        "scpn_quantum_control.analysis.p_h1_open_guard",
        "P_H1OpenGuardReport",
    ),
    "P_H1OpenGuardViolation": (
        "scpn_quantum_control.analysis.p_h1_open_guard",
        "P_H1OpenGuardViolation",
    ),
    "public_markdown_paths": (
        "scpn_quantum_control.analysis.p_h1_open_guard",
        "public_markdown_paths",
    ),
    "run_p_h1_open_guard": (
        "scpn_quantum_control.analysis.p_h1_open_guard",
        "run_p_h1_open_guard",
    ),
    "validate_p_h1_open_claim_text": (
        "scpn_quantum_control.analysis.p_h1_open_guard",
        "validate_p_h1_open_claim_text",
    ),
    "PairingResult": ("scpn_quantum_control.analysis.pairing_correlator", "PairingResult"),
    "pairing_vs_anisotropy": (
        "scpn_quantum_control.analysis.pairing_correlator",
        "pairing_vs_anisotropy",
    ),
    "PhaseBoundary": ("scpn_quantum_control.analysis.phase_diagram", "PhaseBoundary"),
    "PhaseDiagramResult": ("scpn_quantum_control.analysis.phase_diagram", "PhaseDiagramResult"),
    "compute_phase_diagram": (
        "scpn_quantum_control.analysis.phase_diagram",
        "compute_phase_diagram",
    ),
    "critical_coupling_finite_graph": (
        "scpn_quantum_control.analysis.phase_diagram",
        "critical_coupling_finite_graph",
    ),
    "critical_coupling_mean_field": (
        "scpn_quantum_control.analysis.phase_diagram",
        "critical_coupling_mean_field",
    ),
    "decoherence_temperature": (
        "scpn_quantum_control.analysis.phase_diagram",
        "decoherence_temperature",
    ),
    "effective_temperature": (
        "scpn_quantum_control.analysis.phase_diagram",
        "effective_temperature",
    ),
    "order_parameter_steady_state": (
        "scpn_quantum_control.analysis.phase_diagram",
        "order_parameter_steady_state",
    ),
    "QFIResult": ("scpn_quantum_control.analysis.qfi", "QFIResult"),
    "compute_qfi": ("scpn_quantum_control.analysis.qfi", "compute_qfi"),
    "qfi_gap_tradeoff": ("scpn_quantum_control.analysis.qfi", "qfi_gap_tradeoff"),
    "QFICriticalityResult": (
        "scpn_quantum_control.analysis.qfi_criticality",
        "QFICriticalityResult",
    ),
    "qfi_vs_coupling": ("scpn_quantum_control.analysis.qfi_criticality", "qfi_vs_coupling"),
    "QFIGeometricCrosscheck": (
        "scpn_quantum_control.analysis.qfi_geometric_crosscheck",
        "QFIGeometricCrosscheck",
    ),
    "crosscheck_qfi_geometric": (
        "scpn_quantum_control.analysis.qfi_geometric_crosscheck",
        "crosscheck_qfi_geometric",
    ),
    "QRCPhaseResult": ("scpn_quantum_control.analysis.qrc_phase_detector", "QRCPhaseResult"),
    "qrc_phase_detection": (
        "scpn_quantum_control.analysis.qrc_phase_detector",
        "qrc_phase_detection",
    ),
    "QuantumFisherInformation": (
        "scpn_quantum_control.analysis.quantum_fisher_information",
        "QuantumFisherInformation",
    ),
    "MpembaResult": ("scpn_quantum_control.analysis.quantum_mpemba", "MpembaResult"),
    "mpemba_experiment": ("scpn_quantum_control.analysis.quantum_mpemba", "mpemba_experiment"),
    "QuantumPHResult": (
        "scpn_quantum_control.analysis.quantum_persistent_homology",
        "QuantumPHResult",
    ),
    "ph_sync_scan": ("scpn_quantum_control.analysis.quantum_persistent_homology", "ph_sync_scan"),
    "PhiResult": ("scpn_quantum_control.analysis.quantum_phi", "PhiResult"),
    "compute_quantum_phi": ("scpn_quantum_control.analysis.quantum_phi", "compute_quantum_phi"),
    "phi_vs_coupling_scan": ("scpn_quantum_control.analysis.quantum_phi", "phi_vs_coupling_scan"),
    "von_neumann_entropy": ("scpn_quantum_control.analysis.quantum_phi", "von_neumann_entropy"),
    "RESEARCH_LANE_REGISTRY_BOUNDARY": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "RESEARCH_LANE_REGISTRY_BOUNDARY",
    ),
    "RESEARCH_LANE_REGISTRY_SCHEMA": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "RESEARCH_LANE_REGISTRY_SCHEMA",
    ),
    "ResearchLaneClaimStatus": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "ResearchLaneClaimStatus",
    ),
    "ResearchLaneDiffHook": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "ResearchLaneDiffHook",
    ),
    "ResearchLaneInventoryReport": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "ResearchLaneInventoryReport",
    ),
    "ResearchLaneMaturity": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "ResearchLaneMaturity",
    ),
    "ResearchLaneRecord": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "ResearchLaneRecord",
    ),
    "ResearchLaneRegistryReport": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "ResearchLaneRegistryReport",
    ),
    "assert_research_lane_inventory": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "assert_research_lane_inventory",
    ),
    "build_research_lane_registry_report": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "build_research_lane_registry_report",
    ),
    "discover_research_lane_modules": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "discover_research_lane_modules",
    ),
    "get_research_lane": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "get_research_lane",
    ),
    "list_research_lanes": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "list_research_lanes",
    ),
    "render_research_lane_registry_markdown": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "render_research_lane_registry_markdown",
    ),
    "validate_research_lane_inventory": (
        "scpn_quantum_control.analysis.research_lane_registry",
        "validate_research_lane_inventory",
    ),
    "RLDiscoveryAgent": ("scpn_quantum_control.analysis.rl_discovery_agent", "RLDiscoveryAgent"),
    "RLPulseOptimizer": ("scpn_quantum_control.analysis.rl_pulse_optimizer", "RLPulseOptimizer"),
    "DEFAULT_RL_RESEARCH_SEEDS": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "DEFAULT_RL_RESEARCH_SEEDS",
    ),
    "RL_DENSE_REWARD_CONTRACT": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "RL_DENSE_REWARD_CONTRACT",
    ),
    "RL_ENVIRONMENT_API": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "RL_ENVIRONMENT_API",
    ),
    "RL_RESEARCH_CLAIM_BOUNDARY": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "RL_RESEARCH_CLAIM_BOUNDARY",
    ),
    "RL_RESEARCH_GOVERNANCE_SCHEMA": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "RL_RESEARCH_GOVERNANCE_SCHEMA",
    ),
    "RLResearchDecision": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "RLResearchDecision",
    ),
    "RLResearchGovernanceError": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "RLResearchGovernanceError",
    ),
    "RLResearchLane": ("scpn_quantum_control.analysis.rl_research_governance", "RLResearchLane"),
    "RLResearchPolicy": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "RLResearchPolicy",
    ),
    "RLSeedEvaluation": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "RLSeedEvaluation",
    ),
    "RLSeedSuiteReport": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "RLSeedSuiteReport",
    ),
    "assert_rl_research_allowed": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "assert_rl_research_allowed",
    ),
    "assess_rl_research": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "assess_rl_research",
    ),
    "build_rl_research_evidence_report": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "build_rl_research_evidence_report",
    ),
    "build_witness_seed_suite": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "build_witness_seed_suite",
    ),
    "estimate_witness_evaluation_budget": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "estimate_witness_evaluation_budget",
    ),
    "render_rl_research_evidence_markdown": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "render_rl_research_evidence_markdown",
    ),
    "run_governed_witness_seed_suite": (
        "scpn_quantum_control.analysis.rl_research_governance",
        "run_governed_witness_seed_suite",
    ),
    "CRITICALITY_TAIL_SCHEMA": (
        "scpn_quantum_control.analysis.sensing",
        "CRITICALITY_TAIL_SCHEMA",
    ),
    "QUANTUM_SENSING_SCHEMA": ("scpn_quantum_control.analysis.sensing", "QUANTUM_SENSING_SCHEMA"),
    "CriticalitySensingTail": ("scpn_quantum_control.analysis.sensing", "CriticalitySensingTail"),
    "QuantumSensingReadinessConfig": (
        "scpn_quantum_control.analysis.sensing",
        "QuantumSensingReadinessConfig",
    ),
    "SensingGainRow": ("scpn_quantum_control.analysis.sensing", "SensingGainRow"),
    "SensingGainScan": ("scpn_quantum_control.analysis.sensing", "SensingGainScan"),
    "metrological_gain_vs_k": ("scpn_quantum_control.analysis.sensing", "metrological_gain_vs_k"),
    "optimal_sensing_k": ("scpn_quantum_control.analysis.sensing", "optimal_sensing_k"),
    "qfi_criticality_sensing_tail": (
        "scpn_quantum_control.analysis.sensing",
        "qfi_criticality_sensing_tail",
    ),
    "quantum_sensing_markdown": (
        "scpn_quantum_control.analysis.sensing",
        "quantum_sensing_markdown",
    ),
    "quantum_sensing_payload": (
        "scpn_quantum_control.analysis.sensing",
        "quantum_sensing_payload",
    ),
    "ShadowResult": ("scpn_quantum_control.analysis.shadow_tomography", "ShadowResult"),
    "classical_shadow_estimation": (
        "scpn_quantum_control.analysis.shadow_tomography",
        "classical_shadow_estimation",
    ),
    "SFFResult": ("scpn_quantum_control.analysis.spectral_form_factor", "SFFResult"),
    "sff_vs_coupling": ("scpn_quantum_control.analysis.spectral_form_factor", "sff_vs_coupling"),
    "EntanglementWitnessResult": (
        "scpn_quantum_control.analysis.sync_entanglement_witness",
        "EntanglementWitnessResult",
    ),
    "R_from_statevector": (
        "scpn_quantum_control.analysis.sync_entanglement_witness",
        "R_from_statevector",
    ),
    "R_separable_bound": (
        "scpn_quantum_control.analysis.sync_entanglement_witness",
        "R_separable_bound",
    ),
    "SyncOrderParameter": (
        "scpn_quantum_control.analysis.sync_order_parameter",
        "SyncOrderParameter",
    ),
    "UncertaintyInterval": (
        "scpn_quantum_control.analysis.sync_uncertainty",
        "UncertaintyInterval",
    ),
    "metric_bootstrap": ("scpn_quantum_control.analysis.sync_uncertainty", "metric_bootstrap"),
    "order_parameter_bootstrap": (
        "scpn_quantum_control.analysis.sync_uncertainty",
        "order_parameter_bootstrap",
    ),
    "order_parameter_estimate": (
        "scpn_quantum_control.analysis.sync_uncertainty",
        "order_parameter_estimate",
    ),
    "order_parameter_shot_noise": (
        "scpn_quantum_control.analysis.sync_uncertainty",
        "order_parameter_shot_noise",
    ),
    "WitnessResult": ("scpn_quantum_control.analysis.sync_witness", "WitnessResult"),
    "evaluate_all_witnesses": (
        "scpn_quantum_control.analysis.sync_witness",
        "evaluate_all_witnesses",
    ),
    "TCBOWeightedComplexResult": (
        "scpn_quantum_control.analysis.tcbo_weighted_complex",
        "TCBOWeightedComplexResult",
    ),
    "TCBOWeightedThresholdScan": (
        "scpn_quantum_control.analysis.tcbo_weighted_complex",
        "TCBOWeightedThresholdScan",
    ),
    "coupling_weighted_edge_matrix": (
        "scpn_quantum_control.analysis.tcbo_weighted_complex",
        "coupling_weighted_edge_matrix",
    ),
    "tcbo_weighted_complex": (
        "scpn_quantum_control.analysis.tcbo_weighted_complex",
        "tcbo_weighted_complex",
    ),
    "tcbo_weighted_threshold_scan": (
        "scpn_quantum_control.analysis.tcbo_weighted_complex",
        "tcbo_weighted_threshold_scan",
    ),
    "THEORY_HOOK_PROMOTION_BOUNDARY": (
        "scpn_quantum_control.analysis.theory_hook_promotion",
        "THEORY_HOOK_PROMOTION_BOUNDARY",
    ),
    "THEORY_HOOK_PROMOTION_SCHEMA": (
        "scpn_quantum_control.analysis.theory_hook_promotion",
        "THEORY_HOOK_PROMOTION_SCHEMA",
    ),
    "TheoryHookEvidenceRecord": (
        "scpn_quantum_control.analysis.theory_hook_promotion",
        "TheoryHookEvidenceRecord",
    ),
    "TheoryHookPromotionRecord": (
        "scpn_quantum_control.analysis.theory_hook_promotion",
        "TheoryHookPromotionRecord",
    ),
    "TheoryHookPromotionReport": (
        "scpn_quantum_control.analysis.theory_hook_promotion",
        "TheoryHookPromotionReport",
    ),
    "TheoryHookRole": ("scpn_quantum_control.analysis.theory_hook_promotion", "TheoryHookRole"),
    "TheoryHookStatus": (
        "scpn_quantum_control.analysis.theory_hook_promotion",
        "TheoryHookStatus",
    ),
    "TheoryHookTier": ("scpn_quantum_control.analysis.theory_hook_promotion", "TheoryHookTier"),
    "build_theory_hook_promotion_report": (
        "scpn_quantum_control.analysis.theory_hook_promotion",
        "build_theory_hook_promotion_report",
    ),
    "get_theory_hook_promotion": (
        "scpn_quantum_control.analysis.theory_hook_promotion",
        "get_theory_hook_promotion",
    ),
    "list_theory_hook_promotions": (
        "scpn_quantum_control.analysis.theory_hook_promotion",
        "list_theory_hook_promotions",
    ),
    "render_theory_hook_promotion_markdown": (
        "scpn_quantum_control.analysis.theory_hook_promotion",
        "render_theory_hook_promotion_markdown",
    ),
    "run_theory_hook_evidence": (
        "scpn_quantum_control.analysis.theory_hook_promotion",
        "run_theory_hook_evidence",
    ),
    "ThermodynamicWitness": (
        "scpn_quantum_control.analysis.thermodynamic_witness",
        "ThermodynamicWitness",
    ),
    "VortexBindingResult": ("scpn_quantum_control.analysis.vortex_binding", "VortexBindingResult"),
    "compute_vortex_binding": (
        "scpn_quantum_control.analysis.vortex_binding",
        "compute_vortex_binding",
    ),
    "WitnessCandidate": ("scpn_quantum_control.analysis.witness_discovery", "WitnessCandidate"),
    "WitnessDiscoveryEvaluation": (
        "scpn_quantum_control.analysis.witness_discovery",
        "WitnessDiscoveryEvaluation",
    ),
    "WitnessDiscoveryResult": (
        "scpn_quantum_control.analysis.witness_discovery",
        "WitnessDiscoveryResult",
    ),
    "WitnessDiscoverySpec": (
        "scpn_quantum_control.analysis.witness_discovery",
        "WitnessDiscoverySpec",
    ),
    "WitnessSearchMode": ("scpn_quantum_control.analysis.witness_discovery", "WitnessSearchMode"),
    "discover_kuramoto_witnesses": (
        "scpn_quantum_control.analysis.witness_discovery",
        "discover_kuramoto_witnesses",
    ),
    "score_witness_candidates": (
        "scpn_quantum_control.analysis.witness_discovery",
        "score_witness_candidates",
    ),
    "AnisotropyScanResult": (
        "scpn_quantum_control.analysis.xxz_phase_diagram",
        "AnisotropyScanResult",
    ),
    "anisotropy_phase_diagram": (
        "scpn_quantum_control.analysis.xxz_phase_diagram",
        "anisotropy_phase_diagram",
    ),
}


class _ExportModule(ModuleType):
    """Keep declared object exports when Python publishes child modules."""

    def __setattr__(self, name: str, value: object) -> None:
        """Publish a module attribute while preserving same-named exports.

        Parameters
        ----------
        name
            Attribute assigned by the import system or a caller.
        value
            Value to publish in the package namespace.

        """
        target = _PUBLIC_EXPORTS.get(name)
        if (
            target is not None
            and target[1] is not None
            and isinstance(value, ModuleType)
            and value.__name__ in {target[0], f"{__name__}.{name}"}
        ):
            value = __getattr__(name)
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _ExportModule


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
    "DLAParityWitness",
    "SyncOrderParameter",
    "UncertaintyInterval",
    "metric_bootstrap",
    "order_parameter_bootstrap",
    "order_parameter_estimate",
    "order_parameter_shot_noise",
    "IntegratedInformationPhi",
    "QuantumFisherInformation",
    "ThermodynamicWitness",
    "LogicalSyncWitness",
    "RLDiscoveryAgent",
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
    "dla_truncated_tn",
    "RLPulseOptimizer",
    "OTOCResult",
    "OTOC",
    "compute_otoc",
    "P_H1_Derivation",
    "derive_p_h1",
    "P_H1_OPEN_CLAIM_BOUNDARY",
    "P_H1_OPEN_GUARD_SCHEMA",
    "P_H1OpenGuardReport",
    "P_H1OpenGuardViolation",
    "public_markdown_paths",
    "run_p_h1_open_guard",
    "validate_p_h1_open_claim_text",
    "PhaseBoundary",
    "PhaseDiagramResult",
    "compute_phase_diagram",
    "critical_coupling_finite_graph",
    "critical_coupling_mean_field",
    "decoherence_temperature",
    "effective_temperature",
    "order_parameter_steady_state",
    "QFIResult",
    "QFIGeometricCrosscheck",
    "compute_qfi",
    "crosscheck_qfi_geometric",
    "qfi_gap_tradeoff",
    "PhiResult",
    "compute_quantum_phi",
    "phi_vs_coupling_scan",
    "von_neumann_entropy",
    "RESEARCH_LANE_REGISTRY_BOUNDARY",
    "RESEARCH_LANE_REGISTRY_SCHEMA",
    "ResearchLaneClaimStatus",
    "ResearchLaneDiffHook",
    "ResearchLaneInventoryReport",
    "ResearchLaneMaturity",
    "ResearchLaneRecord",
    "ResearchLaneRegistryReport",
    "assert_research_lane_inventory",
    "build_research_lane_registry_report",
    "discover_research_lane_modules",
    "get_research_lane",
    "list_research_lanes",
    "render_research_lane_registry_markdown",
    "validate_research_lane_inventory",
    "DEFAULT_RL_RESEARCH_SEEDS",
    "RL_DENSE_REWARD_CONTRACT",
    "RL_ENVIRONMENT_API",
    "RL_RESEARCH_CLAIM_BOUNDARY",
    "RL_RESEARCH_GOVERNANCE_SCHEMA",
    "RLResearchDecision",
    "RLResearchGovernanceError",
    "RLResearchLane",
    "RLResearchPolicy",
    "RLSeedEvaluation",
    "RLSeedSuiteReport",
    "assert_rl_research_allowed",
    "assess_rl_research",
    "build_rl_research_evidence_report",
    "build_witness_seed_suite",
    "estimate_witness_evaluation_budget",
    "render_rl_research_evidence_markdown",
    "run_governed_witness_seed_suite",
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
    "InitialState",
    "InitialStateControlComparison",
    "SyncTrajectory",
    "compare_all_initial_states",
    "compare_initial_states_with_dephased_controls",
    "entanglement_advantage",
    "local_phase_observables",
    "mean_single_qubit_linear_entropy",
    "prepare_initial_state",
    "simulate_sync_trajectory",
    "transverse_exchange_coherence",
    "entanglement_vs_coupling",
    "PercolationScanResult",
    "percolation_scan",
    "FSSResult",
    "FSSFitDiagnostics",
    "compute_finite_size_scaling",
    "SpectrumSummary",
    "add_fim_feedback",
    "adjacent_gap_ratio",
    "bipartite_entropy_from_statevector",
    "commutator_frobenius_norm_with_diagonal",
    "computational_magnetisations",
    "fim_diagonal",
    "magnetisation_operator_diagonal",
    "magnetisation_sector_indices",
    "sector_coupling_rows",
    "sector_spectrum_rows",
    "summarise_spectrum",
    "ADAPTIVE_FIM_CLAIM_BOUNDARY",
    "ADAPTIVE_FIM_SCHEMA",
    "AdaptiveFIMConfig",
    "AdaptiveFIMObserverRecord",
    "AdaptiveFIMPlan",
    "AdaptiveFIMStep",
    "BinomialInterval",
    "FIMWitness",
    "adaptive_count_aware_schedule",
    "adaptive_lambda_schedule",
    "observer_record_from_step",
    "plan_adaptive_fim_schedule",
    "propose_count_aware_lambda",
    "propose_next_lambda",
    "wilson_score_interval",
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
    "CRITICALITY_TAIL_SCHEMA",
    "QUANTUM_SENSING_SCHEMA",
    "CriticalitySensingTail",
    "QuantumSensingReadinessConfig",
    "SensingGainRow",
    "SensingGainScan",
    "metrological_gain_vs_k",
    "optimal_sensing_k",
    "qfi_criticality_sensing_tail",
    "quantum_sensing_markdown",
    "quantum_sensing_payload",
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
    "WitnessCandidate",
    "WitnessDiscoveryEvaluation",
    "WitnessDiscoveryResult",
    "WitnessDiscoverySpec",
    "WitnessSearchMode",
    "discover_kuramoto_witnesses",
    "score_witness_candidates",
    "TCBOWeightedComplexResult",
    "TCBOWeightedThresholdScan",
    "coupling_weighted_edge_matrix",
    "tcbo_weighted_complex",
    "tcbo_weighted_threshold_scan",
    "THEORY_HOOK_PROMOTION_BOUNDARY",
    "THEORY_HOOK_PROMOTION_SCHEMA",
    "TheoryHookEvidenceRecord",
    "TheoryHookPromotionRecord",
    "TheoryHookPromotionReport",
    "TheoryHookRole",
    "TheoryHookStatus",
    "TheoryHookTier",
    "build_theory_hook_promotion_report",
    "get_theory_hook_promotion",
    "list_theory_hook_promotions",
    "render_theory_hook_promotion_markdown",
    "run_theory_hook_evidence",
    "AnisotropyScanResult",
    "anisotropy_phase_diagram",
]
