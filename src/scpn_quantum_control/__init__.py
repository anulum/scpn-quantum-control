# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — NISQ Quantum Simulation of Coupled Oscillator Networks
"""NISQ quantum simulation of coupled Kuramoto oscillator networks via XY Hamiltonian mapping."""

# Subpackage-level access for new modules
from . import analysis, applications, fep, gauge, l16, pgbo, psi_field, ssgf, tcbo
from .applications.eeg_classification import eeg_plv_to_vqe, eeg_quantum_kernel
from .benchmarks.quantum_advantage import (
    AdvantageResult,
    classical_benchmark,
    estimate_crossover,
    quantum_benchmark,
    run_scaling_benchmark,
)
from .bridge.control_plasma_knm import (
    build_knm_plasma,
    build_knm_plasma_from_config,
    build_knm_plasma_spec,
    plasma_omega,
)
from .bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    build_kuramoto_ring,
    knm_to_ansatz,
    knm_to_hamiltonian,
)
from .bridge.orchestrator_adapter import PhaseOrchestratorAdapter
from .bridge.phase_artifact import LayerStateArtifact, LockSignatureArtifact, UPDEPhaseArtifact
from .bridge.snn_adapter import (
    ArcaneNeuronBridge,
    SNNQuantumBridge,
    quantum_measurement_to_current,
    spike_train_to_rotations,
)
from .bridge.ssgf_adapter import (
    SSGFQuantumLoop,
    quantum_to_ssgf_state,
    ssgf_state_to_quantum,
    ssgf_w_to_hamiltonian,
)
from .control.hardware_topological_optimizer import HardwareTopologicalOptimizer
from .control.q_disruption import QuantumDisruptionClassifier
from .control.q_disruption_iter import (
    DisruptionBenchmark,
    ITERFeatureSpec,
    from_fusion_core_shot,
    generate_synthetic_iter_data,
    normalize_iter_features,
)
from .control.qaoa_mpc import QAOA_MPC
from .control.qpetri import QuantumPetriNet
from .control.topological_optimizer import TopologicalCouplingOptimizer
from .control.vqls_gs import VQLS_GradShafranov
from .hardware.fast_classical import fast_sparse_evolution
from .hardware.runner import HardwareRunner, JobResult
from .hardware.trapped_ion import transpile_for_trapped_ion, trapped_ion_noise_model
from .identity.binding_spec import (
    ARCANE_SAPIENCE_SPEC,
    ORCHESTRATOR_MAPPING,
    build_identity_attractor,
    orchestrator_to_quantum_phases,
    quantum_to_orchestrator_phases,
    solve_identity,
)
from .identity.coherence_budget import coherence_budget, fidelity_at_depth
from .identity.entanglement_witness import chsh_from_statevector, disposition_entanglement_map
from .identity.ground_state import IdentityAttractor
from .identity.identity_key import identity_fingerprint, prove_identity, verify_identity
from .mitigation.compound_mitigation import compound_mitigate_pipeline
from .mitigation.pec import PECResult, pauli_twirl_decompose, pec_sample
from .mitigation.zne import ZNEResult, gate_fold_circuit, zne_extrapolate
from .phase.lindblad_engine import LindbladSyncEngine
from .phase.phase_vqe import PhaseVQE
from .phase.structured_ansatz import build_structured_ansatz
from .phase.trotter_upde import QuantumUPDESolver
from .phase.xy_kuramoto import QuantumKuramotoSolver
from .qec.biological_surface_code import BiologicalMWPMDecoder, BiologicalSurfaceCode
from .qec.control_qec import ControlQEC
from .qec.fault_tolerant import FaultTolerantUPDE, LogicalQubit
from .qec.multiscale_qec import (
    MultiscaleQECResult,
    QECLevel,
    build_multiscale_qec,
    concatenated_logical_rate,
)
from .qec.surface_code_upde import SurfaceCodeSpec, SurfaceCodeUPDE
from .qec.syndrome_flow import syndrome_flow_analysis
from .qsnn.dynamic_coupling import DynamicCouplingEngine
from .qsnn.qlayer import QuantumDenseLayer
from .qsnn.qlif import QuantumLIFNeuron
from .qsnn.qstdp import QuantumSTDP
from .qsnn.qsynapse import QuantumSynapse
from .qsnn.training import QSNNTrainer

__version__ = "0.9.5"

__all__ = [
    "eeg_plv_to_vqe",
    "eeg_quantum_kernel",
    "TopologicalCouplingOptimizer",
    "HardwareTopologicalOptimizer",
    "fast_sparse_evolution",
    "compound_mitigate_pipeline",
    "LindbladSyncEngine",
    "build_structured_ansatz",
    "BiologicalSurfaceCode",
    "BiologicalMWPMDecoder",
    "DynamicCouplingEngine",
    "OMEGA_N_16",
    "build_knm_paper27",
    "build_kuramoto_ring",
    "build_knm_plasma",
    "build_knm_plasma_spec",
    "build_knm_plasma_from_config",
    "plasma_omega",
    "knm_to_hamiltonian",
    "knm_to_ansatz",
    "LockSignatureArtifact",
    "LayerStateArtifact",
    "UPDEPhaseArtifact",
    "PhaseOrchestratorAdapter",
    "ArcaneNeuronBridge",
    "SNNQuantumBridge",
    "spike_train_to_rotations",
    "quantum_measurement_to_current",
    "SSGFQuantumLoop",
    "ssgf_w_to_hamiltonian",
    "ssgf_state_to_quantum",
    "quantum_to_ssgf_state",
    "QuantumKuramotoSolver",
    "QuantumUPDESolver",
    "PhaseVQE",
    "HardwareRunner",
    "JobResult",
    "trapped_ion_noise_model",
    "transpile_for_trapped_ion",
    "QuantumDisruptionClassifier",
    "DisruptionBenchmark",
    "ITERFeatureSpec",
    "from_fusion_core_shot",
    "generate_synthetic_iter_data",
    "normalize_iter_features",
    "QAOA_MPC",
    "QuantumPetriNet",
    "VQLS_GradShafranov",
    "ControlQEC",
    "FaultTolerantUPDE",
    "LogicalQubit",
    "SurfaceCodeSpec",
    "SurfaceCodeUPDE",
    "MultiscaleQECResult",
    "QECLevel",
    "build_multiscale_qec",
    "concatenated_logical_rate",
    "syndrome_flow_analysis",
    "QuantumLIFNeuron",
    "QuantumSynapse",
    "QuantumSTDP",
    "QuantumDenseLayer",
    "QSNNTrainer",
    "ZNEResult",
    "gate_fold_circuit",
    "zne_extrapolate",
    "PECResult",
    "pauli_twirl_decompose",
    "pec_sample",
    "IdentityAttractor",
    "ARCANE_SAPIENCE_SPEC",
    "ORCHESTRATOR_MAPPING",
    "build_identity_attractor",
    "orchestrator_to_quantum_phases",
    "quantum_to_orchestrator_phases",
    "solve_identity",
    "coherence_budget",
    "fidelity_at_depth",
    "chsh_from_statevector",
    "disposition_entanglement_map",
    "identity_fingerprint",
    "prove_identity",
    "verify_identity",
    "AdvantageResult",
    "classical_benchmark",
    "quantum_benchmark",
    "estimate_crossover",
    "run_scaling_benchmark",
    "analysis",
    "applications",
    "gauge",
    "ssgf",
    "tcbo",
    "pgbo",
    "psi_field",
    "fep",
    "l16",
]
