# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Classical-Quantum Bridge
from .control_plasma_knm import (
    build_knm_plasma,
    build_knm_plasma_from_config,
    build_knm_plasma_spec,
    plasma_omega,
)
from .knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    build_kuramoto_ring,
    knm_to_ansatz,
    knm_to_hamiltonian,
)
from .orchestrator_adapter import PhaseOrchestratorAdapter
from .phase_artifact import LayerStateArtifact, LockSignatureArtifact, UPDEPhaseArtifact
from .sc_to_quantum import (
    angle_to_probability,
    bitstream_to_statevector,
    measurement_to_bitstream,
    probability_to_angle,
)
from .snn_adapter import SNNQuantumBridge, quantum_measurement_to_current, spike_train_to_rotations
from .spn_to_qcircuit import inhibitor_anti_control, spn_to_circuit
from .ssgf_adapter import quantum_to_ssgf_state, ssgf_state_to_quantum, ssgf_w_to_hamiltonian

__all__ = [
    "knm_to_hamiltonian",
    "knm_to_ansatz",
    "OMEGA_N_16",
    "build_knm_paper27",
    "build_kuramoto_ring",
    "build_knm_plasma",
    "build_knm_plasma_spec",
    "build_knm_plasma_from_config",
    "plasma_omega",
    "LockSignatureArtifact",
    "LayerStateArtifact",
    "UPDEPhaseArtifact",
    "PhaseOrchestratorAdapter",
    "probability_to_angle",
    "angle_to_probability",
    "bitstream_to_statevector",
    "measurement_to_bitstream",
    "spn_to_circuit",
    "inhibitor_anti_control",
    "SNNQuantumBridge",
    "spike_train_to_rotations",
    "quantum_measurement_to_current",
    "ssgf_w_to_hamiltonian",
    "ssgf_state_to_quantum",
    "quantum_to_ssgf_state",
]
