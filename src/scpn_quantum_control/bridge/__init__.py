# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Classical-Quantum Bridge
"""Classical-Quantum bridge for mapping oscillator networks to XY Hamiltonians.

Provides Kuramoto coupling matrix (K_nm) construction, Hamiltonian compilation,
plasma control adapters, SPN-to-circuit translation, SNN bridging, and SSGF
state conversion.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .control_plasma_knm import (
        build_knm_plasma,
        build_knm_plasma_from_config,
        build_knm_plasma_spec,
        plasma_omega,
    )
    from .fusion_core_frc import (
        FRCEquilibriumLike,
        FusionCoreFRCCalibration,
        calibrate_frc_surrogate_from_equilibrium,
        calibrate_frc_surrogate_from_inputs,
    )
    from .knm_hamiltonian import (
        OMEGA_N_16,
        build_knm_paper27,
        build_kuramoto_ring,
        knm_to_ansatz,
        knm_to_dense_matrix,
        knm_to_hamiltonian,
        knm_to_sparse_matrix,
        knm_to_xxz_hamiltonian,
        omega_for_oscillators,
    )
    from .orchestrator_adapter import PhaseOrchestratorAdapter
    from .phase_artifact import LayerStateArtifact, LockSignatureArtifact, UPDEPhaseArtifact
    from .qpu_data_artifact import (
        QPUDataArtifact,
        artifact_from_arrays,
        artifact_to_kuramoto_problem,
        read_qpu_data_artifact,
        validate_qpu_data_artifact,
        write_qpu_data_artifact,
    )
    from .sc_to_quantum import (
        angle_to_probability,
        bitstream_to_statevector,
        measurement_to_bitstream,
        probability_to_angle,
    )
    from .scpn_upde_edge import (
        PAPER27_PROVISIONAL_BOUNDARY,
        SCPN_UPDE_EDGE_SCHEMA,
        SCPN_UPDE_SCOPE_ENVELOPE,
        SCPNUPDEEdge,
        build_paper27_scpn_upde_edge,
        build_scpn_upde_edge,
        edge_content_digest,
        validate_scpn_upde_edge_payload,
    )
    from .snn_adapter import (
        SNNQuantumBridge,
        quantum_measurement_to_current,
        spike_train_to_rotations,
    )
    from .spn_to_qcircuit import inhibitor_anti_control, spn_to_circuit
    from .ssgf_adapter import quantum_to_ssgf_state, ssgf_state_to_quantum, ssgf_w_to_hamiltonian

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "build_knm_plasma": ("scpn_quantum_control.bridge.control_plasma_knm", "build_knm_plasma"),
    "build_knm_plasma_from_config": (
        "scpn_quantum_control.bridge.control_plasma_knm",
        "build_knm_plasma_from_config",
    ),
    "build_knm_plasma_spec": (
        "scpn_quantum_control.bridge.control_plasma_knm",
        "build_knm_plasma_spec",
    ),
    "plasma_omega": ("scpn_quantum_control.bridge.control_plasma_knm", "plasma_omega"),
    "FRCEquilibriumLike": ("scpn_quantum_control.bridge.fusion_core_frc", "FRCEquilibriumLike"),
    "FusionCoreFRCCalibration": (
        "scpn_quantum_control.bridge.fusion_core_frc",
        "FusionCoreFRCCalibration",
    ),
    "calibrate_frc_surrogate_from_equilibrium": (
        "scpn_quantum_control.bridge.fusion_core_frc",
        "calibrate_frc_surrogate_from_equilibrium",
    ),
    "calibrate_frc_surrogate_from_inputs": (
        "scpn_quantum_control.bridge.fusion_core_frc",
        "calibrate_frc_surrogate_from_inputs",
    ),
    "OMEGA_N_16": ("scpn_quantum_control.bridge.knm_hamiltonian", "OMEGA_N_16"),
    "build_knm_paper27": ("scpn_quantum_control.bridge.knm_hamiltonian", "build_knm_paper27"),
    "build_kuramoto_ring": ("scpn_quantum_control.bridge.knm_hamiltonian", "build_kuramoto_ring"),
    "knm_to_ansatz": ("scpn_quantum_control.bridge.knm_hamiltonian", "knm_to_ansatz"),
    "knm_to_dense_matrix": ("scpn_quantum_control.bridge.knm_hamiltonian", "knm_to_dense_matrix"),
    "knm_to_hamiltonian": ("scpn_quantum_control.bridge.knm_hamiltonian", "knm_to_hamiltonian"),
    "knm_to_sparse_matrix": (
        "scpn_quantum_control.bridge.knm_hamiltonian",
        "knm_to_sparse_matrix",
    ),
    "knm_to_xxz_hamiltonian": (
        "scpn_quantum_control.bridge.knm_hamiltonian",
        "knm_to_xxz_hamiltonian",
    ),
    "omega_for_oscillators": (
        "scpn_quantum_control.bridge.knm_hamiltonian",
        "omega_for_oscillators",
    ),
    "PhaseOrchestratorAdapter": (
        "scpn_quantum_control.bridge.orchestrator_adapter",
        "PhaseOrchestratorAdapter",
    ),
    "LayerStateArtifact": ("scpn_quantum_control.bridge.phase_artifact", "LayerStateArtifact"),
    "LockSignatureArtifact": (
        "scpn_quantum_control.bridge.phase_artifact",
        "LockSignatureArtifact",
    ),
    "UPDEPhaseArtifact": ("scpn_quantum_control.bridge.phase_artifact", "UPDEPhaseArtifact"),
    "QPUDataArtifact": ("scpn_quantum_control.bridge.qpu_data_artifact", "QPUDataArtifact"),
    "artifact_from_arrays": (
        "scpn_quantum_control.bridge.qpu_data_artifact",
        "artifact_from_arrays",
    ),
    "artifact_to_kuramoto_problem": (
        "scpn_quantum_control.bridge.qpu_data_artifact",
        "artifact_to_kuramoto_problem",
    ),
    "read_qpu_data_artifact": (
        "scpn_quantum_control.bridge.qpu_data_artifact",
        "read_qpu_data_artifact",
    ),
    "validate_qpu_data_artifact": (
        "scpn_quantum_control.bridge.qpu_data_artifact",
        "validate_qpu_data_artifact",
    ),
    "write_qpu_data_artifact": (
        "scpn_quantum_control.bridge.qpu_data_artifact",
        "write_qpu_data_artifact",
    ),
    "angle_to_probability": ("scpn_quantum_control.bridge.sc_to_quantum", "angle_to_probability"),
    "bitstream_to_statevector": (
        "scpn_quantum_control.bridge.sc_to_quantum",
        "bitstream_to_statevector",
    ),
    "measurement_to_bitstream": (
        "scpn_quantum_control.bridge.sc_to_quantum",
        "measurement_to_bitstream",
    ),
    "probability_to_angle": ("scpn_quantum_control.bridge.sc_to_quantum", "probability_to_angle"),
    "PAPER27_PROVISIONAL_BOUNDARY": (
        "scpn_quantum_control.bridge.scpn_upde_edge",
        "PAPER27_PROVISIONAL_BOUNDARY",
    ),
    "SCPN_UPDE_EDGE_SCHEMA": (
        "scpn_quantum_control.bridge.scpn_upde_edge",
        "SCPN_UPDE_EDGE_SCHEMA",
    ),
    "SCPN_UPDE_SCOPE_ENVELOPE": (
        "scpn_quantum_control.bridge.scpn_upde_edge",
        "SCPN_UPDE_SCOPE_ENVELOPE",
    ),
    "SCPNUPDEEdge": ("scpn_quantum_control.bridge.scpn_upde_edge", "SCPNUPDEEdge"),
    "build_paper27_scpn_upde_edge": (
        "scpn_quantum_control.bridge.scpn_upde_edge",
        "build_paper27_scpn_upde_edge",
    ),
    "build_scpn_upde_edge": ("scpn_quantum_control.bridge.scpn_upde_edge", "build_scpn_upde_edge"),
    "edge_content_digest": ("scpn_quantum_control.bridge.scpn_upde_edge", "edge_content_digest"),
    "validate_scpn_upde_edge_payload": (
        "scpn_quantum_control.bridge.scpn_upde_edge",
        "validate_scpn_upde_edge_payload",
    ),
    "SNNQuantumBridge": ("scpn_quantum_control.bridge.snn_adapter", "SNNQuantumBridge"),
    "quantum_measurement_to_current": (
        "scpn_quantum_control.bridge.snn_adapter",
        "quantum_measurement_to_current",
    ),
    "spike_train_to_rotations": (
        "scpn_quantum_control.bridge.snn_adapter",
        "spike_train_to_rotations",
    ),
    "inhibitor_anti_control": (
        "scpn_quantum_control.bridge.spn_to_qcircuit",
        "inhibitor_anti_control",
    ),
    "spn_to_circuit": ("scpn_quantum_control.bridge.spn_to_qcircuit", "spn_to_circuit"),
    "quantum_to_ssgf_state": ("scpn_quantum_control.bridge.ssgf_adapter", "quantum_to_ssgf_state"),
    "ssgf_state_to_quantum": ("scpn_quantum_control.bridge.ssgf_adapter", "ssgf_state_to_quantum"),
    "ssgf_w_to_hamiltonian": ("scpn_quantum_control.bridge.ssgf_adapter", "ssgf_w_to_hamiltonian"),
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
    "knm_to_hamiltonian",
    "knm_to_ansatz",
    "OMEGA_N_16",
    "omega_for_oscillators",
    "build_knm_paper27",
    "build_kuramoto_ring",
    "knm_to_dense_matrix",
    "knm_to_sparse_matrix",
    "knm_to_xxz_hamiltonian",
    "build_knm_plasma",
    "build_knm_plasma_spec",
    "build_knm_plasma_from_config",
    "plasma_omega",
    "FRCEquilibriumLike",
    "FusionCoreFRCCalibration",
    "calibrate_frc_surrogate_from_equilibrium",
    "calibrate_frc_surrogate_from_inputs",
    "LockSignatureArtifact",
    "LayerStateArtifact",
    "UPDEPhaseArtifact",
    "PAPER27_PROVISIONAL_BOUNDARY",
    "SCPNUPDEEdge",
    "SCPN_UPDE_EDGE_SCHEMA",
    "SCPN_UPDE_SCOPE_ENVELOPE",
    "build_paper27_scpn_upde_edge",
    "build_scpn_upde_edge",
    "edge_content_digest",
    "validate_scpn_upde_edge_payload",
    "QPUDataArtifact",
    "artifact_from_arrays",
    "artifact_to_kuramoto_problem",
    "read_qpu_data_artifact",
    "validate_qpu_data_artifact",
    "write_qpu_data_artifact",
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
