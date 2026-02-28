"""Quantum-native reformulations of SCPN spiking, phase dynamics, and plasma control."""

from .bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27, knm_to_hamiltonian
from .control.q_disruption import QuantumDisruptionClassifier
from .control.qaoa_mpc import QAOA_MPC
from .control.qpetri import QuantumPetriNet
from .control.vqls_gs import VQLS_GradShafranov
from .hardware.runner import HardwareRunner, JobResult
from .mitigation.zne import ZNEResult, gate_fold_circuit, zne_extrapolate
from .phase.phase_vqe import PhaseVQE
from .phase.trotter_upde import QuantumUPDESolver
from .phase.xy_kuramoto import QuantumKuramotoSolver
from .qec.control_qec import ControlQEC
from .qsnn.qlayer import QuantumDenseLayer
from .qsnn.qlif import QuantumLIFNeuron
from .qsnn.qstdp import QuantumSTDP
from .qsnn.qsynapse import QuantumSynapse

__version__ = "0.4.0"

__all__ = [
    "OMEGA_N_16",
    "build_knm_paper27",
    "knm_to_hamiltonian",
    "QuantumKuramotoSolver",
    "QuantumUPDESolver",
    "PhaseVQE",
    "HardwareRunner",
    "JobResult",
    "QuantumDisruptionClassifier",
    "QAOA_MPC",
    "QuantumPetriNet",
    "VQLS_GradShafranov",
    "ControlQEC",
    "QuantumLIFNeuron",
    "QuantumSynapse",
    "QuantumSTDP",
    "QuantumDenseLayer",
    "ZNEResult",
    "gate_fold_circuit",
    "zne_extrapolate",
]
