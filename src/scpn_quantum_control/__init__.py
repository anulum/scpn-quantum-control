"""Quantum-native reformulations of SCPN spiking, phase dynamics, and plasma control."""

from .bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27, knm_to_hamiltonian
from .hardware.runner import HardwareRunner, JobResult
from .phase.phase_vqe import PhaseVQE
from .phase.trotter_upde import QuantumUPDESolver
from .phase.xy_kuramoto import QuantumKuramotoSolver

__version__ = "0.2.1"

__all__ = [
    "OMEGA_N_16",
    "build_knm_paper27",
    "knm_to_hamiltonian",
    "QuantumKuramotoSolver",
    "QuantumUPDESolver",
    "PhaseVQE",
    "HardwareRunner",
    "JobResult",
]
