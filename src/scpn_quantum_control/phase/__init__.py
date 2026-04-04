# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Dynamics Solvers
"""Phase dynamics solvers: Kuramoto-XY Trotterisation, VQE ground-state search,
UPDE Trotter integration, Trotter error analysis, and ansatz benchmarking.
"""

from .adapt_vqe import ADAPTResult, adapt_vqe
from .adiabatic_preparation import AdiabaticResult, adiabatic_ramp
from .ansatz_bench import benchmark_ansatz, run_ansatz_benchmark
from .ansatz_methodology import AnsatzBenchmarkResult
from .avqds import AVQDSResult, avqds_simulate
from .cross_domain_transfer import TransferResult, build_systems, transfer_experiment
from .floquet_kuramoto import FloquetResult, floquet_evolve, scan_drive_amplitude
from .lindblad_engine import LindbladSyncEngine
from .phase_vqe import PhaseVQE
from .qsvt_evolution import QSVTResourceEstimate
from .structured_ansatz import build_structured_ansatz
from .trotter_error import trotter_error_norm, trotter_error_sweep
from .trotter_upde import QuantumUPDESolver
from .varqite import VarQITEResult, varqite_ground_state
from .xy_kuramoto import QuantumKuramotoSolver

__all__ = [
    "QuantumKuramotoSolver",
    "QuantumUPDESolver",
    "PhaseVQE",
    "trotter_error_norm",
    "trotter_error_sweep",
    "benchmark_ansatz",
    "run_ansatz_benchmark",
    "adapt_vqe",
    "ADAPTResult",
    "adiabatic_ramp",
    "AdiabaticResult",
    "avqds_simulate",
    "AVQDSResult",
    "varqite_ground_state",
    "VarQITEResult",
    "floquet_evolve",
    "scan_drive_amplitude",
    "FloquetResult",
    "build_structured_ansatz",
    "LindbladSyncEngine",
    "transfer_experiment",
    "build_systems",
    "TransferResult",
    "AnsatzBenchmarkResult",
    "QSVTResourceEstimate",
]
