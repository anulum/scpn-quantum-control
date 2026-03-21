# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from .bkt_analysis import (
    BKTResult,
    bkt_analysis,
    coupling_laplacian,
    estimate_t_bkt,
    fiedler_eigenvalue,
    scan_synchronization_transition,
)
from .dynamical_lie_algebra import DLAResult, compute_dla
from .entanglement_spectrum import (
    EntanglementResult,
    entanglement_analysis,
    entropy_vs_coupling_scan,
)
from .koopman import KoopmanResult, koopman_analysis, koopman_to_hamiltonian
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

__all__ = [
    "BKTResult",
    "bkt_analysis",
    "coupling_laplacian",
    "estimate_t_bkt",
    "fiedler_eigenvalue",
    "scan_synchronization_transition",
    "DLAResult",
    "compute_dla",
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
    "KoopmanResult",
    "koopman_analysis",
    "koopman_to_hamiltonian",
    "EntanglementResult",
    "entanglement_analysis",
    "entropy_vs_coupling_scan",
    "PhiResult",
    "compute_quantum_phi",
    "phi_vs_coupling_scan",
    "von_neumann_entropy",
]
