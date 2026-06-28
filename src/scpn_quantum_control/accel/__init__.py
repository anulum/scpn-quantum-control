# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-language acceleration package
"""Multi-language acceleration chain.

Implements the rule codified in ``feedback_multi_language_accel.md``:
every compute function may have one or more acceleration backends
(Rust, Julia, Go, Mojo) with the *measured fastest* at the top of the
fallback chain. Python is always the final fallback.

Current tiers shipped from this package:

* :mod:`.julia` — Julia bindings via ``juliacall``. Activated on first
  successful import; the first Julia call incurs a one-off JIT cost
  (~20 s) that amortises across the process lifetime.

Tiers that exist elsewhere in the repo:

* Rust is shipped via the ``scpn_quantum_engine`` PyO3 wheel at the
  repo root; the dispatchers in this package forward to it by name.

Tiers tracked as future work (empty modules not shipped until a
compute-module actually needs them):

* ``.go`` — for standalone fan-out daemons, tracked under
  ``hardware/async_runner.py`` follow-ups.
* ``.mojo`` — for GPU / MLIR hot paths, not yet relevant at the
  qubit-counts we run today.
"""

from __future__ import annotations

from .daido_mean_field import (
    daido_mean_field_force,
    daido_mean_field_jacobian,
    last_daido_mean_field_force_tier_used,
    last_daido_mean_field_jacobian_tier_used,
)
from .daido_observables import (
    daido_order_parameter,
    daido_order_parameter_gradient,
    daido_order_parameter_hessian,
    last_daido_gradient_tier_used,
    last_daido_hessian_tier_used,
    last_daido_tier_used,
)
from .daido_phase import (
    daido_mode_phase,
    daido_mode_phase_gradient,
    daido_mode_phase_hessian,
    last_daido_mode_phase_gradient_tier_used,
    last_daido_mode_phase_hessian_tier_used,
    last_daido_mode_phase_tier_used,
)
from .diff_kuramoto_adaptive import (
    AdaptiveGradients,
    adaptive_state_sensitivity,
    adaptive_terminal_value_and_grad,
)
from .diff_kuramoto_dopri import (
    DopriTrajectory,
    kuramoto_dopri_trajectory,
    kuramoto_dopri_vjp,
)
from .diff_kuramoto_euler import (
    kuramoto_euler_trajectory,
    kuramoto_euler_vjp,
    last_kuramoto_euler_trajectory_tier_used,
    last_kuramoto_euler_vjp_tier_used,
)
from .diff_kuramoto_inertial import (
    InertialGradients,
    inertial_state_sensitivity,
    inertial_terminal_value_and_grad,
)
from .diff_kuramoto_rk4 import (
    kuramoto_rk4_trajectory,
    kuramoto_rk4_vjp,
    last_kuramoto_rk4_trajectory_tier_used,
    last_kuramoto_rk4_vjp_tier_used,
)
from .dispatcher import (
    MultiLangDispatcher,
    available_tiers,
    dispatch,
)
from .kuramoto_adaptive import (
    AdaptivePhaseForce,
    AdaptiveTrajectory,
    PlasticityRule,
    adaptive_vector_field,
    hebbian_adaptive_jacobian,
    hebbian_coupling_equilibrium,
    hebbian_plasticity_rate,
    integrate_adaptive_kuramoto,
)
from .kuramoto_chimera import (
    ChimeraDiagnostics,
    chimera_diagnostics,
    chimera_index,
    chimera_index_gradient,
    community_metastability,
    community_order_parameters,
    metastability_index,
    metastability_index_gradient,
)
from .kuramoto_clusters import (
    ClusterPartition,
    cluster_count,
    cluster_partition,
    phase_clusters,
)
from .kuramoto_coherence_matrix import (
    coherence_matrix,
    coherence_spectrum,
    leading_coherence_eigenvector,
    mean_coherence_matrix,
    phase_locking_matrix,
)
from .kuramoto_control import (
    TerminalObjective,
    coherence_objective,
    interaction_energy_objective,
    phase_target_objective,
    synchronisation_value_and_grad,
    terminal_objective_value,
    terminal_objective_value_and_grad,
)
from .kuramoto_coupling_design import (
    CouplingDesignResult,
    CouplingProjection,
    design_synchronising_coupling,
    optimise_coupling,
    symmetric_nonnegative_projection,
)
from .kuramoto_critical_coupling import (
    critical_coupling,
    gaussian_critical_coupling,
    gaussian_density,
    lorentzian_critical_coupling,
    lorentzian_density,
    lorentzian_order_parameter,
    synchronised_order_parameter,
)
from .kuramoto_delayed import (
    DelayedForce,
    DelayedTrajectory,
    delayed_mean_field_force,
    delayed_networked_force,
    integrate_delayed_kuramoto,
)
from .kuramoto_delayed_mean_field import (
    is_synchronised_branch_stable,
    stable_synchronised_frequencies,
    synchronised_branch_stability,
    synchronised_frequency_residual,
    synchronised_frequency_roots,
)
from .kuramoto_energy import (
    kuramoto_interaction_energy,
    kuramoto_interaction_energy_gradient,
    kuramoto_interaction_energy_hessian,
    last_kuramoto_interaction_energy_gradient_tier_used,
    last_kuramoto_interaction_energy_hessian_tier_used,
    last_kuramoto_interaction_energy_tier_used,
)
from .kuramoto_explosive_continuation import (
    ContinuationBranch,
    HysteresisLoop,
    MeanFieldForce,
    continuation_sweep,
    hysteresis_loop,
    triadic_hysteresis_loop,
)
from .kuramoto_frequency_order import (
    FrequencyOrder,
    effective_frequencies,
    frequency_locked_fraction,
    frequency_order_diagnostics,
    frequency_synchronisation_index,
    frequency_synchronisation_index_gradient,
)
from .kuramoto_heterogeneous import (
    CouplingTerm,
    heterogeneous_force,
    heterogeneous_force_components,
    heterogeneous_jacobian,
    hyperedge_term,
    pairwise_term,
    simplex_mean_field_term,
)
from .kuramoto_hyperedge import (
    hyperedge_force,
    hyperedge_jacobian,
)
from .kuramoto_inertial import (
    InertialTrajectory,
    PhaseForce,
    PhaseJacobian,
    PhasePotential,
    inertial_energy,
    inertial_jacobian,
    inertial_vector_field,
    integrate_inertial,
)
from .kuramoto_lyapunov import (
    lyapunov_spectrum,
    maximal_lyapunov_exponent,
)
from .kuramoto_mean_field import (
    last_mean_field_force_tier_used,
    last_mean_field_jacobian_tier_used,
    mean_field_force,
    mean_field_jacobian,
)
from .kuramoto_noisy import (
    NoisyKuramotoRun,
    StochasticForce,
    integrate_noisy_kuramoto,
    noisy_kuramoto_step,
)
from .kuramoto_noisy_mean_field import (
    FrequencyDensity,
    lorentzian_noisy_critical_coupling,
    noisy_critical_coupling,
    noisy_stationary_order_parameter,
)
from .kuramoto_ott_antonsen import (
    ott_antonsen_field,
    ott_antonsen_order_parameter,
    ott_antonsen_steady_state,
    ott_antonsen_terminal_order_parameter_value_and_grad,
    ott_antonsen_trajectory,
)
from .kuramoto_phase_information import (
    mutual_information_matrix,
    normalised_phase_entropy,
    pairwise_mutual_information,
    phase_entropy,
    phase_entropy_series,
)
from .kuramoto_pinning_control import (
    PinningDesignResult,
    design_pinning,
    pinning_coherence_value,
    pinning_coherence_value_and_grad,
)
from .kuramoto_ring_basins import (
    BasinEstimate,
    estimate_ring_basins,
    is_twisted_state_stable,
    ring_coupling_matrix,
    twisted_state,
    twisted_state_eigenvalues,
    winding_number,
)
from .kuramoto_simplex_mean_field import (
    simplex_mean_field_force,
    simplex_mean_field_jacobian,
)
from .kuramoto_stability_spectrum import (
    StabilitySpectrum,
    is_synchronisation_stable,
    stability_spectrum,
    synchronisation_rate,
)
from .kuramoto_system_id import (
    SystemIdentificationResult,
    learn_coupling,
    trajectory_match_value,
    trajectory_match_value_and_grad,
)
from .local_order import (
    last_local_order_parameter_jacobian_tier_used,
    last_local_order_parameter_tier_used,
    local_order_parameter,
    local_order_parameter_jacobian,
)
from .local_phase import (
    last_local_mean_phase_jacobian_tier_used,
    last_local_mean_phase_tier_used,
    local_mean_phase,
    local_mean_phase_jacobian,
)
from .mean_phase_observables import (
    last_mean_phase_gradient_tier_used,
    last_mean_phase_hessian_tier_used,
    last_mean_phase_tier_used,
    mean_phase,
    mean_phase_gradient,
    mean_phase_hessian,
)
from .networked_kuramoto import (
    last_networked_kuramoto_force_tier_used,
    last_networked_kuramoto_jacobian_tier_used,
    networked_kuramoto_force,
    networked_kuramoto_jacobian,
)
from .order_parameter_observables import (
    last_gradient_tier_used,
    last_hessian_tier_used,
    last_tier_used,
    order_parameter,
    order_parameter_gradient,
    order_parameter_hessian,
)
from .sakaguchi_kuramoto import (
    last_sakaguchi_force_tier_used,
    last_sakaguchi_jacobian_tier_used,
    sakaguchi_force,
    sakaguchi_jacobian,
)
from .sakaguchi_mean_field import (
    last_sakaguchi_mean_field_force_tier_used,
    last_sakaguchi_mean_field_jacobian_tier_used,
    sakaguchi_mean_field_force,
    sakaguchi_mean_field_jacobian,
)
from .triadic_mean_field import (
    last_triadic_mean_field_force_tier_used,
    last_triadic_mean_field_jacobian_tier_used,
    triadic_mean_field_force,
    triadic_mean_field_jacobian,
)

__all__ = [
    "MultiLangDispatcher",
    "available_tiers",
    "daido_mean_field_force",
    "daido_mean_field_jacobian",
    "last_daido_mean_field_force_tier_used",
    "last_daido_mean_field_jacobian_tier_used",
    "daido_mode_phase",
    "daido_mode_phase_gradient",
    "daido_mode_phase_hessian",
    "last_daido_mode_phase_gradient_tier_used",
    "last_daido_mode_phase_hessian_tier_used",
    "last_daido_mode_phase_tier_used",
    "daido_order_parameter",
    "daido_order_parameter_gradient",
    "daido_order_parameter_hessian",
    "dispatch",
    "TerminalObjective",
    "coherence_objective",
    "interaction_energy_objective",
    "phase_target_objective",
    "synchronisation_value_and_grad",
    "terminal_objective_value",
    "terminal_objective_value_and_grad",
    "CouplingDesignResult",
    "CouplingProjection",
    "design_synchronising_coupling",
    "optimise_coupling",
    "symmetric_nonnegative_projection",
    "PinningDesignResult",
    "design_pinning",
    "pinning_coherence_value",
    "pinning_coherence_value_and_grad",
    "SystemIdentificationResult",
    "learn_coupling",
    "trajectory_match_value",
    "trajectory_match_value_and_grad",
    "StabilitySpectrum",
    "is_synchronisation_stable",
    "stability_spectrum",
    "synchronisation_rate",
    "critical_coupling",
    "gaussian_critical_coupling",
    "gaussian_density",
    "lorentzian_critical_coupling",
    "lorentzian_density",
    "lorentzian_order_parameter",
    "synchronised_order_parameter",
    "ott_antonsen_field",
    "ott_antonsen_order_parameter",
    "ott_antonsen_steady_state",
    "ott_antonsen_terminal_order_parameter_value_and_grad",
    "ott_antonsen_trajectory",
    "lyapunov_spectrum",
    "maximal_lyapunov_exponent",
    "ContinuationBranch",
    "HysteresisLoop",
    "MeanFieldForce",
    "continuation_sweep",
    "hysteresis_loop",
    "triadic_hysteresis_loop",
    "InertialTrajectory",
    "PhaseForce",
    "PhaseJacobian",
    "PhasePotential",
    "inertial_energy",
    "inertial_jacobian",
    "inertial_vector_field",
    "integrate_inertial",
    "InertialGradients",
    "inertial_state_sensitivity",
    "inertial_terminal_value_and_grad",
    "AdaptiveGradients",
    "adaptive_state_sensitivity",
    "adaptive_terminal_value_and_grad",
    "BasinEstimate",
    "estimate_ring_basins",
    "is_twisted_state_stable",
    "ring_coupling_matrix",
    "twisted_state",
    "twisted_state_eigenvalues",
    "winding_number",
    "DopriTrajectory",
    "kuramoto_dopri_trajectory",
    "kuramoto_dopri_vjp",
    "kuramoto_euler_trajectory",
    "kuramoto_euler_vjp",
    "kuramoto_rk4_trajectory",
    "kuramoto_rk4_vjp",
    "last_kuramoto_rk4_trajectory_tier_used",
    "last_kuramoto_rk4_vjp_tier_used",
    "last_kuramoto_euler_trajectory_tier_used",
    "last_kuramoto_euler_vjp_tier_used",
    "kuramoto_interaction_energy",
    "kuramoto_interaction_energy_gradient",
    "kuramoto_interaction_energy_hessian",
    "last_kuramoto_interaction_energy_gradient_tier_used",
    "last_kuramoto_interaction_energy_hessian_tier_used",
    "last_kuramoto_interaction_energy_tier_used",
    "last_daido_gradient_tier_used",
    "last_daido_hessian_tier_used",
    "last_daido_tier_used",
    "last_gradient_tier_used",
    "last_hessian_tier_used",
    "last_local_order_parameter_jacobian_tier_used",
    "last_local_order_parameter_tier_used",
    "local_mean_phase",
    "local_mean_phase_jacobian",
    "last_local_mean_phase_tier_used",
    "last_local_mean_phase_jacobian_tier_used",
    "local_order_parameter",
    "local_order_parameter_jacobian",
    "last_mean_field_force_tier_used",
    "last_mean_field_jacobian_tier_used",
    "last_mean_phase_gradient_tier_used",
    "last_mean_phase_hessian_tier_used",
    "last_mean_phase_tier_used",
    "last_tier_used",
    "mean_field_force",
    "mean_field_jacobian",
    "NoisyKuramotoRun",
    "StochasticForce",
    "integrate_noisy_kuramoto",
    "noisy_kuramoto_step",
    "FrequencyDensity",
    "lorentzian_noisy_critical_coupling",
    "noisy_critical_coupling",
    "noisy_stationary_order_parameter",
    "DelayedForce",
    "DelayedTrajectory",
    "delayed_mean_field_force",
    "delayed_networked_force",
    "integrate_delayed_kuramoto",
    "is_synchronised_branch_stable",
    "stable_synchronised_frequencies",
    "synchronised_branch_stability",
    "synchronised_frequency_residual",
    "synchronised_frequency_roots",
    "AdaptivePhaseForce",
    "AdaptiveTrajectory",
    "PlasticityRule",
    "adaptive_vector_field",
    "hebbian_adaptive_jacobian",
    "hebbian_coupling_equilibrium",
    "hebbian_plasticity_rate",
    "integrate_adaptive_kuramoto",
    "simplex_mean_field_force",
    "simplex_mean_field_jacobian",
    "hyperedge_force",
    "hyperedge_jacobian",
    "CouplingTerm",
    "heterogeneous_force",
    "heterogeneous_force_components",
    "heterogeneous_jacobian",
    "hyperedge_term",
    "pairwise_term",
    "simplex_mean_field_term",
    "FrequencyOrder",
    "effective_frequencies",
    "frequency_locked_fraction",
    "frequency_order_diagnostics",
    "frequency_synchronisation_index",
    "frequency_synchronisation_index_gradient",
    "ChimeraDiagnostics",
    "chimera_diagnostics",
    "chimera_index",
    "chimera_index_gradient",
    "community_metastability",
    "community_order_parameters",
    "metastability_index",
    "metastability_index_gradient",
    "mutual_information_matrix",
    "normalised_phase_entropy",
    "pairwise_mutual_information",
    "phase_entropy",
    "phase_entropy_series",
    "coherence_matrix",
    "coherence_spectrum",
    "leading_coherence_eigenvector",
    "mean_coherence_matrix",
    "phase_locking_matrix",
    "ClusterPartition",
    "cluster_count",
    "cluster_partition",
    "phase_clusters",
    "mean_phase",
    "mean_phase_gradient",
    "mean_phase_hessian",
    "networked_kuramoto_force",
    "networked_kuramoto_jacobian",
    "last_networked_kuramoto_force_tier_used",
    "last_networked_kuramoto_jacobian_tier_used",
    "order_parameter",
    "order_parameter_gradient",
    "order_parameter_hessian",
    "sakaguchi_mean_field_force",
    "sakaguchi_mean_field_jacobian",
    "triadic_mean_field_force",
    "triadic_mean_field_jacobian",
    "last_triadic_mean_field_force_tier_used",
    "last_triadic_mean_field_jacobian_tier_used",
    "last_sakaguchi_mean_field_force_tier_used",
    "last_sakaguchi_mean_field_jacobian_tier_used",
    "sakaguchi_force",
    "sakaguchi_jacobian",
    "last_sakaguchi_force_tier_used",
    "last_sakaguchi_jacobian_tier_used",
]

import numpy as np
from numpy.typing import NDArray


def rust_random_state(n_qubits: int, seed: int = 42) -> NDArray[np.complex128]:
    """Return a normalized complex random state vector for fallback tests."""
    np.random.seed(seed)
    state = np.random.randn(2**n_qubits) + 1j * np.random.randn(2**n_qubits)
    return np.asarray(state / np.linalg.norm(state), dtype=np.complex128)
