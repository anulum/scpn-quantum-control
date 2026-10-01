# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Deprecated Kuramoto accelerator shim
"""Backward-compatible re-export shim for the relocated Kuramoto accelerators.

The accelerated Kuramoto primitives now live in the standalone :mod:`oscillatools.accel`
distribution. This shim keeps ``scpn_quantum_control.accel`` — both the aggregated public
names and the individual submodules (``scpn_quantum_control.accel.networked_kuramoto`` …) —
importable for the deprecation window, forwarding every name to :mod:`oscillatools.accel`
and preserving object identity for already-loaded submodules. It emits a single
:class:`DeprecationWarning` naming the new import path. See ``DEPRECATIONS.md``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oscillatools.accel import (
        AdaptiveGradients as AdaptiveGradients,
    )
    from oscillatools.accel import (
        AdaptivePhaseForce as AdaptivePhaseForce,
    )
    from oscillatools.accel import (
        AdaptiveTrajectory as AdaptiveTrajectory,
    )
    from oscillatools.accel import (
        BasinEstimate as BasinEstimate,
    )
    from oscillatools.accel import (
        BasinStabilityEstimate as BasinStabilityEstimate,
    )
    from oscillatools.accel import (
        ChimeraDiagnostics as ChimeraDiagnostics,
    )
    from oscillatools.accel import (
        ClusterPartition as ClusterPartition,
    )
    from oscillatools.accel import (
        CollectiveControlGradients as CollectiveControlGradients,
    )
    from oscillatools.accel import (
        ContinuationBranch as ContinuationBranch,
    )
    from oscillatools.accel import (
        ControlledNetworkTrajectory as ControlledNetworkTrajectory,
    )
    from oscillatools.accel import (
        CoordinatedResetGradients as CoordinatedResetGradients,
    )
    from oscillatools.accel import (
        CoordinatedResetTrajectory as CoordinatedResetTrajectory,
    )
    from oscillatools.accel import (
        CouplingDesignResult as CouplingDesignResult,
    )
    from oscillatools.accel import (
        CouplingFunctionEstimate as CouplingFunctionEstimate,
    )
    from oscillatools.accel import (
        CouplingFunctionEstimator as CouplingFunctionEstimator,
    )
    from oscillatools.accel import (
        CouplingFunctionGradients as CouplingFunctionGradients,
    )
    from oscillatools.accel import (
        CouplingProjection as CouplingProjection,
    )
    from oscillatools.accel import (
        CouplingTerm as CouplingTerm,
    )
    from oscillatools.accel import (
        DelayedForce as DelayedForce,
    )
    from oscillatools.accel import (
        DelayedGradients as DelayedGradients,
    )
    from oscillatools.accel import (
        DelayedTrajectory as DelayedTrajectory,
    )
    from oscillatools.accel import (
        DesynchronisingPolicy as DesynchronisingPolicy,
    )
    from oscillatools.accel import (
        DopriTrajectory as DopriTrajectory,
    )
    from oscillatools.accel import (
        DynamicalBayesianPosterior as DynamicalBayesianPosterior,
    )
    from oscillatools.accel import (
        ForcedCollectiveTrajectory as ForcedCollectiveTrajectory,
    )
    from oscillatools.accel import (
        FormalLyapunovCertificate as FormalLyapunovCertificate,
    )
    from oscillatools.accel import (
        FrequencyDensity as FrequencyDensity,
    )
    from oscillatools.accel import (
        FrequencyOrder as FrequencyOrder,
    )
    from oscillatools.accel import (
        GraphLike as GraphLike,
    )
    from oscillatools.accel import (
        HigherOrderWatanabeStrogatzTrajectory as HigherOrderWatanabeStrogatzTrajectory,
    )
    from oscillatools.accel import (
        HodgeComponents as HodgeComponents,
    )
    from oscillatools.accel import (
        HodgeStructure as HodgeStructure,
    )
    from oscillatools.accel import (
        HysteresisLoop as HysteresisLoop,
    )
    from oscillatools.accel import (
        InertialGradients as InertialGradients,
    )
    from oscillatools.accel import (
        InertialTrajectory as InertialTrajectory,
    )
    from oscillatools.accel import (
        KuramotoIvpSolution as KuramotoIvpSolution,
    )
    from oscillatools.accel import (
        KuramotoParameterGrid as KuramotoParameterGrid,
    )
    from oscillatools.accel import (
        KuramotoParameters as KuramotoParameters,
    )
    from oscillatools.accel import (
        KuramotoSystem as KuramotoSystem,
    )
    from oscillatools.accel import (
        LyapunovCertificateReport as LyapunovCertificateReport,
    )
    from oscillatools.accel import (
        LyapunovCounterexample as LyapunovCounterexample,
    )
    from oscillatools.accel import (
        LyapunovLipschitzBounds as LyapunovLipschitzBounds,
    )
    from oscillatools.accel import (
        MeanFieldForce as MeanFieldForce,
    )
    from oscillatools.accel import (
        MpcControlGradients as MpcControlGradients,
    )
    from oscillatools.accel import (
        MpcOptimumSensitivity as MpcOptimumSensitivity,
    )
    from oscillatools.accel import (
        MultiLangDispatcher as MultiLangDispatcher,
    )
    from oscillatools.accel import (
        MultiplexSynchronisationStability as MultiplexSynchronisationStability,
    )
    from oscillatools.accel import (
        MultiplexTrajectory as MultiplexTrajectory,
    )
    from oscillatools.accel import (
        NetworkControlGradients as NetworkControlGradients,
    )
    from oscillatools.accel import (
        NeuralLyapunovCertificate as NeuralLyapunovCertificate,
    )
    from oscillatools.accel import (
        NoisyGradients as NoisyGradients,
    )
    from oscillatools.accel import (
        NoisyKuramotoRun as NoisyKuramotoRun,
    )
    from oscillatools.accel import (
        Observable as Observable,
    )
    from oscillatools.accel import (
        OscillatorIsingTrajectory as OscillatorIsingTrajectory,
    )
    from oscillatools.accel import (
        ParameterSweepResult as ParameterSweepResult,
    )
    from oscillatools.accel import (
        PermutationSignificanceResult as PermutationSignificanceResult,
    )
    from oscillatools.accel import (
        PhaseForce as PhaseForce,
    )
    from oscillatools.accel import (
        PhaseJacobian as PhaseJacobian,
    )
    from oscillatools.accel import (
        PhasePotential as PhasePotential,
    )
    from oscillatools.accel import (
        PinningDesignResult as PinningDesignResult,
    )
    from oscillatools.accel import (
        PlasticityRule as PlasticityRule,
    )
    from oscillatools.accel import (
        PolicyRolloutGradients as PolicyRolloutGradients,
    )
    from oscillatools.accel import (
        PseudoArclengthBranch as PseudoArclengthBranch,
    )
    from oscillatools.accel import (
        QifMeanFieldGradients as QifMeanFieldGradients,
    )
    from oscillatools.accel import (
        QifMeanFieldTrajectory as QifMeanFieldTrajectory,
    )
    from oscillatools.accel import (
        QuantumVanDerPolTrajectory as QuantumVanDerPolTrajectory,
    )
    from oscillatools.accel import (
        RecedingHorizonResult as RecedingHorizonResult,
    )
    from oscillatools.accel import (
        SaddleNodePoint as SaddleNodePoint,
    )
    from oscillatools.accel import (
        SparseDynamicsEstimator as SparseDynamicsEstimator,
    )
    from oscillatools.accel import (
        SparseDynamicsModel as SparseDynamicsModel,
    )
    from oscillatools.accel import (
        SparseKuramotoCoupling as SparseKuramotoCoupling,
    )
    from oscillatools.accel import (
        StabilitySpectrum as StabilitySpectrum,
    )
    from oscillatools.accel import (
        StochasticForce as StochasticForce,
    )
    from oscillatools.accel import (
        StuartLandauTrajectory as StuartLandauTrajectory,
    )
    from oscillatools.accel import (
        SurrogateControlComparison as SurrogateControlComparison,
    )
    from oscillatools.accel import (
        SurrogateStepModel as SurrogateStepModel,
    )
    from oscillatools.accel import (
        SwarmalatorOrderParameters as SwarmalatorOrderParameters,
    )
    from oscillatools.accel import (
        SwarmalatorTrajectory as SwarmalatorTrajectory,
    )
    from oscillatools.accel import (
        SynchronisationCertificate as SynchronisationCertificate,
    )
    from oscillatools.accel import (
        SystemIdentificationResult as SystemIdentificationResult,
    )
    from oscillatools.accel import (
        TerminalObjective as TerminalObjective,
    )
    from oscillatools.accel import (
        TimeVaryingCouplingHistory as TimeVaryingCouplingHistory,
    )
    from oscillatools.accel import (
        TopologicalKuramotoTrajectory as TopologicalKuramotoTrajectory,
    )
    from oscillatools.accel import (
        WatanabeStrogatzTrajectory as WatanabeStrogatzTrajectory,
    )
    from oscillatools.accel import (
        WinfreeTrajectory as WinfreeTrajectory,
    )
    from oscillatools.accel import (
        adaptive_state_sensitivity as adaptive_state_sensitivity,
    )
    from oscillatools.accel import (
        adaptive_terminal_value_and_grad as adaptive_terminal_value_and_grad,
    )
    from oscillatools.accel import (
        adaptive_vector_field as adaptive_vector_field,
    )
    from oscillatools.accel import (
        amplitudes as amplitudes,
    )
    from oscillatools.accel import (
        available_tiers as available_tiers,
    )
    from oscillatools.accel import (
        certify_neural_lyapunov as certify_neural_lyapunov,
    )
    from oscillatools.accel import (
        certify_synchronisation as certify_synchronisation,
    )
    from oscillatools.accel import (
        chimera_diagnostics as chimera_diagnostics,
    )
    from oscillatools.accel import (
        chimera_index as chimera_index,
    )
    from oscillatools.accel import (
        chimera_index_gradient as chimera_index_gradient,
    )
    from oscillatools.accel import (
        chimera_snapshot as chimera_snapshot,
    )
    from oscillatools.accel import (
        cluster_count as cluster_count,
    )
    from oscillatools.accel import (
        cluster_partition as cluster_partition,
    )
    from oscillatools.accel import (
        coherence_matrix as coherence_matrix,
    )
    from oscillatools.accel import (
        coherence_objective as coherence_objective,
    )
    from oscillatools.accel import (
        coherence_spectrum as coherence_spectrum,
    )
    from oscillatools.accel import (
        coherent_amplitude as coherent_amplitude,
    )
    from oscillatools.accel import (
        collective_control_value_and_grad as collective_control_value_and_grad,
    )
    from oscillatools.accel import (
        community_metastability as community_metastability,
    )
    from oscillatools.accel import (
        community_order_parameters as community_order_parameters,
    )
    from oscillatools.accel import (
        compare_surrogate_control as compare_surrogate_control,
    )
    from oscillatools.accel import (
        continuation_sweep as continuation_sweep,
    )
    from oscillatools.accel import (
        contraction_rate as contraction_rate,
    )
    from oscillatools.accel import (
        coordinated_reset_phases as coordinated_reset_phases,
    )
    from oscillatools.accel import (
        coordinated_reset_sites as coordinated_reset_sites,
    )
    from oscillatools.accel import (
        coordinated_reset_terminal_value_and_grad as coordinated_reset_terminal_value_and_grad,
    )
    from oscillatools.accel import (
        coupling_from_networkx as coupling_from_networkx,
    )
    from oscillatools.accel import (
        coupling_function_trajectory_value_and_grad as coupling_function_trajectory_value_and_grad,
    )
    from oscillatools.accel import (
        coupling_function_value as coupling_function_value,
    )
    from oscillatools.accel import (
        critical_coupling as critical_coupling,
    )
    from oscillatools.accel import (
        cut_value as cut_value,
    )
    from oscillatools.accel import (
        daido_mean_field_force as daido_mean_field_force,
    )
    from oscillatools.accel import (
        daido_mean_field_jacobian as daido_mean_field_jacobian,
    )
    from oscillatools.accel import (
        daido_mode_phase as daido_mode_phase,
    )
    from oscillatools.accel import (
        daido_mode_phase_gradient as daido_mode_phase_gradient,
    )
    from oscillatools.accel import (
        daido_mode_phase_hessian as daido_mode_phase_hessian,
    )
    from oscillatools.accel import (
        daido_order_parameter as daido_order_parameter,
    )
    from oscillatools.accel import (
        daido_order_parameter_gradient as daido_order_parameter_gradient,
    )
    from oscillatools.accel import (
        daido_order_parameter_hessian as daido_order_parameter_hessian,
    )
    from oscillatools.accel import (
        delayed_delay_gradient as delayed_delay_gradient,
    )
    from oscillatools.accel import (
        delayed_delay_sensitivity as delayed_delay_sensitivity,
    )
    from oscillatools.accel import (
        delayed_mean_field_force as delayed_mean_field_force,
    )
    from oscillatools.accel import (
        delayed_networked_force as delayed_networked_force,
    )
    from oscillatools.accel import (
        delayed_phase_sensitivity as delayed_phase_sensitivity,
    )
    from oscillatools.accel import (
        delayed_terminal_value_and_grad as delayed_terminal_value_and_grad,
    )
    from oscillatools.accel import (
        design_pinning as design_pinning,
    )
    from oscillatools.accel import (
        design_synchronising_coupling as design_synchronising_coupling,
    )
    from oscillatools.accel import (
        discover_phase_dynamics as discover_phase_dynamics,
    )
    from oscillatools.accel import (
        dispatch as dispatch,
    )
    from oscillatools.accel import (
        effective_frequencies as effective_frequencies,
    )
    from oscillatools.accel import (
        estimate_ring_basins as estimate_ring_basins,
    )
    from oscillatools.accel import (
        falsify_neural_lyapunov as falsify_neural_lyapunov,
    )
    from oscillatools.accel import (
        fit_neural_lyapunov_certificate as fit_neural_lyapunov_certificate,
    )
    from oscillatools.accel import (
        fit_surrogate_step_model as fit_surrogate_step_model,
    )
    from oscillatools.accel import (
        fold_defining_jacobian as fold_defining_jacobian,
    )
    from oscillatools.accel import (
        fold_defining_residual as fold_defining_residual,
    )
    from oscillatools.accel import (
        formally_certify_neural_lyapunov as formally_certify_neural_lyapunov,
    )
    from oscillatools.accel import (
        frequency_locked_fraction as frequency_locked_fraction,
    )
    from oscillatools.accel import (
        frequency_order_diagnostics as frequency_order_diagnostics,
    )
    from oscillatools.accel import (
        frequency_spread as frequency_spread,
    )
    from oscillatools.accel import (
        frequency_synchronisation_index as frequency_synchronisation_index,
    )
    from oscillatools.accel import (
        frequency_synchronisation_index_gradient as frequency_synchronisation_index_gradient,
    )
    from oscillatools.accel import (
        gaussian_critical_coupling as gaussian_critical_coupling,
    )
    from oscillatools.accel import (
        gaussian_density as gaussian_density,
    )
    from oscillatools.accel import (
        graph_from_networked_coupling as graph_from_networked_coupling,
    )
    from oscillatools.accel import (
        hebbian_adaptive_jacobian as hebbian_adaptive_jacobian,
    )
    from oscillatools.accel import (
        hebbian_coupling_equilibrium as hebbian_coupling_equilibrium,
    )
    from oscillatools.accel import (
        hebbian_plasticity_rate as hebbian_plasticity_rate,
    )
    from oscillatools.accel import (
        heterogeneous_force as heterogeneous_force,
    )
    from oscillatools.accel import (
        heterogeneous_force_components as heterogeneous_force_components,
    )
    from oscillatools.accel import (
        heterogeneous_jacobian as heterogeneous_jacobian,
    )
    from oscillatools.accel import (
        hodge_decomposition as hodge_decomposition,
    )
    from oscillatools.accel import (
        hyperedge_force as hyperedge_force,
    )
    from oscillatools.accel import (
        hyperedge_jacobian as hyperedge_jacobian,
    )
    from oscillatools.accel import (
        hyperedge_term as hyperedge_term,
    )
    from oscillatools.accel import (
        hysteresis_loop as hysteresis_loop,
    )
    from oscillatools.accel import (
        inertial_energy as inertial_energy,
    )
    from oscillatools.accel import (
        inertial_jacobian as inertial_jacobian,
    )
    from oscillatools.accel import (
        inertial_state_sensitivity as inertial_state_sensitivity,
    )
    from oscillatools.accel import (
        inertial_terminal_value_and_grad as inertial_terminal_value_and_grad,
    )
    from oscillatools.accel import (
        inertial_vector_field as inertial_vector_field,
    )
    from oscillatools.accel import (
        infer_coupling_function as infer_coupling_function,
    )
    from oscillatools.accel import (
        infer_network_bayesian as infer_network_bayesian,
    )
    from oscillatools.accel import (
        integrate_adaptive_kuramoto as integrate_adaptive_kuramoto,
    )
    from oscillatools.accel import (
        integrate_controlled_network as integrate_controlled_network,
    )
    from oscillatools.accel import (
        integrate_coordinated_reset as integrate_coordinated_reset,
    )
    from oscillatools.accel import (
        integrate_delayed_kuramoto as integrate_delayed_kuramoto,
    )
    from oscillatools.accel import (
        integrate_forced_collective as integrate_forced_collective,
    )
    from oscillatools.accel import (
        integrate_higher_order_watanabe_strogatz as integrate_higher_order_watanabe_strogatz,
    )
    from oscillatools.accel import (
        integrate_inertial as integrate_inertial,
    )
    from oscillatools.accel import (
        integrate_multiplex as integrate_multiplex,
    )
    from oscillatools.accel import (
        integrate_noisy_kuramoto as integrate_noisy_kuramoto,
    )
    from oscillatools.accel import (
        integrate_oscillator_ising_machine as integrate_oscillator_ising_machine,
    )
    from oscillatools.accel import (
        integrate_qif_mean_field as integrate_qif_mean_field,
    )
    from oscillatools.accel import (
        integrate_quantum_vanderpol as integrate_quantum_vanderpol,
    )
    from oscillatools.accel import (
        integrate_sdre_controlled_kuramoto as integrate_sdre_controlled_kuramoto,
    )
    from oscillatools.accel import (
        integrate_stuart_landau as integrate_stuart_landau,
    )
    from oscillatools.accel import (
        integrate_swarmalators as integrate_swarmalators,
    )
    from oscillatools.accel import (
        integrate_symplectic_inertial as integrate_symplectic_inertial,
    )
    from oscillatools.accel import (
        integrate_topological_kuramoto as integrate_topological_kuramoto,
    )
    from oscillatools.accel import (
        integrate_watanabe_strogatz as integrate_watanabe_strogatz,
    )
    from oscillatools.accel import (
        integrate_winfree as integrate_winfree,
    )
    from oscillatools.accel import (
        interaction_energy_objective as interaction_energy_objective,
    )
    from oscillatools.accel import (
        interlayer_synchronisation as interlayer_synchronisation,
    )
    from oscillatools.accel import (
        is_oscillation_death as is_oscillation_death,
    )
    from oscillatools.accel import (
        is_synchronisation_stable as is_synchronisation_stable,
    )
    from oscillatools.accel import (
        is_synchronised_branch_stable as is_synchronised_branch_stable,
    )
    from oscillatools.accel import (
        is_twisted_state_stable as is_twisted_state_stable,
    )
    from oscillatools.accel import (
        ising_hamiltonian as ising_hamiltonian,
    )
    from oscillatools.accel import (
        ising_spins as ising_spins,
    )
    from oscillatools.accel import (
        jax_kuramoto_delayed_ensemble as jax_kuramoto_delayed_ensemble,
    )
    from oscillatools.accel import (
        jax_kuramoto_delayed_ensemble_gradient as jax_kuramoto_delayed_ensemble_gradient,
    )
    from oscillatools.accel import (
        jax_kuramoto_delayed_gradient as jax_kuramoto_delayed_gradient,
    )
    from oscillatools.accel import (
        jax_kuramoto_delayed_trajectory as jax_kuramoto_delayed_trajectory,
    )
    from oscillatools.accel import (
        jax_kuramoto_dopri_trajectory as jax_kuramoto_dopri_trajectory,
    )
    from oscillatools.accel import (
        jax_kuramoto_euler_trajectory as jax_kuramoto_euler_trajectory,
    )
    from oscillatools.accel import (
        jax_kuramoto_rk4_ensemble as jax_kuramoto_rk4_ensemble,
    )
    from oscillatools.accel import (
        jax_kuramoto_rk4_ensemble_gradient as jax_kuramoto_rk4_ensemble_gradient,
    )
    from oscillatools.accel import (
        jax_kuramoto_rk4_gradient as jax_kuramoto_rk4_gradient,
    )
    from oscillatools.accel import (
        jax_kuramoto_rk4_trajectory as jax_kuramoto_rk4_trajectory,
    )
    from oscillatools.accel import (
        jax_mpc_control_value_and_grad as jax_mpc_control_value_and_grad,
    )
    from oscillatools.accel import (
        jax_mpc_horizon_control as jax_mpc_horizon_control,
    )
    from oscillatools.accel import (
        jax_mpc_optimum as jax_mpc_optimum,
    )
    from oscillatools.accel import (
        jax_networked_inertial_trajectory as jax_networked_inertial_trajectory,
    )
    from oscillatools.accel import (
        jax_networked_noisy_trajectory as jax_networked_noisy_trajectory,
    )
    from oscillatools.accel import (
        jax_networked_symplectic_inertial_trajectory as jax_networked_symplectic_inertial_trajectory,
    )
    from oscillatools.accel import (
        kuramoto_dopri_trajectory as kuramoto_dopri_trajectory,
    )
    from oscillatools.accel import (
        kuramoto_dopri_vjp as kuramoto_dopri_vjp,
    )
    from oscillatools.accel import (
        kuramoto_euler_trajectory as kuramoto_euler_trajectory,
    )
    from oscillatools.accel import (
        kuramoto_euler_vjp as kuramoto_euler_vjp,
    )
    from oscillatools.accel import (
        kuramoto_interaction_energy as kuramoto_interaction_energy,
    )
    from oscillatools.accel import (
        kuramoto_interaction_energy_gradient as kuramoto_interaction_energy_gradient,
    )
    from oscillatools.accel import (
        kuramoto_interaction_energy_hessian as kuramoto_interaction_energy_hessian,
    )
    from oscillatools.accel import (
        kuramoto_ode_jacobian as kuramoto_ode_jacobian,
    )
    from oscillatools.accel import (
        kuramoto_ode_rhs as kuramoto_ode_rhs,
    )
    from oscillatools.accel import (
        kuramoto_order_parameter_from_macro as kuramoto_order_parameter_from_macro,
    )
    from oscillatools.accel import (
        kuramoto_rk4_trajectory as kuramoto_rk4_trajectory,
    )
    from oscillatools.accel import (
        kuramoto_rk4_vjp as kuramoto_rk4_vjp,
    )
    from oscillatools.accel import (
        kuramoto_sdre_gain as kuramoto_sdre_gain,
    )
    from oscillatools.accel import (
        last_daido_gradient_tier_used as last_daido_gradient_tier_used,
    )
    from oscillatools.accel import (
        last_daido_hessian_tier_used as last_daido_hessian_tier_used,
    )
    from oscillatools.accel import (
        last_daido_mean_field_force_tier_used as last_daido_mean_field_force_tier_used,
    )
    from oscillatools.accel import (
        last_daido_mean_field_jacobian_tier_used as last_daido_mean_field_jacobian_tier_used,
    )
    from oscillatools.accel import (
        last_daido_mode_phase_gradient_tier_used as last_daido_mode_phase_gradient_tier_used,
    )
    from oscillatools.accel import (
        last_daido_mode_phase_hessian_tier_used as last_daido_mode_phase_hessian_tier_used,
    )
    from oscillatools.accel import (
        last_daido_mode_phase_tier_used as last_daido_mode_phase_tier_used,
    )
    from oscillatools.accel import (
        last_daido_tier_used as last_daido_tier_used,
    )
    from oscillatools.accel import (
        last_gradient_tier_used as last_gradient_tier_used,
    )
    from oscillatools.accel import (
        last_hessian_tier_used as last_hessian_tier_used,
    )
    from oscillatools.accel import (
        last_kuramoto_dopri_trajectory_tier_used as last_kuramoto_dopri_trajectory_tier_used,
    )
    from oscillatools.accel import (
        last_kuramoto_euler_trajectory_tier_used as last_kuramoto_euler_trajectory_tier_used,
    )
    from oscillatools.accel import (
        last_kuramoto_euler_vjp_tier_used as last_kuramoto_euler_vjp_tier_used,
    )
    from oscillatools.accel import (
        last_kuramoto_interaction_energy_gradient_tier_used as last_kuramoto_interaction_energy_gradient_tier_used,
    )
    from oscillatools.accel import (
        last_kuramoto_interaction_energy_hessian_tier_used as last_kuramoto_interaction_energy_hessian_tier_used,
    )
    from oscillatools.accel import (
        last_kuramoto_interaction_energy_tier_used as last_kuramoto_interaction_energy_tier_used,
    )
    from oscillatools.accel import (
        last_kuramoto_rk4_trajectory_tier_used as last_kuramoto_rk4_trajectory_tier_used,
    )
    from oscillatools.accel import (
        last_kuramoto_rk4_vjp_tier_used as last_kuramoto_rk4_vjp_tier_used,
    )
    from oscillatools.accel import (
        last_local_mean_phase_jacobian_tier_used as last_local_mean_phase_jacobian_tier_used,
    )
    from oscillatools.accel import (
        last_local_mean_phase_tier_used as last_local_mean_phase_tier_used,
    )
    from oscillatools.accel import (
        last_local_order_parameter_jacobian_tier_used as last_local_order_parameter_jacobian_tier_used,
    )
    from oscillatools.accel import (
        last_local_order_parameter_tier_used as last_local_order_parameter_tier_used,
    )
    from oscillatools.accel import (
        last_mean_field_force_tier_used as last_mean_field_force_tier_used,
    )
    from oscillatools.accel import (
        last_mean_field_jacobian_tier_used as last_mean_field_jacobian_tier_used,
    )
    from oscillatools.accel import (
        last_mean_phase_gradient_tier_used as last_mean_phase_gradient_tier_used,
    )
    from oscillatools.accel import (
        last_mean_phase_hessian_tier_used as last_mean_phase_hessian_tier_used,
    )
    from oscillatools.accel import (
        last_mean_phase_tier_used as last_mean_phase_tier_used,
    )
    from oscillatools.accel import (
        last_networked_delayed_trajectory_tier_used as last_networked_delayed_trajectory_tier_used,
    )
    from oscillatools.accel import (
        last_networked_inertial_trajectory_tier_used as last_networked_inertial_trajectory_tier_used,
    )
    from oscillatools.accel import (
        last_networked_kuramoto_force_tier_used as last_networked_kuramoto_force_tier_used,
    )
    from oscillatools.accel import (
        last_networked_kuramoto_jacobian_tier_used as last_networked_kuramoto_jacobian_tier_used,
    )
    from oscillatools.accel import (
        last_networked_noisy_trajectory_tier_used as last_networked_noisy_trajectory_tier_used,
    )
    from oscillatools.accel import (
        last_networked_symplectic_inertial_trajectory_tier_used as last_networked_symplectic_inertial_trajectory_tier_used,
    )
    from oscillatools.accel import (
        last_sakaguchi_force_tier_used as last_sakaguchi_force_tier_used,
    )
    from oscillatools.accel import (
        last_sakaguchi_jacobian_tier_used as last_sakaguchi_jacobian_tier_used,
    )
    from oscillatools.accel import (
        last_sakaguchi_mean_field_force_tier_used as last_sakaguchi_mean_field_force_tier_used,
    )
    from oscillatools.accel import (
        last_sakaguchi_mean_field_jacobian_tier_used as last_sakaguchi_mean_field_jacobian_tier_used,
    )
    from oscillatools.accel import (
        last_tier_used as last_tier_used,
    )
    from oscillatools.accel import (
        last_triadic_mean_field_force_tier_used as last_triadic_mean_field_force_tier_used,
    )
    from oscillatools.accel import (
        last_triadic_mean_field_jacobian_tier_used as last_triadic_mean_field_jacobian_tier_used,
    )
    from oscillatools.accel import (
        layer_order_parameters as layer_order_parameters,
    )
    from oscillatools.accel import (
        leading_coherence_eigenvector as leading_coherence_eigenvector,
    )
    from oscillatools.accel import (
        learn_coupling as learn_coupling,
    )
    from oscillatools.accel import (
        learn_desynchronising_policy as learn_desynchronising_policy,
    )
    from oscillatools.accel import (
        local_mean_phase as local_mean_phase,
    )
    from oscillatools.accel import (
        local_mean_phase_jacobian as local_mean_phase_jacobian,
    )
    from oscillatools.accel import (
        local_order_parameter as local_order_parameter,
    )
    from oscillatools.accel import (
        local_order_parameter_jacobian as local_order_parameter_jacobian,
    )
    from oscillatools.accel import (
        locate_saddle_node as locate_saddle_node,
    )
    from oscillatools.accel import (
        lorentzian_critical_coupling as lorentzian_critical_coupling,
    )
    from oscillatools.accel import (
        lorentzian_density as lorentzian_density,
    )
    from oscillatools.accel import (
        lorentzian_noisy_critical_coupling as lorentzian_noisy_critical_coupling,
    )
    from oscillatools.accel import (
        lorentzian_order_parameter as lorentzian_order_parameter,
    )
    from oscillatools.accel import (
        lyapunov_spectrum as lyapunov_spectrum,
    )
    from oscillatools.accel import (
        macro_from_kuramoto_order_parameter as macro_from_kuramoto_order_parameter,
    )
    from oscillatools.accel import (
        master_stability_function as master_stability_function,
    )
    from oscillatools.accel import (
        maximal_lyapunov_exponent as maximal_lyapunov_exponent,
    )
    from oscillatools.accel import (
        mean_coherence_matrix as mean_coherence_matrix,
    )
    from oscillatools.accel import (
        mean_field_force as mean_field_force,
    )
    from oscillatools.accel import (
        mean_field_jacobian as mean_field_jacobian,
    )
    from oscillatools.accel import (
        mean_field_phase_rule as mean_field_phase_rule,
    )
    from oscillatools.accel import (
        mean_field_phase_rule_jacobian as mean_field_phase_rule_jacobian,
    )
    from oscillatools.accel import (
        mean_order_parameter as mean_order_parameter,
    )
    from oscillatools.accel import (
        mean_phase as mean_phase,
    )
    from oscillatools.accel import (
        mean_phase_gradient as mean_phase_gradient,
    )
    from oscillatools.accel import (
        mean_phase_hessian as mean_phase_hessian,
    )
    from oscillatools.accel import (
        mean_photon_number as mean_photon_number,
    )
    from oscillatools.accel import (
        metastability as metastability,
    )
    from oscillatools.accel import (
        metastability_index as metastability_index,
    )
    from oscillatools.accel import (
        metastability_index_gradient as metastability_index_gradient,
    )
    from oscillatools.accel import (
        mpc_optimum_parameter_sensitivity as mpc_optimum_parameter_sensitivity,
    )
    from oscillatools.accel import (
        mpc_plan_energy_gradient as mpc_plan_energy_gradient,
    )
    from oscillatools.accel import (
        multiplex_field as multiplex_field,
    )
    from oscillatools.accel import (
        multiplex_jacobian as multiplex_jacobian,
    )
    from oscillatools.accel import (
        multiplex_synchronisation_stability as multiplex_synchronisation_stability,
    )
    from oscillatools.accel import (
        mutual_information_matrix as mutual_information_matrix,
    )
    from oscillatools.accel import (
        network_control_value_and_grad as network_control_value_and_grad,
    )
    from oscillatools.accel import (
        network_phase_embedding as network_phase_embedding,
    )
    from oscillatools.accel import (
        networked_delayed_trajectory as networked_delayed_trajectory,
    )
    from oscillatools.accel import (
        networked_inertial_trajectory as networked_inertial_trajectory,
    )
    from oscillatools.accel import (
        networked_kuramoto_force as networked_kuramoto_force,
    )
    from oscillatools.accel import (
        networked_kuramoto_jacobian as networked_kuramoto_jacobian,
    )
    from oscillatools.accel import (
        networked_noisy_trajectory as networked_noisy_trajectory,
    )
    from oscillatools.accel import (
        networked_phase_rule as networked_phase_rule,
    )
    from oscillatools.accel import (
        networked_phase_rule_jacobian as networked_phase_rule_jacobian,
    )
    from oscillatools.accel import (
        networked_symplectic_inertial_trajectory as networked_symplectic_inertial_trajectory,
    )
    from oscillatools.accel import (
        neural_lyapunov_decrease as neural_lyapunov_decrease,
    )
    from oscillatools.accel import (
        neural_lyapunov_lipschitz_bounds as neural_lyapunov_lipschitz_bounds,
    )
    from oscillatools.accel import (
        neural_lyapunov_value as neural_lyapunov_value,
    )
    from oscillatools.accel import (
        noisy_critical_coupling as noisy_critical_coupling,
    )
    from oscillatools.accel import (
        noisy_kuramoto_step as noisy_kuramoto_step,
    )
    from oscillatools.accel import (
        noisy_phase_sensitivity as noisy_phase_sensitivity,
    )
    from oscillatools.accel import (
        noisy_stationary_order_parameter as noisy_stationary_order_parameter,
    )
    from oscillatools.accel import (
        noisy_terminal_value_and_grad as noisy_terminal_value_and_grad,
    )
    from oscillatools.accel import (
        normalised_phase_entropy as normalised_phase_entropy,
    )
    from oscillatools.accel import (
        optimise_collective_forcing as optimise_collective_forcing,
    )
    from oscillatools.accel import (
        optimise_coupling as optimise_coupling,
    )
    from oscillatools.accel import (
        optimise_network_control as optimise_network_control,
    )
    from oscillatools.accel import (
        order_parameter as order_parameter,
    )
    from oscillatools.accel import (
        order_parameter_gradient as order_parameter_gradient,
    )
    from oscillatools.accel import (
        order_parameter_hessian as order_parameter_hessian,
    )
    from oscillatools.accel import (
        order_parameter_timeseries as order_parameter_timeseries,
    )
    from oscillatools.accel import (
        oscillator_ising_energy as oscillator_ising_energy,
    )
    from oscillatools.accel import (
        oscillator_ising_field as oscillator_ising_field,
    )
    from oscillatools.accel import (
        ott_antonsen_field as ott_antonsen_field,
    )
    from oscillatools.accel import (
        ott_antonsen_order_parameter as ott_antonsen_order_parameter,
    )
    from oscillatools.accel import (
        ott_antonsen_steady_state as ott_antonsen_steady_state,
    )
    from oscillatools.accel import (
        ott_antonsen_terminal_order_parameter_value_and_grad as ott_antonsen_terminal_order_parameter_value_and_grad,
    )
    from oscillatools.accel import (
        ott_antonsen_trajectory as ott_antonsen_trajectory,
    )
    from oscillatools.accel import (
        pairwise_mutual_information as pairwise_mutual_information,
    )
    from oscillatools.accel import (
        pairwise_term as pairwise_term,
    )
    from oscillatools.accel import (
        permutation_significance_test as permutation_significance_test,
    )
    from oscillatools.accel import (
        phase_clusters as phase_clusters,
    )
    from oscillatools.accel import (
        phase_cohesiveness as phase_cohesiveness,
    )
    from oscillatools.accel import (
        phase_distribution as phase_distribution,
    )
    from oscillatools.accel import (
        phase_entropy as phase_entropy,
    )
    from oscillatools.accel import (
        phase_entropy_series as phase_entropy_series,
    )
    from oscillatools.accel import (
        phase_locking_matrix as phase_locking_matrix,
    )
    from oscillatools.accel import (
        phase_raster as phase_raster,
    )
    from oscillatools.accel import (
        phase_synchronisation as phase_synchronisation,
    )
    from oscillatools.accel import (
        phase_target_objective as phase_target_objective,
    )
    from oscillatools.accel import (
        pinning_coherence_value as pinning_coherence_value,
    )
    from oscillatools.accel import (
        pinning_coherence_value_and_grad as pinning_coherence_value_and_grad,
    )
    from oscillatools.accel import (
        policy_rollout_value_and_grad as policy_rollout_value_and_grad,
    )
    from oscillatools.accel import (
        potential_decrease_rate as potential_decrease_rate,
    )
    from oscillatools.accel import (
        pseudo_arclength_continuation as pseudo_arclength_continuation,
    )
    from oscillatools.accel import (
        qif_mean_field_fixed_point as qif_mean_field_fixed_point,
    )
    from oscillatools.accel import (
        qif_mean_field_jacobian as qif_mean_field_jacobian,
    )
    from oscillatools.accel import (
        qif_mean_field_rates as qif_mean_field_rates,
    )
    from oscillatools.accel import (
        qif_mean_field_terminal_value_and_grad as qif_mean_field_terminal_value_and_grad,
    )
    from oscillatools.accel import (
        qif_potential_from_theta as qif_potential_from_theta,
    )
    from oscillatools.accel import (
        receding_horizon_control as receding_horizon_control,
    )
    from oscillatools.accel import (
        refine_coupling_function as refine_coupling_function,
    )
    from oscillatools.accel import (
        ring_coupling_matrix as ring_coupling_matrix,
    )
    from oscillatools.accel import (
        ring_sparse_coupling as ring_sparse_coupling,
    )
    from oscillatools.accel import (
        sakaguchi_force as sakaguchi_force,
    )
    from oscillatools.accel import (
        sakaguchi_jacobian as sakaguchi_jacobian,
    )
    from oscillatools.accel import (
        sakaguchi_mean_field_force as sakaguchi_mean_field_force,
    )
    from oscillatools.accel import (
        sakaguchi_mean_field_jacobian as sakaguchi_mean_field_jacobian,
    )
    from oscillatools.accel import (
        sdre_control_input as sdre_control_input,
    )
    from oscillatools.accel import (
        simplex_mean_field_force as simplex_mean_field_force,
    )
    from oscillatools.accel import (
        simplex_mean_field_jacobian as simplex_mean_field_jacobian,
    )
    from oscillatools.accel import (
        simplex_mean_field_term as simplex_mean_field_term,
    )
    from oscillatools.accel import (
        simplicial_hodge_structure as simplicial_hodge_structure,
    )
    from oscillatools.accel import (
        solve_kuramoto_ivp as solve_kuramoto_ivp,
    )
    from oscillatools.accel import (
        sparse_coupling_from_scipy as sparse_coupling_from_scipy,
    )
    from oscillatools.accel import (
        sparse_kuramoto_euler_trajectory as sparse_kuramoto_euler_trajectory,
    )
    from oscillatools.accel import (
        sparse_kuramoto_rk4_trajectory as sparse_kuramoto_rk4_trajectory,
    )
    from oscillatools.accel import (
        sparse_networked_kuramoto_force as sparse_networked_kuramoto_force,
    )
    from oscillatools.accel import (
        stability_spectrum as stability_spectrum,
    )
    from oscillatools.accel import (
        stable_synchronised_frequencies as stable_synchronised_frequencies,
    )
    from oscillatools.accel import (
        stuart_landau_field as stuart_landau_field,
    )
    from oscillatools.accel import (
        stuart_landau_jacobian as stuart_landau_jacobian,
    )
    from oscillatools.accel import (
        stuart_landau_order_parameter as stuart_landau_order_parameter,
    )
    from oscillatools.accel import (
        surrogate_receding_horizon_control as surrogate_receding_horizon_control,
    )
    from oscillatools.accel import (
        surrogate_step as surrogate_step,
    )
    from oscillatools.accel import (
        swarmalator_field as swarmalator_field,
    )
    from oscillatools.accel import (
        swarmalator_order_parameters as swarmalator_order_parameters,
    )
    from oscillatools.accel import (
        sweep_parameter_grid as sweep_parameter_grid,
    )
    from oscillatools.accel import (
        symmetric_nonnegative_projection as symmetric_nonnegative_projection,
    )
    from oscillatools.accel import (
        synchronisation_basin_stability as synchronisation_basin_stability,
    )
    from oscillatools.accel import (
        synchronisation_potential as synchronisation_potential,
    )
    from oscillatools.accel import (
        synchronisation_rate as synchronisation_rate,
    )
    from oscillatools.accel import (
        synchronisation_value_and_grad as synchronisation_value_and_grad,
    )
    from oscillatools.accel import (
        synchronised_branch_stability as synchronised_branch_stability,
    )
    from oscillatools.accel import (
        synchronised_frequency_residual as synchronised_frequency_residual,
    )
    from oscillatools.accel import (
        synchronised_frequency_roots as synchronised_frequency_roots,
    )
    from oscillatools.accel import (
        synchronised_order_parameter as synchronised_order_parameter,
    )
    from oscillatools.accel import (
        terminal_objective_value as terminal_objective_value,
    )
    from oscillatools.accel import (
        terminal_objective_value_and_grad as terminal_objective_value_and_grad,
    )
    from oscillatools.accel import (
        terminal_order_parameter as terminal_order_parameter,
    )
    from oscillatools.accel import (
        theta_from_qif_potential as theta_from_qif_potential,
    )
    from oscillatools.accel import (
        topological_kuramoto_field as topological_kuramoto_field,
    )
    from oscillatools.accel import (
        topological_order_parameter as topological_order_parameter,
    )
    from oscillatools.accel import (
        track_time_varying_coupling as track_time_varying_coupling,
    )
    from oscillatools.accel import (
        trajectory_match_value as trajectory_match_value,
    )
    from oscillatools.accel import (
        trajectory_match_value_and_grad as trajectory_match_value_and_grad,
    )
    from oscillatools.accel import (
        triadic_hysteresis_loop as triadic_hysteresis_loop,
    )
    from oscillatools.accel import (
        triadic_mean_field_force as triadic_mean_field_force,
    )
    from oscillatools.accel import (
        triadic_mean_field_jacobian as triadic_mean_field_jacobian,
    )
    from oscillatools.accel import (
        twisted_state as twisted_state,
    )
    from oscillatools.accel import (
        twisted_state_eigenvalues as twisted_state_eigenvalues,
    )
    from oscillatools.accel import (
        vacuum_state as vacuum_state,
    )
    from oscillatools.accel import (
        watanabe_strogatz_constants as watanabe_strogatz_constants,
    )
    from oscillatools.accel import (
        watanabe_strogatz_invariant as watanabe_strogatz_invariant,
    )
    from oscillatools.accel import (
        watanabe_strogatz_order_parameter as watanabe_strogatz_order_parameter,
    )
    from oscillatools.accel import (
        watanabe_strogatz_phases as watanabe_strogatz_phases,
    )
    from oscillatools.accel import (
        winding_number as winding_number,
    )
    from oscillatools.accel import (
        winfree_field as winfree_field,
    )
    from oscillatools.accel import (
        winfree_jacobian as winfree_jacobian,
    )

import sys as _sys
import warnings as _warnings

import oscillatools.accel as _target

__path__ = list(_target.__path__)

_prefix = _target.__name__ + "."

for _name in [_n for _n in _sys.modules if _n.startswith(_prefix)]:
    _sys.modules[__name__ + "." + _name[len(_prefix) :]] = _sys.modules[_name]

_warnings.warn(
    "scpn_quantum_control.accel has moved to oscillatools.accel; import from "
    "`oscillatools.accel` instead. This compatibility shim will be removed no earlier "
    "than the next major release.",
    DeprecationWarning,
    stacklevel=2,
)

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "MultiLangDispatcher": ("oscillatools.accel", "MultiLangDispatcher"),
    "available_tiers": ("oscillatools.accel", "available_tiers"),
    "daido_mean_field_force": ("oscillatools.accel", "daido_mean_field_force"),
    "daido_mean_field_jacobian": ("oscillatools.accel", "daido_mean_field_jacobian"),
    "last_daido_mean_field_force_tier_used": (
        "oscillatools.accel",
        "last_daido_mean_field_force_tier_used",
    ),
    "last_daido_mean_field_jacobian_tier_used": (
        "oscillatools.accel",
        "last_daido_mean_field_jacobian_tier_used",
    ),
    "daido_mode_phase": ("oscillatools.accel", "daido_mode_phase"),
    "daido_mode_phase_gradient": ("oscillatools.accel", "daido_mode_phase_gradient"),
    "daido_mode_phase_hessian": ("oscillatools.accel", "daido_mode_phase_hessian"),
    "last_daido_mode_phase_gradient_tier_used": (
        "oscillatools.accel",
        "last_daido_mode_phase_gradient_tier_used",
    ),
    "last_daido_mode_phase_hessian_tier_used": (
        "oscillatools.accel",
        "last_daido_mode_phase_hessian_tier_used",
    ),
    "last_daido_mode_phase_tier_used": ("oscillatools.accel", "last_daido_mode_phase_tier_used"),
    "daido_order_parameter": ("oscillatools.accel", "daido_order_parameter"),
    "daido_order_parameter_gradient": ("oscillatools.accel", "daido_order_parameter_gradient"),
    "daido_order_parameter_hessian": ("oscillatools.accel", "daido_order_parameter_hessian"),
    "dispatch": ("oscillatools.accel", "dispatch"),
    "jax_kuramoto_delayed_ensemble": ("oscillatools.accel", "jax_kuramoto_delayed_ensemble"),
    "jax_kuramoto_delayed_ensemble_gradient": (
        "oscillatools.accel",
        "jax_kuramoto_delayed_ensemble_gradient",
    ),
    "jax_kuramoto_delayed_gradient": ("oscillatools.accel", "jax_kuramoto_delayed_gradient"),
    "jax_kuramoto_delayed_trajectory": ("oscillatools.accel", "jax_kuramoto_delayed_trajectory"),
    "jax_kuramoto_dopri_trajectory": ("oscillatools.accel", "jax_kuramoto_dopri_trajectory"),
    "jax_kuramoto_euler_trajectory": ("oscillatools.accel", "jax_kuramoto_euler_trajectory"),
    "jax_kuramoto_rk4_ensemble": ("oscillatools.accel", "jax_kuramoto_rk4_ensemble"),
    "jax_kuramoto_rk4_ensemble_gradient": (
        "oscillatools.accel",
        "jax_kuramoto_rk4_ensemble_gradient",
    ),
    "jax_kuramoto_rk4_gradient": ("oscillatools.accel", "jax_kuramoto_rk4_gradient"),
    "jax_kuramoto_rk4_trajectory": ("oscillatools.accel", "jax_kuramoto_rk4_trajectory"),
    "jax_networked_inertial_trajectory": (
        "oscillatools.accel",
        "jax_networked_inertial_trajectory",
    ),
    "jax_networked_noisy_trajectory": ("oscillatools.accel", "jax_networked_noisy_trajectory"),
    "jax_networked_symplectic_inertial_trajectory": (
        "oscillatools.accel",
        "jax_networked_symplectic_inertial_trajectory",
    ),
    "MpcControlGradients": ("oscillatools.accel", "MpcControlGradients"),
    "MpcOptimumSensitivity": ("oscillatools.accel", "MpcOptimumSensitivity"),
    "RecedingHorizonResult": ("oscillatools.accel", "RecedingHorizonResult"),
    "jax_mpc_control_value_and_grad": ("oscillatools.accel", "jax_mpc_control_value_and_grad"),
    "jax_mpc_horizon_control": ("oscillatools.accel", "jax_mpc_horizon_control"),
    "jax_mpc_optimum": ("oscillatools.accel", "jax_mpc_optimum"),
    "mpc_optimum_parameter_sensitivity": (
        "oscillatools.accel",
        "mpc_optimum_parameter_sensitivity",
    ),
    "mpc_plan_energy_gradient": ("oscillatools.accel", "mpc_plan_energy_gradient"),
    "receding_horizon_control": ("oscillatools.accel", "receding_horizon_control"),
    "SurrogateStepModel": ("oscillatools.accel", "SurrogateStepModel"),
    "SurrogateControlComparison": ("oscillatools.accel", "SurrogateControlComparison"),
    "fit_surrogate_step_model": ("oscillatools.accel", "fit_surrogate_step_model"),
    "surrogate_step": ("oscillatools.accel", "surrogate_step"),
    "surrogate_receding_horizon_control": (
        "oscillatools.accel",
        "surrogate_receding_horizon_control",
    ),
    "compare_surrogate_control": ("oscillatools.accel", "compare_surrogate_control"),
    "TerminalObjective": ("oscillatools.accel", "TerminalObjective"),
    "coherence_objective": ("oscillatools.accel", "coherence_objective"),
    "interaction_energy_objective": ("oscillatools.accel", "interaction_energy_objective"),
    "phase_target_objective": ("oscillatools.accel", "phase_target_objective"),
    "synchronisation_value_and_grad": ("oscillatools.accel", "synchronisation_value_and_grad"),
    "terminal_objective_value": ("oscillatools.accel", "terminal_objective_value"),
    "terminal_objective_value_and_grad": (
        "oscillatools.accel",
        "terminal_objective_value_and_grad",
    ),
    "CouplingDesignResult": ("oscillatools.accel", "CouplingDesignResult"),
    "CouplingProjection": ("oscillatools.accel", "CouplingProjection"),
    "design_synchronising_coupling": ("oscillatools.accel", "design_synchronising_coupling"),
    "optimise_coupling": ("oscillatools.accel", "optimise_coupling"),
    "symmetric_nonnegative_projection": ("oscillatools.accel", "symmetric_nonnegative_projection"),
    "PinningDesignResult": ("oscillatools.accel", "PinningDesignResult"),
    "design_pinning": ("oscillatools.accel", "design_pinning"),
    "pinning_coherence_value": ("oscillatools.accel", "pinning_coherence_value"),
    "pinning_coherence_value_and_grad": ("oscillatools.accel", "pinning_coherence_value_and_grad"),
    "SystemIdentificationResult": ("oscillatools.accel", "SystemIdentificationResult"),
    "learn_coupling": ("oscillatools.accel", "learn_coupling"),
    "trajectory_match_value": ("oscillatools.accel", "trajectory_match_value"),
    "trajectory_match_value_and_grad": ("oscillatools.accel", "trajectory_match_value_and_grad"),
    "phase_raster": ("oscillatools.accel", "phase_raster"),
    "order_parameter_timeseries": ("oscillatools.accel", "order_parameter_timeseries"),
    "chimera_snapshot": ("oscillatools.accel", "chimera_snapshot"),
    "network_phase_embedding": ("oscillatools.accel", "network_phase_embedding"),
    "StabilitySpectrum": ("oscillatools.accel", "StabilitySpectrum"),
    "is_synchronisation_stable": ("oscillatools.accel", "is_synchronisation_stable"),
    "stability_spectrum": ("oscillatools.accel", "stability_spectrum"),
    "synchronisation_rate": ("oscillatools.accel", "synchronisation_rate"),
    "SaddleNodePoint": ("oscillatools.accel", "SaddleNodePoint"),
    "fold_defining_jacobian": ("oscillatools.accel", "fold_defining_jacobian"),
    "fold_defining_residual": ("oscillatools.accel", "fold_defining_residual"),
    "locate_saddle_node": ("oscillatools.accel", "locate_saddle_node"),
    "critical_coupling": ("oscillatools.accel", "critical_coupling"),
    "gaussian_critical_coupling": ("oscillatools.accel", "gaussian_critical_coupling"),
    "gaussian_density": ("oscillatools.accel", "gaussian_density"),
    "lorentzian_critical_coupling": ("oscillatools.accel", "lorentzian_critical_coupling"),
    "lorentzian_density": ("oscillatools.accel", "lorentzian_density"),
    "lorentzian_order_parameter": ("oscillatools.accel", "lorentzian_order_parameter"),
    "synchronised_order_parameter": ("oscillatools.accel", "synchronised_order_parameter"),
    "ott_antonsen_field": ("oscillatools.accel", "ott_antonsen_field"),
    "ott_antonsen_order_parameter": ("oscillatools.accel", "ott_antonsen_order_parameter"),
    "ott_antonsen_steady_state": ("oscillatools.accel", "ott_antonsen_steady_state"),
    "ott_antonsen_terminal_order_parameter_value_and_grad": (
        "oscillatools.accel",
        "ott_antonsen_terminal_order_parameter_value_and_grad",
    ),
    "ott_antonsen_trajectory": ("oscillatools.accel", "ott_antonsen_trajectory"),
    "lyapunov_spectrum": ("oscillatools.accel", "lyapunov_spectrum"),
    "maximal_lyapunov_exponent": ("oscillatools.accel", "maximal_lyapunov_exponent"),
    "ContinuationBranch": ("oscillatools.accel", "ContinuationBranch"),
    "HysteresisLoop": ("oscillatools.accel", "HysteresisLoop"),
    "MeanFieldForce": ("oscillatools.accel", "MeanFieldForce"),
    "continuation_sweep": ("oscillatools.accel", "continuation_sweep"),
    "hysteresis_loop": ("oscillatools.accel", "hysteresis_loop"),
    "triadic_hysteresis_loop": ("oscillatools.accel", "triadic_hysteresis_loop"),
    "InertialTrajectory": ("oscillatools.accel", "InertialTrajectory"),
    "PhaseForce": ("oscillatools.accel", "PhaseForce"),
    "PhaseJacobian": ("oscillatools.accel", "PhaseJacobian"),
    "PhasePotential": ("oscillatools.accel", "PhasePotential"),
    "inertial_energy": ("oscillatools.accel", "inertial_energy"),
    "inertial_jacobian": ("oscillatools.accel", "inertial_jacobian"),
    "inertial_vector_field": ("oscillatools.accel", "inertial_vector_field"),
    "integrate_inertial": ("oscillatools.accel", "integrate_inertial"),
    "InertialGradients": ("oscillatools.accel", "InertialGradients"),
    "inertial_state_sensitivity": ("oscillatools.accel", "inertial_state_sensitivity"),
    "inertial_terminal_value_and_grad": ("oscillatools.accel", "inertial_terminal_value_and_grad"),
    "AdaptiveGradients": ("oscillatools.accel", "AdaptiveGradients"),
    "adaptive_state_sensitivity": ("oscillatools.accel", "adaptive_state_sensitivity"),
    "adaptive_terminal_value_and_grad": ("oscillatools.accel", "adaptive_terminal_value_and_grad"),
    "NoisyGradients": ("oscillatools.accel", "NoisyGradients"),
    "noisy_phase_sensitivity": ("oscillatools.accel", "noisy_phase_sensitivity"),
    "noisy_terminal_value_and_grad": ("oscillatools.accel", "noisy_terminal_value_and_grad"),
    "DelayedGradients": ("oscillatools.accel", "DelayedGradients"),
    "delayed_delay_gradient": ("oscillatools.accel", "delayed_delay_gradient"),
    "delayed_delay_sensitivity": ("oscillatools.accel", "delayed_delay_sensitivity"),
    "delayed_phase_sensitivity": ("oscillatools.accel", "delayed_phase_sensitivity"),
    "delayed_terminal_value_and_grad": ("oscillatools.accel", "delayed_terminal_value_and_grad"),
    "QifMeanFieldGradients": ("oscillatools.accel", "QifMeanFieldGradients"),
    "QifMeanFieldTrajectory": ("oscillatools.accel", "QifMeanFieldTrajectory"),
    "integrate_qif_mean_field": ("oscillatools.accel", "integrate_qif_mean_field"),
    "kuramoto_order_parameter_from_macro": (
        "oscillatools.accel",
        "kuramoto_order_parameter_from_macro",
    ),
    "macro_from_kuramoto_order_parameter": (
        "oscillatools.accel",
        "macro_from_kuramoto_order_parameter",
    ),
    "qif_mean_field_fixed_point": ("oscillatools.accel", "qif_mean_field_fixed_point"),
    "qif_mean_field_jacobian": ("oscillatools.accel", "qif_mean_field_jacobian"),
    "qif_mean_field_rates": ("oscillatools.accel", "qif_mean_field_rates"),
    "qif_mean_field_terminal_value_and_grad": (
        "oscillatools.accel",
        "qif_mean_field_terminal_value_and_grad",
    ),
    "qif_potential_from_theta": ("oscillatools.accel", "qif_potential_from_theta"),
    "theta_from_qif_potential": ("oscillatools.accel", "theta_from_qif_potential"),
    "WatanabeStrogatzTrajectory": ("oscillatools.accel", "WatanabeStrogatzTrajectory"),
    "integrate_watanabe_strogatz": ("oscillatools.accel", "integrate_watanabe_strogatz"),
    "watanabe_strogatz_constants": ("oscillatools.accel", "watanabe_strogatz_constants"),
    "watanabe_strogatz_invariant": ("oscillatools.accel", "watanabe_strogatz_invariant"),
    "watanabe_strogatz_order_parameter": (
        "oscillatools.accel",
        "watanabe_strogatz_order_parameter",
    ),
    "watanabe_strogatz_phases": ("oscillatools.accel", "watanabe_strogatz_phases"),
    "CoordinatedResetGradients": ("oscillatools.accel", "CoordinatedResetGradients"),
    "CoordinatedResetTrajectory": ("oscillatools.accel", "CoordinatedResetTrajectory"),
    "coordinated_reset_phases": ("oscillatools.accel", "coordinated_reset_phases"),
    "coordinated_reset_sites": ("oscillatools.accel", "coordinated_reset_sites"),
    "coordinated_reset_terminal_value_and_grad": (
        "oscillatools.accel",
        "coordinated_reset_terminal_value_and_grad",
    ),
    "integrate_coordinated_reset": ("oscillatools.accel", "integrate_coordinated_reset"),
    "BasinStabilityEstimate": ("oscillatools.accel", "BasinStabilityEstimate"),
    "synchronisation_basin_stability": ("oscillatools.accel", "synchronisation_basin_stability"),
    "integrate_symplectic_inertial": ("oscillatools.accel", "integrate_symplectic_inertial"),
    "KuramotoParameters": ("oscillatools.accel", "KuramotoParameters"),
    "KuramotoSystem": ("oscillatools.accel", "KuramotoSystem"),
    "KuramotoIvpSolution": ("oscillatools.accel", "KuramotoIvpSolution"),
    "kuramoto_ode_rhs": ("oscillatools.accel", "kuramoto_ode_rhs"),
    "kuramoto_ode_jacobian": ("oscillatools.accel", "kuramoto_ode_jacobian"),
    "solve_kuramoto_ivp": ("oscillatools.accel", "solve_kuramoto_ivp"),
    "mean_field_phase_rule": ("oscillatools.accel", "mean_field_phase_rule"),
    "mean_field_phase_rule_jacobian": ("oscillatools.accel", "mean_field_phase_rule_jacobian"),
    "networked_phase_rule": ("oscillatools.accel", "networked_phase_rule"),
    "networked_phase_rule_jacobian": ("oscillatools.accel", "networked_phase_rule_jacobian"),
    "KuramotoParameterGrid": ("oscillatools.accel", "KuramotoParameterGrid"),
    "Observable": ("oscillatools.accel", "Observable"),
    "ParameterSweepResult": ("oscillatools.accel", "ParameterSweepResult"),
    "sweep_parameter_grid": ("oscillatools.accel", "sweep_parameter_grid"),
    "mean_order_parameter": ("oscillatools.accel", "mean_order_parameter"),
    "terminal_order_parameter": ("oscillatools.accel", "terminal_order_parameter"),
    "metastability": ("oscillatools.accel", "metastability"),
    "frequency_spread": ("oscillatools.accel", "frequency_spread"),
    "PseudoArclengthBranch": ("oscillatools.accel", "PseudoArclengthBranch"),
    "pseudo_arclength_continuation": ("oscillatools.accel", "pseudo_arclength_continuation"),
    "CollectiveControlGradients": ("oscillatools.accel", "CollectiveControlGradients"),
    "ForcedCollectiveTrajectory": ("oscillatools.accel", "ForcedCollectiveTrajectory"),
    "collective_control_value_and_grad": (
        "oscillatools.accel",
        "collective_control_value_and_grad",
    ),
    "integrate_forced_collective": ("oscillatools.accel", "integrate_forced_collective"),
    "optimise_collective_forcing": ("oscillatools.accel", "optimise_collective_forcing"),
    "ControlledNetworkTrajectory": ("oscillatools.accel", "ControlledNetworkTrajectory"),
    "NetworkControlGradients": ("oscillatools.accel", "NetworkControlGradients"),
    "integrate_controlled_network": ("oscillatools.accel", "integrate_controlled_network"),
    "network_control_value_and_grad": ("oscillatools.accel", "network_control_value_and_grad"),
    "optimise_network_control": ("oscillatools.accel", "optimise_network_control"),
    "CouplingFunctionEstimate": ("oscillatools.accel", "CouplingFunctionEstimate"),
    "CouplingFunctionGradients": ("oscillatools.accel", "CouplingFunctionGradients"),
    "coupling_function_value": ("oscillatools.accel", "coupling_function_value"),
    "infer_coupling_function": ("oscillatools.accel", "infer_coupling_function"),
    "coupling_function_trajectory_value_and_grad": (
        "oscillatools.accel",
        "coupling_function_trajectory_value_and_grad",
    ),
    "refine_coupling_function": ("oscillatools.accel", "refine_coupling_function"),
    "DynamicalBayesianPosterior": ("oscillatools.accel", "DynamicalBayesianPosterior"),
    "TimeVaryingCouplingHistory": ("oscillatools.accel", "TimeVaryingCouplingHistory"),
    "infer_network_bayesian": ("oscillatools.accel", "infer_network_bayesian"),
    "track_time_varying_coupling": ("oscillatools.accel", "track_time_varying_coupling"),
    "kuramoto_sdre_gain": ("oscillatools.accel", "kuramoto_sdre_gain"),
    "sdre_control_input": ("oscillatools.accel", "sdre_control_input"),
    "integrate_sdre_controlled_kuramoto": (
        "oscillatools.accel",
        "integrate_sdre_controlled_kuramoto",
    ),
    "DesynchronisingPolicy": ("oscillatools.accel", "DesynchronisingPolicy"),
    "PolicyRolloutGradients": ("oscillatools.accel", "PolicyRolloutGradients"),
    "policy_rollout_value_and_grad": ("oscillatools.accel", "policy_rollout_value_and_grad"),
    "learn_desynchronising_policy": ("oscillatools.accel", "learn_desynchronising_policy"),
    "OscillatorIsingTrajectory": ("oscillatools.accel", "OscillatorIsingTrajectory"),
    "oscillator_ising_field": ("oscillatools.accel", "oscillator_ising_field"),
    "oscillator_ising_energy": ("oscillatools.accel", "oscillator_ising_energy"),
    "ising_spins": ("oscillatools.accel", "ising_spins"),
    "ising_hamiltonian": ("oscillatools.accel", "ising_hamiltonian"),
    "cut_value": ("oscillatools.accel", "cut_value"),
    "integrate_oscillator_ising_machine": (
        "oscillatools.accel",
        "integrate_oscillator_ising_machine",
    ),
    "QuantumVanDerPolTrajectory": ("oscillatools.accel", "QuantumVanDerPolTrajectory"),
    "vacuum_state": ("oscillatools.accel", "vacuum_state"),
    "integrate_quantum_vanderpol": ("oscillatools.accel", "integrate_quantum_vanderpol"),
    "coherent_amplitude": ("oscillatools.accel", "coherent_amplitude"),
    "mean_photon_number": ("oscillatools.accel", "mean_photon_number"),
    "phase_distribution": ("oscillatools.accel", "phase_distribution"),
    "phase_synchronisation": ("oscillatools.accel", "phase_synchronisation"),
    "HodgeStructure": ("oscillatools.accel", "HodgeStructure"),
    "HodgeComponents": ("oscillatools.accel", "HodgeComponents"),
    "TopologicalKuramotoTrajectory": ("oscillatools.accel", "TopologicalKuramotoTrajectory"),
    "simplicial_hodge_structure": ("oscillatools.accel", "simplicial_hodge_structure"),
    "hodge_decomposition": ("oscillatools.accel", "hodge_decomposition"),
    "topological_kuramoto_field": ("oscillatools.accel", "topological_kuramoto_field"),
    "integrate_topological_kuramoto": ("oscillatools.accel", "integrate_topological_kuramoto"),
    "topological_order_parameter": ("oscillatools.accel", "topological_order_parameter"),
    "SynchronisationCertificate": ("oscillatools.accel", "SynchronisationCertificate"),
    "phase_cohesiveness": ("oscillatools.accel", "phase_cohesiveness"),
    "contraction_rate": ("oscillatools.accel", "contraction_rate"),
    "synchronisation_potential": ("oscillatools.accel", "synchronisation_potential"),
    "potential_decrease_rate": ("oscillatools.accel", "potential_decrease_rate"),
    "certify_synchronisation": ("oscillatools.accel", "certify_synchronisation"),
    "NeuralLyapunovCertificate": ("oscillatools.accel", "NeuralLyapunovCertificate"),
    "LyapunovCertificateReport": ("oscillatools.accel", "LyapunovCertificateReport"),
    "LyapunovCounterexample": ("oscillatools.accel", "LyapunovCounterexample"),
    "fit_neural_lyapunov_certificate": ("oscillatools.accel", "fit_neural_lyapunov_certificate"),
    "certify_neural_lyapunov": ("oscillatools.accel", "certify_neural_lyapunov"),
    "falsify_neural_lyapunov": ("oscillatools.accel", "falsify_neural_lyapunov"),
    "neural_lyapunov_value": ("oscillatools.accel", "neural_lyapunov_value"),
    "neural_lyapunov_decrease": ("oscillatools.accel", "neural_lyapunov_decrease"),
    "FormalLyapunovCertificate": ("oscillatools.accel", "FormalLyapunovCertificate"),
    "LyapunovLipschitzBounds": ("oscillatools.accel", "LyapunovLipschitzBounds"),
    "neural_lyapunov_lipschitz_bounds": ("oscillatools.accel", "neural_lyapunov_lipschitz_bounds"),
    "formally_certify_neural_lyapunov": ("oscillatools.accel", "formally_certify_neural_lyapunov"),
    "PermutationSignificanceResult": ("oscillatools.accel", "PermutationSignificanceResult"),
    "permutation_significance_test": ("oscillatools.accel", "permutation_significance_test"),
    "SparseDynamicsModel": ("oscillatools.accel", "SparseDynamicsModel"),
    "discover_phase_dynamics": ("oscillatools.accel", "discover_phase_dynamics"),
    "SparseDynamicsEstimator": ("oscillatools.accel", "SparseDynamicsEstimator"),
    "CouplingFunctionEstimator": ("oscillatools.accel", "CouplingFunctionEstimator"),
    "StuartLandauTrajectory": ("oscillatools.accel", "StuartLandauTrajectory"),
    "stuart_landau_field": ("oscillatools.accel", "stuart_landau_field"),
    "stuart_landau_jacobian": ("oscillatools.accel", "stuart_landau_jacobian"),
    "integrate_stuart_landau": ("oscillatools.accel", "integrate_stuart_landau"),
    "amplitudes": ("oscillatools.accel", "amplitudes"),
    "stuart_landau_order_parameter": ("oscillatools.accel", "stuart_landau_order_parameter"),
    "is_oscillation_death": ("oscillatools.accel", "is_oscillation_death"),
    "SwarmalatorTrajectory": ("oscillatools.accel", "SwarmalatorTrajectory"),
    "SwarmalatorOrderParameters": ("oscillatools.accel", "SwarmalatorOrderParameters"),
    "swarmalator_field": ("oscillatools.accel", "swarmalator_field"),
    "integrate_swarmalators": ("oscillatools.accel", "integrate_swarmalators"),
    "swarmalator_order_parameters": ("oscillatools.accel", "swarmalator_order_parameters"),
    "WinfreeTrajectory": ("oscillatools.accel", "WinfreeTrajectory"),
    "winfree_field": ("oscillatools.accel", "winfree_field"),
    "winfree_jacobian": ("oscillatools.accel", "winfree_jacobian"),
    "integrate_winfree": ("oscillatools.accel", "integrate_winfree"),
    "MultiplexTrajectory": ("oscillatools.accel", "MultiplexTrajectory"),
    "multiplex_field": ("oscillatools.accel", "multiplex_field"),
    "multiplex_jacobian": ("oscillatools.accel", "multiplex_jacobian"),
    "integrate_multiplex": ("oscillatools.accel", "integrate_multiplex"),
    "layer_order_parameters": ("oscillatools.accel", "layer_order_parameters"),
    "interlayer_synchronisation": ("oscillatools.accel", "interlayer_synchronisation"),
    "MultiplexSynchronisationStability": (
        "oscillatools.accel",
        "MultiplexSynchronisationStability",
    ),
    "master_stability_function": ("oscillatools.accel", "master_stability_function"),
    "multiplex_synchronisation_stability": (
        "oscillatools.accel",
        "multiplex_synchronisation_stability",
    ),
    "HigherOrderWatanabeStrogatzTrajectory": (
        "oscillatools.accel",
        "HigherOrderWatanabeStrogatzTrajectory",
    ),
    "integrate_higher_order_watanabe_strogatz": (
        "oscillatools.accel",
        "integrate_higher_order_watanabe_strogatz",
    ),
    "BasinEstimate": ("oscillatools.accel", "BasinEstimate"),
    "estimate_ring_basins": ("oscillatools.accel", "estimate_ring_basins"),
    "is_twisted_state_stable": ("oscillatools.accel", "is_twisted_state_stable"),
    "ring_coupling_matrix": ("oscillatools.accel", "ring_coupling_matrix"),
    "twisted_state": ("oscillatools.accel", "twisted_state"),
    "twisted_state_eigenvalues": ("oscillatools.accel", "twisted_state_eigenvalues"),
    "winding_number": ("oscillatools.accel", "winding_number"),
    "DopriTrajectory": ("oscillatools.accel", "DopriTrajectory"),
    "kuramoto_dopri_trajectory": ("oscillatools.accel", "kuramoto_dopri_trajectory"),
    "kuramoto_dopri_vjp": ("oscillatools.accel", "kuramoto_dopri_vjp"),
    "last_kuramoto_dopri_trajectory_tier_used": (
        "oscillatools.accel",
        "last_kuramoto_dopri_trajectory_tier_used",
    ),
    "kuramoto_euler_trajectory": ("oscillatools.accel", "kuramoto_euler_trajectory"),
    "kuramoto_euler_vjp": ("oscillatools.accel", "kuramoto_euler_vjp"),
    "kuramoto_rk4_trajectory": ("oscillatools.accel", "kuramoto_rk4_trajectory"),
    "kuramoto_rk4_vjp": ("oscillatools.accel", "kuramoto_rk4_vjp"),
    "last_kuramoto_rk4_trajectory_tier_used": (
        "oscillatools.accel",
        "last_kuramoto_rk4_trajectory_tier_used",
    ),
    "last_kuramoto_rk4_vjp_tier_used": ("oscillatools.accel", "last_kuramoto_rk4_vjp_tier_used"),
    "last_kuramoto_euler_trajectory_tier_used": (
        "oscillatools.accel",
        "last_kuramoto_euler_trajectory_tier_used",
    ),
    "last_kuramoto_euler_vjp_tier_used": (
        "oscillatools.accel",
        "last_kuramoto_euler_vjp_tier_used",
    ),
    "kuramoto_interaction_energy": ("oscillatools.accel", "kuramoto_interaction_energy"),
    "kuramoto_interaction_energy_gradient": (
        "oscillatools.accel",
        "kuramoto_interaction_energy_gradient",
    ),
    "kuramoto_interaction_energy_hessian": (
        "oscillatools.accel",
        "kuramoto_interaction_energy_hessian",
    ),
    "last_kuramoto_interaction_energy_gradient_tier_used": (
        "oscillatools.accel",
        "last_kuramoto_interaction_energy_gradient_tier_used",
    ),
    "last_kuramoto_interaction_energy_hessian_tier_used": (
        "oscillatools.accel",
        "last_kuramoto_interaction_energy_hessian_tier_used",
    ),
    "last_kuramoto_interaction_energy_tier_used": (
        "oscillatools.accel",
        "last_kuramoto_interaction_energy_tier_used",
    ),
    "last_daido_gradient_tier_used": ("oscillatools.accel", "last_daido_gradient_tier_used"),
    "last_daido_hessian_tier_used": ("oscillatools.accel", "last_daido_hessian_tier_used"),
    "last_daido_tier_used": ("oscillatools.accel", "last_daido_tier_used"),
    "last_gradient_tier_used": ("oscillatools.accel", "last_gradient_tier_used"),
    "last_hessian_tier_used": ("oscillatools.accel", "last_hessian_tier_used"),
    "last_local_order_parameter_jacobian_tier_used": (
        "oscillatools.accel",
        "last_local_order_parameter_jacobian_tier_used",
    ),
    "last_local_order_parameter_tier_used": (
        "oscillatools.accel",
        "last_local_order_parameter_tier_used",
    ),
    "local_mean_phase": ("oscillatools.accel", "local_mean_phase"),
    "local_mean_phase_jacobian": ("oscillatools.accel", "local_mean_phase_jacobian"),
    "last_local_mean_phase_tier_used": ("oscillatools.accel", "last_local_mean_phase_tier_used"),
    "last_local_mean_phase_jacobian_tier_used": (
        "oscillatools.accel",
        "last_local_mean_phase_jacobian_tier_used",
    ),
    "local_order_parameter": ("oscillatools.accel", "local_order_parameter"),
    "local_order_parameter_jacobian": ("oscillatools.accel", "local_order_parameter_jacobian"),
    "last_mean_field_force_tier_used": ("oscillatools.accel", "last_mean_field_force_tier_used"),
    "last_mean_field_jacobian_tier_used": (
        "oscillatools.accel",
        "last_mean_field_jacobian_tier_used",
    ),
    "last_mean_phase_gradient_tier_used": (
        "oscillatools.accel",
        "last_mean_phase_gradient_tier_used",
    ),
    "last_mean_phase_hessian_tier_used": (
        "oscillatools.accel",
        "last_mean_phase_hessian_tier_used",
    ),
    "last_mean_phase_tier_used": ("oscillatools.accel", "last_mean_phase_tier_used"),
    "last_tier_used": ("oscillatools.accel", "last_tier_used"),
    "mean_field_force": ("oscillatools.accel", "mean_field_force"),
    "mean_field_jacobian": ("oscillatools.accel", "mean_field_jacobian"),
    "NoisyKuramotoRun": ("oscillatools.accel", "NoisyKuramotoRun"),
    "StochasticForce": ("oscillatools.accel", "StochasticForce"),
    "integrate_noisy_kuramoto": ("oscillatools.accel", "integrate_noisy_kuramoto"),
    "noisy_kuramoto_step": ("oscillatools.accel", "noisy_kuramoto_step"),
    "FrequencyDensity": ("oscillatools.accel", "FrequencyDensity"),
    "lorentzian_noisy_critical_coupling": (
        "oscillatools.accel",
        "lorentzian_noisy_critical_coupling",
    ),
    "noisy_critical_coupling": ("oscillatools.accel", "noisy_critical_coupling"),
    "noisy_stationary_order_parameter": ("oscillatools.accel", "noisy_stationary_order_parameter"),
    "DelayedForce": ("oscillatools.accel", "DelayedForce"),
    "DelayedTrajectory": ("oscillatools.accel", "DelayedTrajectory"),
    "delayed_mean_field_force": ("oscillatools.accel", "delayed_mean_field_force"),
    "delayed_networked_force": ("oscillatools.accel", "delayed_networked_force"),
    "integrate_delayed_kuramoto": ("oscillatools.accel", "integrate_delayed_kuramoto"),
    "is_synchronised_branch_stable": ("oscillatools.accel", "is_synchronised_branch_stable"),
    "stable_synchronised_frequencies": ("oscillatools.accel", "stable_synchronised_frequencies"),
    "synchronised_branch_stability": ("oscillatools.accel", "synchronised_branch_stability"),
    "synchronised_frequency_residual": ("oscillatools.accel", "synchronised_frequency_residual"),
    "synchronised_frequency_roots": ("oscillatools.accel", "synchronised_frequency_roots"),
    "AdaptivePhaseForce": ("oscillatools.accel", "AdaptivePhaseForce"),
    "AdaptiveTrajectory": ("oscillatools.accel", "AdaptiveTrajectory"),
    "PlasticityRule": ("oscillatools.accel", "PlasticityRule"),
    "adaptive_vector_field": ("oscillatools.accel", "adaptive_vector_field"),
    "hebbian_adaptive_jacobian": ("oscillatools.accel", "hebbian_adaptive_jacobian"),
    "hebbian_coupling_equilibrium": ("oscillatools.accel", "hebbian_coupling_equilibrium"),
    "hebbian_plasticity_rate": ("oscillatools.accel", "hebbian_plasticity_rate"),
    "integrate_adaptive_kuramoto": ("oscillatools.accel", "integrate_adaptive_kuramoto"),
    "simplex_mean_field_force": ("oscillatools.accel", "simplex_mean_field_force"),
    "simplex_mean_field_jacobian": ("oscillatools.accel", "simplex_mean_field_jacobian"),
    "hyperedge_force": ("oscillatools.accel", "hyperedge_force"),
    "hyperedge_jacobian": ("oscillatools.accel", "hyperedge_jacobian"),
    "CouplingTerm": ("oscillatools.accel", "CouplingTerm"),
    "heterogeneous_force": ("oscillatools.accel", "heterogeneous_force"),
    "heterogeneous_force_components": ("oscillatools.accel", "heterogeneous_force_components"),
    "heterogeneous_jacobian": ("oscillatools.accel", "heterogeneous_jacobian"),
    "hyperedge_term": ("oscillatools.accel", "hyperedge_term"),
    "pairwise_term": ("oscillatools.accel", "pairwise_term"),
    "simplex_mean_field_term": ("oscillatools.accel", "simplex_mean_field_term"),
    "FrequencyOrder": ("oscillatools.accel", "FrequencyOrder"),
    "effective_frequencies": ("oscillatools.accel", "effective_frequencies"),
    "frequency_locked_fraction": ("oscillatools.accel", "frequency_locked_fraction"),
    "frequency_order_diagnostics": ("oscillatools.accel", "frequency_order_diagnostics"),
    "frequency_synchronisation_index": ("oscillatools.accel", "frequency_synchronisation_index"),
    "frequency_synchronisation_index_gradient": (
        "oscillatools.accel",
        "frequency_synchronisation_index_gradient",
    ),
    "ChimeraDiagnostics": ("oscillatools.accel", "ChimeraDiagnostics"),
    "chimera_diagnostics": ("oscillatools.accel", "chimera_diagnostics"),
    "chimera_index": ("oscillatools.accel", "chimera_index"),
    "chimera_index_gradient": ("oscillatools.accel", "chimera_index_gradient"),
    "community_metastability": ("oscillatools.accel", "community_metastability"),
    "community_order_parameters": ("oscillatools.accel", "community_order_parameters"),
    "metastability_index": ("oscillatools.accel", "metastability_index"),
    "metastability_index_gradient": ("oscillatools.accel", "metastability_index_gradient"),
    "mutual_information_matrix": ("oscillatools.accel", "mutual_information_matrix"),
    "normalised_phase_entropy": ("oscillatools.accel", "normalised_phase_entropy"),
    "pairwise_mutual_information": ("oscillatools.accel", "pairwise_mutual_information"),
    "phase_entropy": ("oscillatools.accel", "phase_entropy"),
    "phase_entropy_series": ("oscillatools.accel", "phase_entropy_series"),
    "coherence_matrix": ("oscillatools.accel", "coherence_matrix"),
    "coherence_spectrum": ("oscillatools.accel", "coherence_spectrum"),
    "leading_coherence_eigenvector": ("oscillatools.accel", "leading_coherence_eigenvector"),
    "mean_coherence_matrix": ("oscillatools.accel", "mean_coherence_matrix"),
    "phase_locking_matrix": ("oscillatools.accel", "phase_locking_matrix"),
    "ClusterPartition": ("oscillatools.accel", "ClusterPartition"),
    "cluster_count": ("oscillatools.accel", "cluster_count"),
    "cluster_partition": ("oscillatools.accel", "cluster_partition"),
    "phase_clusters": ("oscillatools.accel", "phase_clusters"),
    "mean_phase": ("oscillatools.accel", "mean_phase"),
    "mean_phase_gradient": ("oscillatools.accel", "mean_phase_gradient"),
    "mean_phase_hessian": ("oscillatools.accel", "mean_phase_hessian"),
    "networked_kuramoto_force": ("oscillatools.accel", "networked_kuramoto_force"),
    "networked_kuramoto_jacobian": ("oscillatools.accel", "networked_kuramoto_jacobian"),
    "SparseKuramotoCoupling": ("oscillatools.accel", "SparseKuramotoCoupling"),
    "ring_sparse_coupling": ("oscillatools.accel", "ring_sparse_coupling"),
    "sparse_coupling_from_scipy": ("oscillatools.accel", "sparse_coupling_from_scipy"),
    "sparse_networked_kuramoto_force": ("oscillatools.accel", "sparse_networked_kuramoto_force"),
    "sparse_kuramoto_euler_trajectory": ("oscillatools.accel", "sparse_kuramoto_euler_trajectory"),
    "sparse_kuramoto_rk4_trajectory": ("oscillatools.accel", "sparse_kuramoto_rk4_trajectory"),
    "last_networked_kuramoto_force_tier_used": (
        "oscillatools.accel",
        "last_networked_kuramoto_force_tier_used",
    ),
    "last_networked_kuramoto_jacobian_tier_used": (
        "oscillatools.accel",
        "last_networked_kuramoto_jacobian_tier_used",
    ),
    "GraphLike": ("oscillatools.accel", "GraphLike"),
    "coupling_from_networkx": ("oscillatools.accel", "coupling_from_networkx"),
    "graph_from_networked_coupling": ("oscillatools.accel", "graph_from_networked_coupling"),
    "networked_delayed_trajectory": ("oscillatools.accel", "networked_delayed_trajectory"),
    "last_networked_delayed_trajectory_tier_used": (
        "oscillatools.accel",
        "last_networked_delayed_trajectory_tier_used",
    ),
    "networked_inertial_trajectory": ("oscillatools.accel", "networked_inertial_trajectory"),
    "last_networked_inertial_trajectory_tier_used": (
        "oscillatools.accel",
        "last_networked_inertial_trajectory_tier_used",
    ),
    "networked_noisy_trajectory": ("oscillatools.accel", "networked_noisy_trajectory"),
    "last_networked_noisy_trajectory_tier_used": (
        "oscillatools.accel",
        "last_networked_noisy_trajectory_tier_used",
    ),
    "networked_symplectic_inertial_trajectory": (
        "oscillatools.accel",
        "networked_symplectic_inertial_trajectory",
    ),
    "last_networked_symplectic_inertial_trajectory_tier_used": (
        "oscillatools.accel",
        "last_networked_symplectic_inertial_trajectory_tier_used",
    ),
    "order_parameter": ("oscillatools.accel", "order_parameter"),
    "order_parameter_gradient": ("oscillatools.accel", "order_parameter_gradient"),
    "order_parameter_hessian": ("oscillatools.accel", "order_parameter_hessian"),
    "sakaguchi_mean_field_force": ("oscillatools.accel", "sakaguchi_mean_field_force"),
    "sakaguchi_mean_field_jacobian": ("oscillatools.accel", "sakaguchi_mean_field_jacobian"),
    "triadic_mean_field_force": ("oscillatools.accel", "triadic_mean_field_force"),
    "triadic_mean_field_jacobian": ("oscillatools.accel", "triadic_mean_field_jacobian"),
    "last_triadic_mean_field_force_tier_used": (
        "oscillatools.accel",
        "last_triadic_mean_field_force_tier_used",
    ),
    "last_triadic_mean_field_jacobian_tier_used": (
        "oscillatools.accel",
        "last_triadic_mean_field_jacobian_tier_used",
    ),
    "last_sakaguchi_mean_field_force_tier_used": (
        "oscillatools.accel",
        "last_sakaguchi_mean_field_force_tier_used",
    ),
    "last_sakaguchi_mean_field_jacobian_tier_used": (
        "oscillatools.accel",
        "last_sakaguchi_mean_field_jacobian_tier_used",
    ),
    "sakaguchi_force": ("oscillatools.accel", "sakaguchi_force"),
    "sakaguchi_jacobian": ("oscillatools.accel", "sakaguchi_jacobian"),
    "last_sakaguchi_force_tier_used": ("oscillatools.accel", "last_sakaguchi_force_tier_used"),
    "last_sakaguchi_jacobian_tier_used": (
        "oscillatools.accel",
        "last_sakaguchi_jacobian_tier_used",
    ),
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


__all__ = list(getattr(_target, "__all__", []))
