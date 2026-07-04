// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Rust Acceleration Engine

//! Rust acceleration for scpn-quantum-control.
//!
//! Hot paths moved from Python to Rust via PyO3:
//! - PEC Monte Carlo sampling (parallel via rayon)
//! - Classical Kuramoto ODE (vectorised)
//! - K_nm matrix construction
//! - DLA commutator closure (parallel via rayon)
//! - Monte Carlo XY model simulation
//! - Sparse Pauli expectation values
//! - XY Hamiltonian construction (dense + sparse)
//! - Krylov complexity (operator Lanczos)
//! - OTOC via eigendecomposition (parallel via rayon)
//! - Lindblad jump operators (COO sparse)
//! - Symmetry sectors, correlation, parity filtering
//! - Brute-force MPC
//! - Real-time feedback policy updates
//! - Analog Kuramoto coupling compilation
//! - DLA-protected logical memory diagnostics
//! - Kuramoto witness-discovery candidate features

use pyo3::prelude::*;

pub mod analog;
pub mod biological_qec;
pub mod community;
pub mod compiler_ad;
pub mod complex_utils;
pub mod concat_qec;
pub mod cosimulation;
pub mod dla;
pub mod entropy;
pub mod feedback;
pub mod fep;
pub mod frc;
pub mod gauge_lattice;
pub mod hamiltonian;
pub mod hls_quantise;
pub mod knm;
pub mod koopman;
pub mod krylov;
pub mod kuramoto;
pub mod kuramoto_autodiff;
pub mod kuramoto_common;
pub mod kuramoto_coupling;
pub mod kuramoto_observables;
pub mod lindblad;
pub mod ml_dsa;
pub mod monte_carlo;
pub mod mpc;
pub mod otoc;
pub mod pauli;
pub mod pec;
pub mod program_ad_ir;
mod program_ad_linalg_array;
mod program_ad_linalg_spectral;
mod program_ad_linalg_svd;
mod program_ad_order_statistic_reduction;
mod program_ad_product_reduction;
pub mod program_ad_registry_mirror;
mod program_ad_static_source_map;
mod program_ad_trapezoid_reduction;
mod program_ad_variance_reduction;
pub mod pulse_shaping;
pub mod qnode_metrics;
pub mod qpetri;
pub mod realtime;
pub mod sectors;
pub mod sensing;
pub mod stochastic_gradient;
pub mod symmetry_decay;
pub mod validation;

#[pymodule]
fn scpn_quantum_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // PEC
    m.add_function(wrap_pyfunction!(pec::pec_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(pec::pec_sample_parallel, m)?)?;

    // K_nm
    m.add_function(wrap_pyfunction!(knm::build_knm, m)?)?;

    // Kuramoto
    m.add_function(wrap_pyfunction!(kuramoto::kuramoto_euler, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto_observables::order_parameter, m)?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_observables::order_parameter_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_observables::order_parameter_hessian,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(kuramoto_observables::mean_phase, m)?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_observables::mean_phase_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_observables::mean_phase_hessian,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_observables::daido_order_parameter,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_observables::daido_order_parameter_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_observables::daido_order_parameter_hessian,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(kuramoto_observables::daido_mode_phase, m)?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_observables::daido_mode_phase_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_observables::daido_mode_phase_hessian,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(kuramoto_coupling::mean_field_force, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto_coupling::mean_field_jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::daido_mean_field_force,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::daido_mean_field_jacobian,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::networked_kuramoto_force,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::networked_kuramoto_jacobian,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::kuramoto_interaction_energy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::kuramoto_interaction_energy_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::kuramoto_interaction_energy_hessian,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(kuramoto_coupling::sakaguchi_force, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto_coupling::sakaguchi_jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::sakaguchi_mean_field_force,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::sakaguchi_mean_field_jacobian,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::triadic_mean_field_force,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::triadic_mean_field_jacobian,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::local_order_parameter,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::local_order_parameter_jacobian,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(kuramoto_coupling::local_mean_phase, m)?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_coupling::local_mean_phase_jacobian,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(kuramoto::kuramoto_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_autodiff::kuramoto_euler_trajectory,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(kuramoto_autodiff::kuramoto_euler_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto_autodiff::kuramoto_rk4_trajectory,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(kuramoto_autodiff::kuramoto_rk4_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto::higher_order_kuramoto_trajectory,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto::monitored_kuramoto_trajectory,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto::pt_symmetric_kuramoto_trajectory,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        kuramoto::kuramoto_witness_candidate_features,
        m
    )?)?;

    // Koopman
    m.add_function(wrap_pyfunction!(koopman::koopman_generator, m)?)?;

    // Concatenated QEC
    m.add_function(wrap_pyfunction!(
        concat_qec::concatenated_logical_rate_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(concat_qec::knm_domain_coupling, m)?)?;
    m.add_function(wrap_pyfunction!(
        biological_qec::biological_decode_z_errors,
        m
    )?)?;

    // FEP
    m.add_function(wrap_pyfunction!(fep::free_energy_gradient_rust, m)?)?;
    m.add_function(wrap_pyfunction!(
        fep::hierarchical_prediction_error_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(fep::variational_free_energy_rust, m)?)?;

    // Gauge lattice
    m.add_function(wrap_pyfunction!(gauge_lattice::plaquette_action_batch, m)?)?;
    m.add_function(wrap_pyfunction!(gauge_lattice::gauge_force_batch, m)?)?;
    m.add_function(wrap_pyfunction!(
        gauge_lattice::gauge_covariant_kinetic_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(gauge_lattice::topological_charge_rust, m)?)?;

    // DLA
    m.add_function(wrap_pyfunction!(dla::dla_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(dla::dla_protected_memory_mask, m)?)?;
    m.add_function(wrap_pyfunction!(dla::dla_protected_memory_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(dla::dla_protected_trajectory_metrics, m)?)?;

    // Monte Carlo XY
    m.add_function(wrap_pyfunction!(monte_carlo::mc_xy_simulate, m)?)?;

    // Pauli expectations
    m.add_function(wrap_pyfunction!(pauli::state_order_param_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(pauli::expectation_pauli_fast, m)?)?;
    m.add_function(wrap_pyfunction!(pauli::all_xy_expectations, m)?)?;

    // MPC
    m.add_function(wrap_pyfunction!(mpc::brute_mpc, m)?)?;

    // Feedback control
    m.add_function(wrap_pyfunction!(feedback::feedback_policy_batch, m)?)?;
    m.add_function(wrap_pyfunction!(feedback::run_realtime_feedback_loop, m)?)?;

    // Analog backend compilation
    m.add_function(wrap_pyfunction!(analog::analog_coupling_terms, m)?)?;
    m.add_function(wrap_pyfunction!(analog::hybrid_coupling_partition, m)?)?;

    // Krylov
    m.add_function(wrap_pyfunction!(krylov::lanczos_b_coefficients, m)?)?;

    // OTOC
    m.add_function(wrap_pyfunction!(otoc::otoc_from_eigendecomp, m)?)?;

    // Hamiltonian
    m.add_function(wrap_pyfunction!(
        hamiltonian::build_xy_hamiltonian_dense,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        hamiltonian::build_sparse_xy_hamiltonian,
        m
    )?)?;

    // Lindblad
    m.add_function(wrap_pyfunction!(lindblad::lindblad_jump_ops_coo, m)?)?;
    m.add_function(wrap_pyfunction!(lindblad::lindblad_anti_hermitian_diag, m)?)?;

    // Sectors
    m.add_function(wrap_pyfunction!(sectors::magnetisation_labels, m)?)?;
    m.add_function(wrap_pyfunction!(sectors::order_param_from_statevector, m)?)?;
    m.add_function(wrap_pyfunction!(sectors::correlation_matrix_xy, m)?)?;
    m.add_function(wrap_pyfunction!(sectors::parity_filter_mask, m)?)?;

    // Symmetry decay (GUESS)
    m.add_function(wrap_pyfunction!(
        symmetry_decay::guess_extrapolate_batch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(symmetry_decay::fit_symmetry_decay, m)?)?;

    // Community scoring (DynQ)
    m.add_function(wrap_pyfunction!(community::score_regions_batch, m)?)?;

    // Phase-QNode differentiable metric and transform parity
    m.add_function(wrap_pyfunction!(
        qnode_metrics::phase_qnode_fubini_study_metric_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        qnode_metrics::phase_qnode_computational_basis_fisher_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        qnode_metrics::phase_qnode_vector_jvp_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        qnode_metrics::phase_qnode_vector_vjp_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        qnode_metrics::phase_qnode_hessian_vector_product_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        qnode_metrics::phase_qnode_vector_hessian_tensor_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        qnode_metrics::phase_qnode_complex_derivative_contract_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stochastic_gradient::parameter_shift_gradient_uncertainty_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stochastic_gradient::spsa_gradient_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stochastic_gradient::score_function_gradient_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stochastic_gradient::gradient_confidence_interval_rust,
        m
    )?)?;

    // Compiler-backed AD native parity
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_eigenvalues_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_eigenvalues_jvp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_eigenvalues_vjp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_eigenvalues_sum_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_eigensystem_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_eigensystem_jvp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_eigensystem_vjp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_eigensystem_sum_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_determinant_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_determinant_jvp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_determinant_vjp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_determinant_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_2x2_inverse_value, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_2x2_inverse_jvp, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_2x2_inverse_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_inverse_sum_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_2x2_solve_value, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_2x2_solve_jvp, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_2x2_solve_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_2x2_solve_sum_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::symmetric_2x2_cholesky_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::symmetric_2x2_cholesky_jvp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::symmetric_2x2_cholesky_vjp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::symmetric_2x2_cholesky_sum_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::symmetric_2x2_eigenvalues_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::symmetric_2x2_eigenvalues_jvp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::symmetric_2x2_eigenvalues_vjp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::symmetric_2x2_eigenvalues_sum_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_quadratic_form_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_quadratic_form_jvp, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_quadratic_form_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_quadratic_form_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_trace_value, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_trace_jvp, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_trace_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_trace_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_frobenius_norm_squared_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_frobenius_norm_squared_jvp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_frobenius_norm_squared_vjp,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_frobenius_norm_squared_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_vector_product_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_vector_product_jvp, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_vector_product_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_vector_product_sum_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_matrix_product_value,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_matrix_product_jvp, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::matrix_matrix_product_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::matrix_matrix_product_sum_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::vector_dot_value, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::vector_dot_jvp, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::vector_dot_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::vector_dot_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::vector_squared_norm_value, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::vector_squared_norm_jvp, m)?)?;
    m.add_function(wrap_pyfunction!(compiler_ad::vector_squared_norm_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(
        compiler_ad::vector_squared_norm_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        program_ad_ir::program_ad_effect_ir_metadata_summary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        program_ad_ir::program_ad_effect_ir_interpret_forward,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        program_ad_ir::program_ad_effect_ir_interpret_value_and_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        program_ad_registry_mirror::program_ad_registry_metadata_mirror,
        m
    )?)?;

    // Pulse shaping (hypergeometric + ICI)
    m.add_function(wrap_pyfunction!(
        pulse_shaping::hypergeometric_envelope_batch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pulse_shaping::ici_mixing_angle_batch, m)?)?;
    m.add_function(wrap_pyfunction!(
        pulse_shaping::ici_three_level_evolution_batch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pulse_shaping::rabi_pi_amplitude_fit, m)?)?;

    // Quantum Petri superposition diagnostics
    m.add_function(wrap_pyfunction!(qpetri::qpetri_transition_activity, m)?)?;
    m.add_function(wrap_pyfunction!(qpetri::qpetri_state_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(qpetri::qpetri_sample_marking, m)?)?;
    m.add_function(wrap_pyfunction!(qpetri::qpetri_campaign_aggregate, m)?)?;

    // Sub-microsecond realtime telemetry
    m.add_function(wrap_pyfunction!(realtime::sub_us_jitter_percentiles, m)?)?;
    m.add_function(wrap_pyfunction!(realtime::sub_us_tracker_summary, m)?)?;

    // NIST SP 800-22 hot-path statistics
    m.add_function(wrap_pyfunction!(entropy::nist_berlekamp_massey, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::nist_monobit_sum, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::nist_runs_counts, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::nist_block_longest_runs, m)?)?;

    // FRC pulsed-shot physics
    m.add_function(wrap_pyfunction!(frc::frc_mrti_growth, m)?)?;

    // NV-centre ODMR spectrum
    m.add_function(wrap_pyfunction!(sensing::nv_odmr_spectrum, m)?)?;

    // ML-DSA number-theoretic transform
    m.add_function(wrap_pyfunction!(ml_dsa::ml_dsa_ntt, m)?)?;
    m.add_function(wrap_pyfunction!(ml_dsa::ml_dsa_intt, m)?)?;

    // UltraScale+ HLS Q-format quantisation
    m.add_function(wrap_pyfunction!(hls_quantise::quantise_q_format, m)?)?;

    // Quantum/classical co-simulation classical substep
    m.add_function(wrap_pyfunction!(cosimulation::cosim_classical_substep, m)?)?;

    Ok(())
}
