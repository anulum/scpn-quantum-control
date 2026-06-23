# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-engine - PyO3 extension typing contract

"""Typing contract for the optional Rust acceleration extension."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

_BoolArray: TypeAlias = NDArray[np.bool_]
_F64Array: TypeAlias = NDArray[np.float64]
_I32Array: TypeAlias = NDArray[np.int32]
_I64Array: TypeAlias = NDArray[np.int64]
_U64Array: TypeAlias = NDArray[np.uint64]

def analog_coupling_terms(
    k_flat: _F64Array,
    n: int,
    platform_code: int,
    coupling_scale: float,
    c6_coefficient: float,
    zero_threshold: float,
) -> tuple[_I64Array, _I64Array, _F64Array, _F64Array, _F64Array]: ...
def hybrid_coupling_partition(
    k_flat: _F64Array,
    n: int,
    analog_budget: int,
    analog_threshold: float,
    zero_threshold: float,
) -> tuple[_F64Array, _F64Array, _I64Array, _I64Array, _I64Array]: ...
def all_xy_expectations(
    psi_re: _F64Array,
    psi_im: _F64Array,
    n_osc: int,
) -> tuple[_F64Array, _F64Array]: ...
def biological_decode_z_errors(
    edge_u: _I64Array,
    edge_v: _I64Array,
    edge_weight: _F64Array,
    n_nodes: int,
    syndrome_x: NDArray[np.int8],
) -> NDArray[np.int8]: ...
def brute_mpc(
    b_flat: _F64Array,
    target: _F64Array,
    dim: int,
    horizon: int,
) -> tuple[_I64Array, float, _F64Array, int]: ...
def build_knm(n: int, k_base: float, alpha: float) -> _F64Array: ...
def cosim_classical_substep(
    theta: _F64Array,
    omega: _F64Array,
    k_classical: _F64Array,
    drive_a: _F64Array,
    drive_b: _F64Array,
    dt: float,
) -> _F64Array: ...
def frc_mrti_growth(
    field: _F64Array,
    dt_s: float,
    wavenumber: float,
    atwood: float,
    areal_mass: float,
    density: float,
    max_growth: float,
) -> float: ...
def ml_dsa_ntt(poly: list[int]) -> list[int]: ...
def ml_dsa_intt(poly: list[int]) -> list[int]: ...
def nist_berlekamp_massey(bits: NDArray[np.int8]) -> int: ...
def nist_monobit_sum(bits: NDArray[np.int8]) -> int: ...
def nist_runs_counts(bits: NDArray[np.int8]) -> tuple[int, int]: ...
def nist_block_longest_runs(bits: NDArray[np.int8], block_size: int) -> list[int]: ...
def nv_odmr_spectrum(
    freqs: _F64Array,
    centers: _F64Array,
    fwhm: float,
    contrast: float,
) -> _F64Array: ...
def quantise_q_format(values: list[float], frac_bits: int, total_bits: int) -> list[int]: ...
def sub_us_jitter_percentiles(jitters: _F64Array) -> tuple[float, float, float, float]: ...
def sub_us_tracker_summary(
    start_ns: _I64Array,
    end_ns: _I64Array,
    deadline_ns: _I64Array,
    target_period_ns: float,
) -> tuple[float, float, float, float, int, int]: ...
def vector_dot_value(values: _F64Array) -> float: ...
def vector_dot_jvp(values: _F64Array, tangent: _F64Array) -> float: ...
def vector_dot_vjp(values: _F64Array, cotangent: float) -> _F64Array: ...
def vector_dot_gradient(values: _F64Array) -> _F64Array: ...
def vector_squared_norm_value(values: _F64Array) -> float: ...
def vector_squared_norm_jvp(values: _F64Array, tangent: _F64Array) -> float: ...
def vector_squared_norm_vjp(values: _F64Array, cotangent: float) -> _F64Array: ...
def vector_squared_norm_gradient(values: _F64Array) -> _F64Array: ...
def matrix_vector_product_value(values: _F64Array, dimension: int) -> _F64Array: ...
def matrix_vector_product_jvp(
    values: _F64Array,
    tangent: _F64Array,
    dimension: int,
) -> _F64Array: ...
def matrix_vector_product_vjp(
    values: _F64Array,
    cotangent: _F64Array,
    dimension: int,
) -> _F64Array: ...
def matrix_vector_product_sum_gradient(values: _F64Array, dimension: int) -> _F64Array: ...
def matrix_matrix_product_value(values: _F64Array, dimension: int) -> _F64Array: ...
def matrix_matrix_product_jvp(
    values: _F64Array,
    tangent: _F64Array,
    dimension: int,
) -> _F64Array: ...
def matrix_matrix_product_vjp(
    values: _F64Array,
    cotangent: _F64Array,
    dimension: int,
) -> _F64Array: ...
def matrix_matrix_product_sum_gradient(values: _F64Array, dimension: int) -> _F64Array: ...
def matrix_trace_value(values: _F64Array, dimension: int) -> float: ...
def matrix_trace_jvp(values: _F64Array, tangent: _F64Array, dimension: int) -> float: ...
def matrix_trace_vjp(values: _F64Array, cotangent: float, dimension: int) -> _F64Array: ...
def matrix_trace_gradient(values: _F64Array, dimension: int) -> _F64Array: ...
def matrix_frobenius_norm_squared_value(values: _F64Array) -> float: ...
def matrix_frobenius_norm_squared_jvp(values: _F64Array, tangent: _F64Array) -> float: ...
def matrix_frobenius_norm_squared_vjp(values: _F64Array, cotangent: float) -> _F64Array: ...
def matrix_frobenius_norm_squared_gradient(values: _F64Array) -> _F64Array: ...
def matrix_2x2_determinant_value(values: _F64Array) -> float: ...
def matrix_2x2_determinant_jvp(values: _F64Array, tangent: _F64Array) -> float: ...
def matrix_2x2_determinant_vjp(values: _F64Array, cotangent: float) -> _F64Array: ...
def matrix_2x2_determinant_gradient(values: _F64Array) -> _F64Array: ...
def matrix_2x2_inverse_value(values: _F64Array) -> _F64Array: ...
def matrix_2x2_inverse_jvp(values: _F64Array, tangent: _F64Array) -> _F64Array: ...
def matrix_2x2_inverse_vjp(values: _F64Array, cotangent: _F64Array) -> _F64Array: ...
def matrix_2x2_inverse_sum_gradient(values: _F64Array) -> _F64Array: ...
def matrix_2x2_solve_value(values: _F64Array) -> _F64Array: ...
def matrix_2x2_solve_jvp(values: _F64Array, tangent: _F64Array) -> _F64Array: ...
def matrix_2x2_solve_vjp(values: _F64Array, cotangent: _F64Array) -> _F64Array: ...
def matrix_2x2_solve_sum_gradient(values: _F64Array) -> _F64Array: ...
def matrix_2x2_eigenvalues_value(values: _F64Array) -> _F64Array: ...
def matrix_2x2_eigenvalues_jvp(values: _F64Array, tangent: _F64Array) -> _F64Array: ...
def matrix_2x2_eigenvalues_vjp(values: _F64Array, cotangent: _F64Array) -> _F64Array: ...
def matrix_2x2_eigenvalues_sum_gradient(values: _F64Array) -> _F64Array: ...
def matrix_2x2_eigensystem_value(values: _F64Array) -> _F64Array: ...
def matrix_2x2_eigensystem_jvp(values: _F64Array, tangent: _F64Array) -> _F64Array: ...
def matrix_2x2_eigensystem_vjp(values: _F64Array, cotangent: _F64Array) -> _F64Array: ...
def matrix_2x2_eigensystem_sum_gradient(values: _F64Array) -> _F64Array: ...
def symmetric_2x2_eigenvalues_value(values: _F64Array) -> _F64Array: ...
def symmetric_2x2_eigenvalues_jvp(values: _F64Array, tangent: _F64Array) -> _F64Array: ...
def symmetric_2x2_eigenvalues_vjp(values: _F64Array, cotangent: _F64Array) -> _F64Array: ...
def symmetric_2x2_eigenvalues_sum_gradient(values: _F64Array) -> _F64Array: ...
def symmetric_2x2_cholesky_value(values: _F64Array) -> _F64Array: ...
def symmetric_2x2_cholesky_jvp(values: _F64Array, tangent: _F64Array) -> _F64Array: ...
def symmetric_2x2_cholesky_vjp(values: _F64Array, cotangent: _F64Array) -> _F64Array: ...
def symmetric_2x2_cholesky_sum_gradient(values: _F64Array) -> _F64Array: ...
def matrix_quadratic_form_value(values: _F64Array, dimension: int) -> float: ...
def matrix_quadratic_form_jvp(
    values: _F64Array,
    tangent: _F64Array,
    dimension: int,
) -> float: ...
def matrix_quadratic_form_vjp(
    values: _F64Array,
    cotangent: float,
    dimension: int,
) -> _F64Array: ...
def matrix_quadratic_form_gradient(values: _F64Array, dimension: int) -> _F64Array: ...
def phase_qnode_fubini_study_metric_rust(
    state_re: _F64Array,
    state_im: _F64Array,
    derivatives_re: _F64Array,
    derivatives_im: _F64Array,
) -> tuple[_F64Array, _F64Array, _F64Array]: ...
def phase_qnode_computational_basis_fisher_rust(
    state_re: _F64Array,
    state_im: _F64Array,
    derivatives_re: _F64Array,
    derivatives_im: _F64Array,
    min_probability: float,
) -> tuple[_F64Array, _F64Array, _F64Array]: ...
def phase_qnode_vector_jvp_rust(jacobian: _F64Array, tangent: _F64Array) -> _F64Array: ...
def phase_qnode_vector_vjp_rust(jacobian: _F64Array, cotangent: _F64Array) -> _F64Array: ...
def phase_qnode_hessian_vector_product_rust(
    hessian: _F64Array, vector: _F64Array
) -> _F64Array: ...
def phase_qnode_vector_hessian_tensor_rust(
    hessian_tensor: _F64Array, symmetry_tolerance: float = ...
) -> _F64Array: ...
def phase_qnode_complex_derivative_contract_rust() -> dict[str, object]: ...
def program_ad_effect_ir_metadata_summary(serialization: str) -> str: ...
def program_ad_effect_ir_interpret_forward(serialization: str, inputs: list[float]) -> str: ...
def program_ad_effect_ir_interpret_value_and_gradient(
    serialization: str, inputs: list[float]
) -> str: ...
def parameter_shift_gradient_uncertainty_rust(
    plus_values: _F64Array,
    minus_values: _F64Array,
    plus_variances: _F64Array,
    minus_variances: _F64Array,
    plus_shots: _F64Array,
    minus_shots: _F64Array,
    coefficients: _F64Array,
    trainable: _BoolArray,
    confidence_z: float = ...,
) -> tuple[_F64Array, _F64Array, _F64Array, _F64Array]: ...
def spsa_gradient_rust(
    plus_values: _F64Array,
    minus_values: _F64Array,
    perturbations: _F64Array,
    plus_variances: _F64Array,
    minus_variances: _F64Array,
    plus_shots: _F64Array,
    minus_shots: _F64Array,
    trainable: _BoolArray,
    perturbation_radius: float,
    confidence_z: float = ...,
) -> tuple[_F64Array, _F64Array, _F64Array, _F64Array]: ...
def score_function_gradient_rust(
    rewards: _F64Array,
    score_vectors: _F64Array,
    trainable: _BoolArray,
    baseline: float = ...,
    confidence_z: float = ...,
) -> tuple[_F64Array, _F64Array, _F64Array, _F64Array]: ...
def gradient_confidence_interval_rust(
    gradient: _F64Array,
    standard_error: _F64Array,
    trainable: _BoolArray,
    confidence_z: float = ...,
    max_standard_error: float | None = ...,
    max_confidence_radius: float | None = ...,
) -> tuple[_F64Array, _F64Array, str, list[str]]: ...
def build_sparse_xy_hamiltonian(
    k_flat: _F64Array,
    omega: _F64Array,
    n: int,
) -> tuple[_I64Array, _I64Array, _F64Array]: ...
def build_xy_hamiltonian_dense(k_flat: _F64Array, omega: _F64Array, n: int) -> _F64Array: ...
def concatenated_logical_rate_rust(
    p_physical: float,
    distances: _I64Array,
    p_threshold: float,
    prefactor: float,
) -> _F64Array: ...
def correlation_matrix_xy(psi_re: _F64Array, psi_im: _F64Array, n_osc: int) -> _F64Array: ...
def dla_dimension(
    generators_flat: _F64Array,
    dim: int,
    n_generators: int,
    max_iterations: int,
    max_dimension: int,
    tol: float,
) -> int: ...
def dla_protected_memory_mask(
    n_logical: int,
    code_distance: int,
    target_parity: int,
) -> _BoolArray: ...
def dla_protected_memory_metrics(
    probabilities: _F64Array,
    n_logical: int,
    code_distance: int,
    target_parity: int,
) -> tuple[float, float, float, float, float]: ...
def dla_protected_trajectory_metrics(
    probabilities: _F64Array,
    n_logical: int,
    code_distance: int,
    target_parity: int,
) -> tuple[_F64Array, _F64Array, _F64Array, _F64Array, _F64Array]: ...
def expectation_pauli_fast(
    psi_re: _F64Array,
    psi_im: _F64Array,
    n: int,
    qubit: int,
    pauli: int,
) -> float: ...
def feedback_policy_batch(
    r_values: _F64Array,
    target_r: float,
    deadband: float,
    base_gain: float,
    max_gain: float,
) -> tuple[_I32Array, _F64Array, _F64Array]: ...
def run_realtime_feedback_loop(
    theta0: _F64Array,
    omega: _F64Array,
    k: _F64Array,
    target_r: float,
    deadband: float,
    base_gain: float,
    max_gain: float,
    dt: float,
    n_steps: int,
) -> tuple[_F64Array, _F64Array, _F64Array, _F64Array, _I32Array, _F64Array, _F64Array]: ...
def fit_symmetry_decay(
    s_ideal: float,
    noisy_values: _F64Array,
    noise_scales: _F64Array,
) -> tuple[float, float]: ...
def free_energy_gradient_rust(
    mu: _F64Array,
    x_observed: _F64Array,
    k_precision: _F64Array,
    sensory_precision: _F64Array,
    ridge: float,
) -> _F64Array: ...
def gauge_covariant_kinetic_rust(
    phi_re: _F64Array,
    phi_im: _F64Array,
    links: _F64Array,
    edges: _I64Array,
    g_coupling: float,
) -> float: ...
def gauge_force_batch(
    links: _F64Array,
    triangles: _I64Array,
    triangle_signs: _F64Array,
    n_triangles: int,
    n_edges: int,
    beta: float,
) -> _F64Array: ...
def guess_extrapolate_batch(
    target_noisy: _F64Array,
    symmetry_noisy: _F64Array,
    s_ideal: float,
    alpha: float,
) -> _F64Array: ...
def hierarchical_prediction_error_rust(
    observations: _F64Array,
    beliefs: _F64Array,
    k: _F64Array,
) -> _F64Array: ...
def hypergeometric_envelope_batch(
    times: _F64Array,
    alpha: float,
    beta: float,
    gamma_width: float,
) -> _F64Array: ...
def ici_mixing_angle_batch(times: _F64Array, t_total: float, theta_jump: float) -> _F64Array: ...
def ici_three_level_evolution_batch(
    times: _F64Array,
    omega_p: _F64Array,
    omega_s: _F64Array,
    gamma: float,
) -> _F64Array: ...
def knm_domain_coupling(
    k: _F64Array,
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> float: ...
def kuramoto_euler(
    theta0: _F64Array,
    omega: _F64Array,
    k: _F64Array,
    dt: float,
    n_steps: int,
) -> _F64Array: ...
def kuramoto_trajectory(
    theta0: _F64Array,
    omega: _F64Array,
    k: _F64Array,
    dt: float,
    n_steps: int,
) -> tuple[_F64Array, _F64Array]: ...
def higher_order_kuramoto_trajectory(
    theta0: _F64Array,
    omega: _F64Array,
    k: _F64Array,
    hyperedges: _I64Array,
    hyper_weights: _F64Array,
    dt: float,
    n_steps: int,
) -> tuple[_F64Array, _F64Array]: ...
def monitored_kuramoto_trajectory(
    theta0: _F64Array,
    omega: _F64Array,
    k: _F64Array,
    target_r: float,
    monitor_gain: float,
    measurement_strength: float,
    dt: float,
    n_steps: int,
) -> tuple[_F64Array, _F64Array, _F64Array, _F64Array]: ...
def pt_symmetric_kuramoto_trajectory(
    theta0: _F64Array,
    omega: _F64Array,
    k: _F64Array,
    gain_loss: _F64Array,
    dt: float,
    n_steps: int,
) -> tuple[_F64Array, _F64Array, _F64Array, _F64Array]: ...
def kuramoto_witness_candidate_features(
    theta0: _F64Array,
    omega: _F64Array,
    k: _F64Array,
    candidates: _F64Array,
    dt: float,
    n_steps: int,
) -> tuple[_F64Array, _F64Array, _F64Array]: ...
def koopman_generator(k: _F64Array, omega: _F64Array, theta_ref: _F64Array) -> _F64Array: ...
def lanczos_b_coefficients(
    h_re: _F64Array,
    h_im: _F64Array,
    o_re: _F64Array,
    o_im: _F64Array,
    dim: int,
    max_steps: int,
    tol: float,
) -> list[float]: ...
def lindblad_anti_hermitian_diag(k_flat: _F64Array, n: int, threshold: float) -> _F64Array: ...
def lindblad_jump_ops_coo(
    k_flat: _F64Array,
    n: int,
    threshold: float,
) -> tuple[_I64Array, _I64Array, _I64Array, int]: ...
def magnetisation_labels(n: int) -> _I32Array: ...
def mc_xy_simulate(
    k_flat: _F64Array,
    n: int,
    temperature: float,
    n_thermalize: int,
    n_measure: int,
    seed: int,
) -> tuple[float, float, float]: ...
def order_param_from_statevector(psi_re: _F64Array, psi_im: _F64Array, n: int) -> float: ...
def order_parameter(theta: _F64Array) -> float: ...
def order_parameter_gradient(theta: _F64Array) -> _F64Array: ...
def order_parameter_hessian(theta: _F64Array) -> _F64Array: ...
def mean_phase(theta: _F64Array) -> float: ...
def mean_phase_gradient(theta: _F64Array) -> _F64Array: ...
def mean_phase_hessian(theta: _F64Array) -> _F64Array: ...
def daido_order_parameter(theta: _F64Array, m: int) -> float: ...
def daido_order_parameter_gradient(theta: _F64Array, m: int) -> _F64Array: ...
def daido_order_parameter_hessian(theta: _F64Array, m: int) -> _F64Array: ...
def otoc_from_eigendecomp(
    eigenvalues: _F64Array,
    eigvecs_re: _F64Array,
    eigvecs_im: _F64Array,
    w_re: _F64Array,
    w_im: _F64Array,
    v_re: _F64Array,
    v_im: _F64Array,
    psi_re: _F64Array,
    psi_im: _F64Array,
    times: _F64Array,
    dim: int,
) -> _F64Array: ...
def parity_filter_mask(bitstrings: _U64Array, expected_parity: int) -> _BoolArray: ...
def pec_coefficients(gate_error_rate: float) -> list[float]: ...
def pec_sample_parallel(
    gate_error_rate: float,
    n_gates: int,
    n_samples: int,
    base_exp_z: float,
    seed: int,
) -> tuple[float, float, list[float]]: ...
def qpetri_sample_marking(
    probabilities: _F64Array,
    shots: int,
    seed: int,
) -> _F64Array: ...
def qpetri_campaign_aggregate(
    output_markings_flat: _F64Array,
    transition_activity_flat: _F64Array,
    entropies: _F64Array,
    purities: _F64Array,
    n_steps: int,
    n_places: int,
    n_transitions: int,
) -> tuple[_F64Array, _F64Array, float, float]: ...
def qpetri_state_metrics(probabilities: _F64Array) -> tuple[float, float]: ...
def qpetri_transition_activity(
    w_in_flat: _F64Array,
    marking: _F64Array,
    thresholds: _F64Array,
    n_transitions: int,
    n_places: int,
    sparsity_eps: float,
) -> _F64Array: ...
def plaquette_action_batch(
    links: _F64Array,
    triangles: _I64Array,
    triangle_signs: _F64Array,
    n_triangles: int,
    beta: float,
) -> tuple[float, float]: ...
def rabi_pi_amplitude_fit(
    amplitudes: _F64Array,
    excited_population: _F64Array,
) -> tuple[float, float, float]: ...
def score_regions_batch(
    gate_errors_flat: _F64Array,
    n_qubits: int,
    region_offsets: _I64Array,
    region_qubits: _I64Array,
) -> tuple[_F64Array, _F64Array, _F64Array]: ...
def state_order_param_sparse(psi_re: _F64Array, psi_im: _F64Array, n_osc: int) -> float: ...
def topological_charge_rust(
    links: _F64Array,
    triangles: _I64Array,
    triangle_signs: _F64Array,
    n_triangles: int,
) -> float: ...
def variational_free_energy_rust(
    mu: _F64Array,
    x_observed: _F64Array,
    k_precision: _F64Array,
    sensory_precision: _F64Array,
    sigma_diag: float,
    ridge: float,
) -> tuple[float, float, float]: ...
