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

use pyo3::prelude::*;

pub mod community;
pub mod complex_utils;
pub mod validation;
pub mod concat_qec;
pub mod symmetry_decay;
pub mod dla;
pub mod fep;
pub mod gauge_lattice;
pub mod hamiltonian;
pub mod knm;
pub mod krylov;
pub mod kuramoto;
pub mod lindblad;
pub mod monte_carlo;
pub mod mpc;
pub mod otoc;
pub mod pauli;
pub mod pec;
pub mod pulse_shaping;
pub mod sectors;

#[pymodule]
fn scpn_quantum_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // PEC
    m.add_function(wrap_pyfunction!(pec::pec_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(pec::pec_sample_parallel, m)?)?;

    // K_nm
    m.add_function(wrap_pyfunction!(knm::build_knm, m)?)?;

    // Kuramoto
    m.add_function(wrap_pyfunction!(kuramoto::kuramoto_euler, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto::order_parameter, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto::kuramoto_trajectory, m)?)?;

    // Concatenated QEC
    m.add_function(wrap_pyfunction!(concat_qec::concatenated_logical_rate_rust, m)?)?;
    m.add_function(wrap_pyfunction!(concat_qec::knm_domain_coupling, m)?)?;

    // FEP
    m.add_function(wrap_pyfunction!(fep::free_energy_gradient_rust, m)?)?;
    m.add_function(wrap_pyfunction!(fep::hierarchical_prediction_error_rust, m)?)?;
    m.add_function(wrap_pyfunction!(fep::variational_free_energy_rust, m)?)?;

    // Gauge lattice
    m.add_function(wrap_pyfunction!(gauge_lattice::plaquette_action_batch, m)?)?;
    m.add_function(wrap_pyfunction!(gauge_lattice::gauge_force_batch, m)?)?;
    m.add_function(wrap_pyfunction!(gauge_lattice::gauge_covariant_kinetic_rust, m)?)?;
    m.add_function(wrap_pyfunction!(gauge_lattice::topological_charge_rust, m)?)?;

    // DLA
    m.add_function(wrap_pyfunction!(dla::dla_dimension, m)?)?;

    // Monte Carlo XY
    m.add_function(wrap_pyfunction!(monte_carlo::mc_xy_simulate, m)?)?;

    // Pauli expectations
    m.add_function(wrap_pyfunction!(pauli::state_order_param_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(pauli::expectation_pauli_fast, m)?)?;
    m.add_function(wrap_pyfunction!(pauli::all_xy_expectations, m)?)?;

    // MPC
    m.add_function(wrap_pyfunction!(mpc::brute_mpc, m)?)?;

    // Krylov
    m.add_function(wrap_pyfunction!(krylov::lanczos_b_coefficients, m)?)?;

    // OTOC
    m.add_function(wrap_pyfunction!(otoc::otoc_from_eigendecomp, m)?)?;

    // Hamiltonian
    m.add_function(wrap_pyfunction!(hamiltonian::build_xy_hamiltonian_dense, m)?)?;
    m.add_function(wrap_pyfunction!(hamiltonian::build_sparse_xy_hamiltonian, m)?)?;

    // Lindblad
    m.add_function(wrap_pyfunction!(lindblad::lindblad_jump_ops_coo, m)?)?;
    m.add_function(wrap_pyfunction!(lindblad::lindblad_anti_hermitian_diag, m)?)?;

    // Sectors
    m.add_function(wrap_pyfunction!(sectors::magnetisation_labels, m)?)?;
    m.add_function(wrap_pyfunction!(sectors::order_param_from_statevector, m)?)?;
    m.add_function(wrap_pyfunction!(sectors::correlation_matrix_xy, m)?)?;
    m.add_function(wrap_pyfunction!(sectors::parity_filter_mask, m)?)?;

    // Symmetry decay (GUESS)
    m.add_function(wrap_pyfunction!(symmetry_decay::guess_extrapolate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(symmetry_decay::fit_symmetry_decay, m)?)?;

    // Community scoring (DynQ)
    m.add_function(wrap_pyfunction!(community::score_regions_batch, m)?)?;

    // Pulse shaping (hypergeometric + ICI)
    m.add_function(wrap_pyfunction!(pulse_shaping::hypergeometric_envelope_batch, m)?)?;
    m.add_function(wrap_pyfunction!(pulse_shaping::ici_mixing_angle_batch, m)?)?;
    m.add_function(wrap_pyfunction!(pulse_shaping::ici_three_level_evolution_batch, m)?)?;

    Ok(())
}
