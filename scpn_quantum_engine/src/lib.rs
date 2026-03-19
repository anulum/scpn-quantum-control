// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Rust acceleration for scpn-quantum-control.
//!
//! Hot paths moved from Python to Rust via PyO3:
//! - PEC Monte Carlo sampling (parallel via rayon)
//! - Classical Kuramoto ODE (vectorized)
//! - K_nm matrix construction

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

/// PEC quasi-probability coefficients for single-qubit depolarizing channel.
/// Returns [q_I, q_X, q_Y, q_Z].
#[pyfunction]
fn pec_coefficients(gate_error_rate: f64) -> [f64; 4] {
    let p = gate_error_rate;
    let denom = 4.0 - 4.0 * p;
    let q_i = 1.0 + 3.0 * p / denom;
    let q_xyz = -p / denom;
    [q_i, q_xyz, q_xyz, q_xyz]
}

/// PEC sign-sampling in parallel (rayon). Single-qubit depolarizing model.
///
/// Returns (mitigated_value, overhead, sign_distribution).
/// `base_exp_z` is the noiseless <Z> expectation from the ideal circuit.
/// Each sample draws a Pauli correction per gate, accumulates the sign
/// product, and scales by gamma^n_gates. The sampled operator identity
/// affects the sign but not the base expectation — this is the single-qubit
/// approximation where all corrections act on one qubit.
#[pyfunction]
fn pec_sample_parallel(
    gate_error_rate: f64,
    n_gates: usize,
    n_samples: usize,
    base_exp_z: f64,
    seed: u64,
) -> (f64, f64, Vec<f64>) {
    let coeffs = pec_coefficients(gate_error_rate);
    let abs_coeffs: Vec<f64> = coeffs.iter().map(|c| c.abs()).collect();
    let gamma_single: f64 = abs_coeffs.iter().sum();
    let probs: Vec<f64> = abs_coeffs.iter().map(|a| a / gamma_single).collect();
    let signs: Vec<f64> = coeffs.iter().map(|c| c.signum()).collect();
    let gamma_total = gamma_single.powi(n_gates as i32);

    // Cumulative probabilities for sampling
    let cum_probs: Vec<f64> = probs
        .iter()
        .scan(0.0, |acc, &p| {
            *acc += p;
            Some(*acc)
        })
        .collect();

    // Parallel Monte Carlo
    let results: Vec<(f64, f64)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            let mut total_sign = 1.0_f64;

            for _ in 0..n_gates {
                let r: f64 = rng.random();
                let idx = cum_probs.iter().position(|&c| r < c).unwrap_or(3);
                total_sign *= signs[idx];
            }

            let value = gamma_total * total_sign * base_exp_z;
            (value, total_sign)
        })
        .collect();

    let mut acc = 0.0;
    let mut sign_dist = Vec::with_capacity(n_samples);
    for (value, sign) in &results {
        acc += value;
        sign_dist.push(*sign);
    }

    (acc / n_samples as f64, gamma_total, sign_dist)
}

/// Build K_nm coupling matrix from Paper 27 parameters.
/// K_nm = K_base * exp(-alpha * |n - m|) with calibration anchors.
#[pyfunction]
fn build_knm<'py>(
    py: Python<'py>,
    n: usize,
    k_base: f64,
    alpha: f64,
) -> Bound<'py, PyArray2<f64>> {
    // Full exponential matrix including diagonal (K_base at i==j)
    let mut k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            k[[i, j]] = k_base * (-alpha * (i as f64 - j as f64).abs()).exp();
        }
    }

    // Calibration anchors (Paper 27 Table 2)
    let anchors: [(usize, usize, f64); 4] = [(0, 1, 0.302), (1, 2, 0.201), (2, 3, 0.252), (3, 4, 0.154)];
    for &(i, j, val) in &anchors {
        if i < n && j < n {
            k[[i, j]] = val;
            k[[j, i]] = val;
        }
    }

    // Cross-hierarchy boosts (max preserves exponential if already larger)
    if n > 15 {
        k[[0, 15]] = k[[0, 15]].max(0.05);
        k[[15, 0]] = k[[15, 0]].max(0.05);
    }
    if n > 6 {
        k[[4, 6]] = k[[4, 6]].max(0.15);
        k[[6, 4]] = k[[6, 4]].max(0.15);
    }

    PyArray2::from_owned_array(py, k)
}

/// Classical Kuramoto ODE step (vectorized, no Python overhead).
/// theta' = omega + K @ sin(theta_outer - theta_inner)
/// Returns new theta after n_steps of Euler integration.
#[pyfunction]
fn kuramoto_euler<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> Bound<'py, PyArray1<f64>> {
    let mut theta = theta0.as_array().to_owned();
    let omega = omega.as_array();
    let k = k.as_array();
    let n = theta.len();

    for _ in 0..n_steps {
        let mut dtheta = Array1::<f64>::zeros(n);
        for i in 0..n {
            dtheta[i] = omega[i];
            for j in 0..n {
                dtheta[i] += k[[i, j]] * (theta[j] - theta[i]).sin();
            }
        }
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }
    }

    PyArray1::from_owned_array(py, theta)
}

/// Compute Kuramoto order parameter R from phase array.
/// R = (1/N) |sum_i exp(i * theta_i)|
#[pyfunction]
fn order_parameter(theta: PyReadonlyArray1<'_, f64>) -> f64 {
    let theta = theta.as_array();
    let n = theta.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for &t in theta.iter() {
        re += t.cos();
        im += t.sin();
    }
    (re * re + im * im).sqrt() / n
}

/// Parallel classical Kuramoto trajectory.
/// Returns (times, R_values) for each timestep.
#[pyfunction]
fn kuramoto_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let mut theta = theta0.as_array().to_owned();
    let omega_arr = omega.as_array();
    let k_arr = k.as_array();
    let n = theta.len();

    let mut times = Array1::<f64>::zeros(n_steps + 1);
    let mut r_values = Array1::<f64>::zeros(n_steps + 1);

    // Initial R
    let (mut re, mut im) = (0.0, 0.0);
    for &t in theta.iter() { re += t.cos(); im += t.sin(); }
    r_values[0] = (re * re + im * im).sqrt() / n as f64;

    for step in 0..n_steps {
        let mut dtheta = Array1::<f64>::zeros(n);
        for i in 0..n {
            dtheta[i] = omega_arr[i];
            for j in 0..n {
                dtheta[i] += k_arr[[i, j]] * (theta[j] - theta[i]).sin();
            }
        }
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }

        times[step + 1] = (step + 1) as f64 * dt;
        let (mut re, mut im) = (0.0, 0.0);
        for &t in theta.iter() { re += t.cos(); im += t.sin(); }
        r_values[step + 1] = (re * re + im * im).sqrt() / n as f64;
    }

    (
        PyArray1::from_owned_array(py, times),
        PyArray1::from_owned_array(py, r_values),
    )
}

#[pymodule]
fn scpn_quantum_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pec_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(pec_sample_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(build_knm, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto_euler, m)?)?;
    m.add_function(wrap_pyfunction!(order_parameter, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto_trajectory, m)?)?;
    Ok(())
}
