// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Classical Kuramoto ODE Integration

//! Classical Kuramoto oscillator dynamics.
//!
//! Vectorised Euler integration of the coupled phase equation:
//! dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i)
//!
//! Includes order parameter R = (1/N)|��_i exp(iθ_i)| computation
//! and full trajectory recording.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::validation::{validate_contiguous_slice, validate_positive};

/// Classical Kuramoto ODE step (vectorised, no Python overhead).
/// θ' = ω + K @ sin(��_outer − θ_inner)
/// Returns new θ after n_steps of Euler integration.
#[pyfunction]
pub fn kuramoto_euler<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_positive(dt, "dt")?;
    let theta0_slice = validate_phase_vector(&theta0, "theta0")?;
    let omega_slice = validate_phase_vector(&omega, "omega")?;
    let n = theta0_slice.len();
    let k_flat = validate_square_matrix(&k, n, "k")?;
    if omega_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "omega must have length {n}, got {}",
            omega_slice.len()
        )));
    }
    let mut theta = Array1::from_vec(theta0_slice.to_vec());

    for _ in 0..n_steps {
        let theta_slice = theta
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("theta buffer is not contiguous"))?;
        let dtheta = phase_step_pairwise(theta_slice, omega_slice, &k_flat, n);
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }
    }

    Ok(PyArray1::from_owned_array(py, theta))
}

/// Compute Kuramoto order parameter R from phase array.
/// R = (1/N) |Σ_i exp(i × θ_i)|
#[pyfunction]
pub fn order_parameter(theta: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(order_parameter_inner(theta))
}

/// Pure Rust order parameter (no PyO3).
pub fn order_parameter_inner(theta: &[f64]) -> f64 {
    let n = theta.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for &t in theta {
        re += t.cos();
        im += t.sin();
    }
    (re * re + im * im).sqrt() / n
}

/// Compute the gradient ∂R/∂θ of the Kuramoto order parameter R = (1/N)|Σ exp(i θ)|.
///
/// With C = ⟨cos θ⟩, S = ⟨sin θ⟩ and R = hypot(C, S):
///   ∂R/∂θ_j = (S cos θ_j - C sin θ_j) / (N R) = (1/N) sin(ψ - θ_j),
/// where ψ = atan2(S, C) is the mean phase. The incoherent state R = 0 has an
/// undefined mean phase and returns the zero subgradient.
#[pyfunction]
pub fn order_parameter_gradient<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let gradient = Array1::from_vec(order_parameter_gradient_inner(theta));
    Ok(PyArray1::from_owned_array(py, gradient))
}

/// Pure Rust order parameter gradient (no PyO3).
pub fn order_parameter_gradient_inner(theta: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; theta.len()];
    if theta.is_empty() {
        return out;
    }
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &t in theta {
        re += t.cos();
        im += t.sin();
    }
    let count = theta.len() as f64;
    let cos_mean = re / count;
    let sin_mean = im / count;
    let magnitude = (cos_mean * cos_mean + sin_mean * sin_mean).sqrt();
    if magnitude == 0.0 {
        return out;
    }
    let scale = 1.0 / (count * magnitude);
    for (slot, &t) in out.iter_mut().zip(theta.iter()) {
        *slot = (sin_mean * t.cos() - cos_mean * t.sin()) * scale;
    }
    out
}

/// Compute the Hessian ∂²R/∂θ_i∂θ_j of the Kuramoto order parameter.
///
/// With C = ⟨cos θ⟩, S = ⟨sin θ⟩, R = hypot(C, S) and the alignment
/// a_j = cos(ψ − θ_j) = (C cos θ_j + S sin θ_j) / R:
///   H_ij = a_i a_j / (N² R) − δ_ij a_j / N.
/// The matrix is symmetric and every row sums to zero; the incoherent state R = 0
/// returns the zero matrix.
#[pyfunction]
pub fn order_parameter_hessian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let hessian = order_parameter_hessian_inner(theta);
    Ok(PyArray2::from_owned_array(py, hessian))
}

/// Pure Rust order parameter Hessian (no PyO3), returned as a row-major N×N array.
pub fn order_parameter_hessian_inner(theta: &[f64]) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    if n == 0 {
        return out;
    }
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &t in theta {
        re += t.cos();
        im += t.sin();
    }
    let count = n as f64;
    let cos_mean = re / count;
    let sin_mean = im / count;
    let magnitude = (cos_mean * cos_mean + sin_mean * sin_mean).sqrt();
    if magnitude == 0.0 {
        return out;
    }
    let aligned: Vec<f64> = theta
        .iter()
        .map(|&t| (cos_mean * t.cos() + sin_mean * t.sin()) / magnitude)
        .collect();
    let scale = 1.0 / (count * count * magnitude);
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = aligned[i] * aligned[j] * scale;
        }
        out[[i, i]] -= aligned[i] / count;
    }
    out
}

/// Compute the circular mean phase ψ = atan2(⟨sin θ⟩, ⟨cos θ⟩) of a Kuramoto ensemble.
///
/// The 1/N scaling cancels inside atan2, so the raw sums are used. An empty input and
/// the incoherent state both report 0.0.
#[pyfunction]
pub fn mean_phase(theta: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(mean_phase_inner(theta))
}

/// Pure Rust mean phase (no PyO3).
pub fn mean_phase_inner(theta: &[f64]) -> f64 {
    if theta.is_empty() {
        return 0.0;
    }
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &t in theta {
        re += t.cos();
        im += t.sin();
    }
    im.atan2(re)
}

/// Compute the gradient ∂ψ/∂θ of the Kuramoto mean phase.
///
/// With C = ⟨cos θ⟩, S = ⟨sin θ⟩ and R = hypot(C, S):
///   ∂ψ/∂θ_j = cos(ψ − θ_j) / (N R) = (C cos θ_j + S sin θ_j) / (N R²).
/// The components sum to one (a global phase shift advances ψ identically). The
/// incoherent state R = 0 returns the zero gradient.
#[pyfunction]
pub fn mean_phase_gradient<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let gradient = Array1::from_vec(mean_phase_gradient_inner(theta));
    Ok(PyArray1::from_owned_array(py, gradient))
}

/// Pure Rust mean phase gradient (no PyO3).
pub fn mean_phase_gradient_inner(theta: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; theta.len()];
    if theta.is_empty() {
        return out;
    }
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &t in theta {
        re += t.cos();
        im += t.sin();
    }
    let count = theta.len() as f64;
    let cos_mean = re / count;
    let sin_mean = im / count;
    let magnitude = (cos_mean * cos_mean + sin_mean * sin_mean).sqrt();
    if magnitude == 0.0 {
        return out;
    }
    let scale = 1.0 / (count * magnitude * magnitude);
    for (slot, &t) in out.iter_mut().zip(theta.iter()) {
        *slot = (cos_mean * t.cos() + sin_mean * t.sin()) * scale;
    }
    out
}

/// Compute the Hessian ∂²ψ/∂θ_i∂θ_j of the Kuramoto mean phase.
///
/// With c_k = cos(ψ − θ_k) = (C cos θ_k + S sin θ_k) / R and
/// s_k = sin(ψ − θ_k) = (S cos θ_k − C sin θ_k) / R:
///   H_ij = δ_ij s_j / (N R) − (s_i c_j + c_i s_j) / (N² R²).
/// The matrix is symmetric and every row sums to zero; the incoherent state R = 0
/// returns the zero matrix.
#[pyfunction]
pub fn mean_phase_hessian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let hessian = mean_phase_hessian_inner(theta);
    Ok(PyArray2::from_owned_array(py, hessian))
}

/// Pure Rust mean phase Hessian (no PyO3), returned as a row-major N×N array.
pub fn mean_phase_hessian_inner(theta: &[f64]) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    if n == 0 {
        return out;
    }
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &t in theta {
        re += t.cos();
        im += t.sin();
    }
    let count = n as f64;
    let cos_mean = re / count;
    let sin_mean = im / count;
    let magnitude = (cos_mean * cos_mean + sin_mean * sin_mean).sqrt();
    if magnitude == 0.0 {
        return out;
    }
    let aligned_cos: Vec<f64> = theta
        .iter()
        .map(|&t| (cos_mean * t.cos() + sin_mean * t.sin()) / magnitude)
        .collect();
    let aligned_sin: Vec<f64> = theta
        .iter()
        .map(|&t| (sin_mean * t.cos() - cos_mean * t.sin()) / magnitude)
        .collect();
    let scale = 1.0 / (count * count * magnitude * magnitude);
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] =
                -(aligned_sin[i] * aligned_cos[j] + aligned_cos[i] * aligned_sin[j]) * scale;
        }
        out[[i, i]] += aligned_sin[i] / (count * magnitude);
    }
    out
}

/// Compute the m-th Daido order parameter r_m = (1/N)|Σ exp(i m θ)|.
///
/// Detects m-cluster synchronisation; for m = 1 it is the ordinary Kuramoto order
/// parameter. The harmonic order m must be a positive integer.
#[pyfunction]
pub fn daido_order_parameter(theta: PyReadonlyArray1<'_, f64>, m: i64) -> PyResult<f64> {
    if m < 1 {
        return Err(PyValueError::new_err(format!(
            "Daido harmonic order m must be a positive integer, got {m}"
        )));
    }
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(daido_order_parameter_inner(theta, m as f64))
}

/// Pure Rust m-th Daido order parameter (no PyO3).
pub fn daido_order_parameter_inner(theta: &[f64], m: f64) -> f64 {
    if theta.is_empty() {
        return 0.0;
    }
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &t in theta {
        re += (m * t).cos();
        im += (m * t).sin();
    }
    (re * re + im * im).sqrt() / theta.len() as f64
}

/// Compute the gradient ∂r_m/∂θ of the m-th Daido order parameter.
///
/// ∂r_m/∂θ_j = (m/N) sin(ψ_m − m θ_j) = (m / (N R_m)) (S_m cos(m θ_j) − C_m sin(m θ_j)).
/// The components sum to zero; the incoherent state R_m = 0 returns the zero gradient.
#[pyfunction]
pub fn daido_order_parameter_gradient<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    m: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if m < 1 {
        return Err(PyValueError::new_err(format!(
            "Daido harmonic order m must be a positive integer, got {m}"
        )));
    }
    let theta = validate_phase_vector(&theta, "theta")?;
    let gradient = Array1::from_vec(daido_order_parameter_gradient_inner(theta, m as f64));
    Ok(PyArray1::from_owned_array(py, gradient))
}

/// Pure Rust m-th Daido order parameter gradient (no PyO3).
pub fn daido_order_parameter_gradient_inner(theta: &[f64], m: f64) -> Vec<f64> {
    let mut out = vec![0.0_f64; theta.len()];
    if theta.is_empty() {
        return out;
    }
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &t in theta {
        re += (m * t).cos();
        im += (m * t).sin();
    }
    let count = theta.len() as f64;
    let cos_mean = re / count;
    let sin_mean = im / count;
    let magnitude = (cos_mean * cos_mean + sin_mean * sin_mean).sqrt();
    if magnitude == 0.0 {
        return out;
    }
    let scale = m / (count * magnitude);
    for (slot, &t) in out.iter_mut().zip(theta.iter()) {
        *slot = (sin_mean * (m * t).cos() - cos_mean * (m * t).sin()) * scale;
    }
    out
}

/// Compute the Hessian ∂²r_m/∂θ_i∂θ_j of the m-th Daido order parameter.
///
/// With a_k = cos(ψ_m − m θ_k) = (C_m cos(m θ_k) + S_m sin(m θ_k)) / R_m:
///   H_ij = m² (a_i a_j / (N² R_m) − δ_ij a_j / N).
/// The matrix is symmetric and every row sums to zero; the incoherent state R_m = 0
/// returns the zero matrix. For m = 1 it is the order-parameter Hessian.
#[pyfunction]
pub fn daido_order_parameter_hessian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    m: i64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if m < 1 {
        return Err(PyValueError::new_err(format!(
            "Daido harmonic order m must be a positive integer, got {m}"
        )));
    }
    let theta = validate_phase_vector(&theta, "theta")?;
    let hessian = daido_order_parameter_hessian_inner(theta, m as f64);
    Ok(PyArray2::from_owned_array(py, hessian))
}

/// Pure Rust m-th Daido order parameter Hessian (no PyO3), returned row-major.
pub fn daido_order_parameter_hessian_inner(theta: &[f64], m: f64) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    if n == 0 {
        return out;
    }
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &t in theta {
        re += (m * t).cos();
        im += (m * t).sin();
    }
    let count = n as f64;
    let cos_mean = re / count;
    let sin_mean = im / count;
    let magnitude = (cos_mean * cos_mean + sin_mean * sin_mean).sqrt();
    if magnitude == 0.0 {
        return out;
    }
    let aligned: Vec<f64> = theta
        .iter()
        .map(|&t| (cos_mean * (m * t).cos() + sin_mean * (m * t).sin()) / magnitude)
        .collect();
    let m_squared = m * m;
    let scale = m_squared / (count * count * magnitude);
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = aligned[i] * aligned[j] * scale;
        }
        out[[i, i]] -= m_squared * aligned[i] / count;
    }
    out
}

/// Compute the Kuramoto mean-field coupling force F_j = K (S cos θ_j − C sin θ_j).
///
/// This is the phase-coupling term of the all-to-all Kuramoto dynamics, with
/// C = <cos θ> and S = <sin θ>.
#[pyfunction]
pub fn mean_field_force<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(PyArray1::from_owned_array(
        py,
        mean_field_force_inner(theta, coupling),
    ))
}

/// Pure Rust mean-field force (no PyO3).
pub fn mean_field_force_inner(theta: &[f64], coupling: f64) -> Array1<f64> {
    let n = theta.len();
    if n == 0 {
        return Array1::zeros(0);
    }
    let (mut c, mut s) = (0.0_f64, 0.0_f64);
    for &t in theta {
        c += t.cos();
        s += t.sin();
    }
    let count = n as f64;
    let cos_mean = c / count;
    let sin_mean = s / count;
    Array1::from_iter(
        theta
            .iter()
            .map(|&t| coupling * (sin_mean * t.cos() - cos_mean * t.sin())),
    )
}

/// Compute the Kuramoto synchronisation stability Jacobian J_jk = ∂F_j/∂θ_k.
///
/// J_jk = (K/N) cos(θ_j − θ_k) − K δ_jk (C cos θ_j + S sin θ_j). The matrix is symmetric
/// and every row sums to zero (the global-phase Goldstone mode).
#[pyfunction]
pub fn mean_field_jacobian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(PyArray2::from_owned_array(
        py,
        mean_field_jacobian_inner(theta, coupling),
    ))
}

/// Pure Rust mean-field stability Jacobian (no PyO3), returned row-major.
pub fn mean_field_jacobian_inner(theta: &[f64], coupling: f64) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    if n == 0 {
        return out;
    }
    let (mut c, mut s) = (0.0_f64, 0.0_f64);
    for &t in theta {
        c += t.cos();
        s += t.sin();
    }
    let count = n as f64;
    let cos_mean = c / count;
    let sin_mean = s / count;
    let scale = coupling / count;
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = scale * (theta[i] - theta[j]).cos();
        }
        out[[i, i]] -= coupling * (cos_mean * theta[i].cos() + sin_mean * theta[i].sin());
    }
    out
}

fn validate_finite_slice(values: &[f64], name: &str) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

fn validate_phase_vector<'a>(
    values: &'a PyReadonlyArray1<'_, f64>,
    name: &str,
) -> PyResult<&'a [f64]> {
    let slice = validate_contiguous_slice(values, name)?;
    validate_finite_slice(slice, name)?;
    Ok(slice)
}

fn validate_square_matrix(
    matrix: &PyReadonlyArray2<'_, f64>,
    n: usize,
    name: &str,
) -> PyResult<Vec<f64>> {
    let arr = matrix.as_array();
    if arr.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape ({n}, {n}), got {:?}",
            arr.shape()
        )));
    }
    let slice = arr.as_slice().ok_or_else(|| {
        PyValueError::new_err(format!("{name} must be a C-contiguous NumPy array"))
    })?;
    validate_finite_slice(slice, name)?;
    Ok(slice.to_vec())
}

fn validate_rect_f64_matrix(
    matrix: &PyReadonlyArray2<'_, f64>,
    ncols: usize,
    name: &str,
) -> PyResult<(Vec<f64>, usize)> {
    let arr = matrix.as_array();
    if arr.ndim() != 2 || arr.shape()[1] != ncols {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (n_candidates, {ncols})"
        )));
    }
    let slice = arr.as_slice().ok_or_else(|| {
        PyValueError::new_err(format!("{name} must be a C-contiguous NumPy array"))
    })?;
    validate_finite_slice(slice, name)?;
    Ok((slice.to_vec(), arr.shape()[0]))
}

fn validate_hyperedges(
    hyperedges: &PyReadonlyArray2<'_, i64>,
    n: usize,
) -> PyResult<(Vec<i64>, usize)> {
    let edge_arr = hyperedges.as_array();
    if edge_arr.ndim() != 2 || edge_arr.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "hyperedges must have shape (n_edges, 3)",
        ));
    }
    let edge_slice = edge_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("hyperedges must be a C-contiguous NumPy array"))?;
    for &index in edge_slice {
        if index < 0 || index as usize >= n {
            return Err(PyValueError::new_err(format!(
                "hyperedge index {index} is outside [0, {n})"
            )));
        }
    }
    Ok((edge_slice.to_vec(), edge_arr.shape()[0]))
}

fn phase_step_pairwise(theta: &[f64], omega: &[f64], k_flat: &[f64], n: usize) -> Vec<f64> {
    let mut dtheta = vec![0.0; n];
    for i in 0..n {
        dtheta[i] = omega[i];
        for j in 0..n {
            dtheta[i] += k_flat[i * n + j] * (theta[j] - theta[i]).sin();
        }
    }
    dtheta
}

fn fill_time_and_r(
    times: &mut Array1<f64>,
    r_values: &mut Array1<f64>,
    step: usize,
    dt: f64,
    theta: &[f64],
) {
    times[step] = step as f64 * dt;
    r_values[step] = order_parameter_inner(theta);
}

/// Higher-order simplicial Kuramoto trajectory.
///
/// Pairwise term: K_ij sin(theta_j - theta_i).
/// Anchored triadic term: B_a sin(theta_j + theta_k - 2 theta_i) for each
/// hyperedge row (i, j, k).
#[pyfunction]
pub fn higher_order_kuramoto_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    hyperedges: PyReadonlyArray2<'_, i64>,
    hyper_weights: PyReadonlyArray1<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validate_positive(dt, "dt")?;
    let theta0_slice = validate_phase_vector(&theta0, "theta0")?;
    let omega_slice = validate_phase_vector(&omega, "omega")?;
    let n = theta0_slice.len();
    if omega_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "omega must have length {n}, got {}",
            omega_slice.len()
        )));
    }
    let k_flat = validate_square_matrix(&k, n, "k")?;
    let (edge_slice, n_edges) = validate_hyperedges(&hyperedges, n)?;
    let weight_slice = validate_phase_vector(&hyper_weights, "hyper_weights")?;
    if weight_slice.len() != n_edges {
        return Err(PyValueError::new_err(format!(
            "hyper_weights must have length {n_edges}, got {}",
            weight_slice.len()
        )));
    }
    let mut theta = Array1::from_vec(theta0_slice.to_vec());
    let mut times = Array1::<f64>::zeros(n_steps + 1);
    let mut r_values = Array1::<f64>::zeros(n_steps + 1);
    fill_time_and_r(
        &mut times,
        &mut r_values,
        0,
        dt,
        theta
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("theta buffer is not contiguous"))?,
    );

    for step in 0..n_steps {
        let theta_slice = theta
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("theta buffer is not contiguous"))?;
        let mut dtheta = phase_step_pairwise(theta_slice, omega_slice, &k_flat, n);
        for (edge_index, weight) in weight_slice.iter().enumerate() {
            let base = 3 * edge_index;
            let i = edge_slice[base] as usize;
            let j = edge_slice[base + 1] as usize;
            let l = edge_slice[base + 2] as usize;
            dtheta[i] += *weight * (theta_slice[j] + theta_slice[l] - 2.0 * theta_slice[i]).sin();
        }
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }
        fill_time_and_r(
            &mut times,
            &mut r_values,
            step + 1,
            dt,
            theta
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("theta buffer is not contiguous"))?,
        );
    }

    Ok((
        PyArray1::from_owned_array(py, times),
        PyArray1::from_owned_array(py, r_values),
    ))
}

/// Monitored Kuramoto trajectory with deterministic measurement-feedback closure.
///
/// The instantaneous readout is R_m = (1-strength)R + strength*target_R.  The
/// feedback term g(target_R - R_m) sin(psi - theta_i) pulls phases toward the
/// measured mean phase without changing the pairwise Kuramoto law.
#[pyfunction]
pub fn monitored_kuramoto_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    target_r: f64,
    monitor_gain: f64,
    measurement_strength: f64,
    dt: f64,
    n_steps: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validate_positive(dt, "dt")?;
    if !target_r.is_finite() || !(0.0..=1.0).contains(&target_r) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_r must be finite and in [0, 1]",
        ));
    }
    if !monitor_gain.is_finite() || monitor_gain < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "monitor_gain must be finite and non-negative",
        ));
    }
    if !measurement_strength.is_finite() || !(0.0..=1.0).contains(&measurement_strength) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "measurement_strength must be finite and in [0, 1]",
        ));
    }

    let theta0_slice = validate_phase_vector(&theta0, "theta0")?;
    let omega_slice = validate_phase_vector(&omega, "omega")?;
    let n = theta0_slice.len();
    if omega_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "omega must have length {n}, got {}",
            omega_slice.len()
        )));
    }
    let k_flat = validate_square_matrix(&k, n, "k")?;
    let mut theta = Array1::from_vec(theta0_slice.to_vec());
    let mut times = Array1::<f64>::zeros(n_steps + 1);
    let mut r_values = Array1::<f64>::zeros(n_steps + 1);
    let mut readouts = Array1::<f64>::zeros(n_steps + 1);
    let mut feedback = Array1::<f64>::zeros(n_steps + 1);

    for step in 0..=n_steps {
        let theta_slice = theta
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("theta buffer is not contiguous"))?;
        let r = order_parameter_inner(theta_slice);
        let readout = (1.0 - measurement_strength) * r + measurement_strength * target_r;
        times[step] = step as f64 * dt;
        r_values[step] = r;
        readouts[step] = readout;
        feedback[step] = monitor_gain * (target_r - readout);
        if step == n_steps {
            break;
        }
        let (mut re, mut im) = (0.0, 0.0);
        for &phase in theta_slice {
            re += phase.cos();
            im += phase.sin();
        }
        let mean_phase = im.atan2(re);
        let mut dtheta = phase_step_pairwise(theta_slice, omega_slice, &k_flat, n);
        for i in 0..n {
            dtheta[i] += feedback[step] * (mean_phase - theta_slice[i]).sin();
        }
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }
    }

    Ok((
        PyArray1::from_owned_array(py, times),
        PyArray1::from_owned_array(py, r_values),
        PyArray1::from_owned_array(py, readouts),
        PyArray1::from_owned_array(py, feedback),
    ))
}

/// PT-symmetric complex Kuramoto trajectory with balanced gain/loss.
///
/// The complex oscillator evolves as z_i' = (gain_i + i dtheta_i) z_i and is
/// renormalised after each Euler step so the returned R isolates phase locking
/// while pt_norm and imbalance expose the non-Hermitian gain/loss channel.
#[pyfunction]
pub fn pt_symmetric_kuramoto_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    gain_loss: PyReadonlyArray1<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validate_positive(dt, "dt")?;
    let theta_slice = validate_phase_vector(&theta0, "theta0")?;
    let omega_slice = validate_phase_vector(&omega, "omega")?;
    let gain_slice = validate_phase_vector(&gain_loss, "gain_loss")?;
    let n = theta_slice.len();
    if omega_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "omega must have length {n}, got {}",
            omega_slice.len()
        )));
    }
    if gain_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "gain_loss must have length {n}, got {}",
            gain_slice.len()
        )));
    }
    let k_flat = validate_square_matrix(&k, n, "k")?;
    let gain_sum: f64 = gain_slice.iter().sum();
    if gain_sum.abs() > 1e-10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "gain_loss must sum to zero for balanced PT symmetry",
        ));
    }

    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for &phase in theta_slice {
        re.push(phase.cos());
        im.push(phase.sin());
    }
    let mut theta = Array1::from_vec(theta_slice.to_vec());
    let mut times = Array1::<f64>::zeros(n_steps + 1);
    let mut r_values = Array1::<f64>::zeros(n_steps + 1);
    let mut pt_norm = Array1::<f64>::zeros(n_steps + 1);
    let mut imbalance = Array1::<f64>::zeros(n_steps + 1);

    for step in 0..=n_steps {
        let mut total_re = 0.0;
        let mut total_im = 0.0;
        let mut norm = 0.0;
        let mut signed_power = 0.0;
        for i in 0..n {
            total_re += re[i];
            total_im += im[i];
            let power = re[i] * re[i] + im[i] * im[i];
            norm += power;
            signed_power += gain_slice[i] * power;
            theta[i] = im[i].atan2(re[i]);
        }
        times[step] = step as f64 * dt;
        r_values[step] =
            (total_re * total_re + total_im * total_im).sqrt() / norm.sqrt() / (n as f64).sqrt();
        pt_norm[step] = norm / n as f64;
        imbalance[step] = signed_power;
        if step == n_steps {
            break;
        }
        let theta_now = theta
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("theta buffer is not contiguous"))?;
        let dtheta = phase_step_pairwise(theta_now, omega_slice, &k_flat, n);
        for i in 0..n {
            let z_re = re[i];
            let z_im = im[i];
            let gain = gain_slice[i];
            let freq = dtheta[i];
            re[i] += dt * (gain * z_re - freq * z_im);
            im[i] += dt * (freq * z_re + gain * z_im);
        }
        let norm_after = re
            .iter()
            .zip(im.iter())
            .map(|(x, y)| x * x + y * y)
            .sum::<f64>()
            .sqrt();
        if norm_after > 0.0 {
            let scale = (n as f64).sqrt() / norm_after;
            for i in 0..n {
                re[i] *= scale;
                im[i] *= scale;
            }
        }
    }

    Ok((
        PyArray1::from_owned_array(py, times),
        PyArray1::from_owned_array(py, r_values),
        PyArray1::from_owned_array(py, pt_norm),
        PyArray1::from_owned_array(py, imbalance),
    ))
}

/// Batch candidate features for automated Kuramoto witness discovery.
///
/// Candidate columns are `(coupling_scale, omega_scale, phase_bias)`.  The
/// kernel integrates the classical Kuramoto dynamics and returns final R,
/// mean pairwise cos(theta_i-theta_j), and final phases for downstream witness
/// scoring in Python.
#[pyfunction]
pub fn kuramoto_witness_candidate_features<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    candidates: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    validate_positive(dt, "dt")?;
    let theta0_slice = validate_phase_vector(&theta0, "theta0")?;
    let omega_slice = validate_phase_vector(&omega, "omega")?;
    let n = theta0_slice.len();
    if omega_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "omega must have length {n}, got {}",
            omega_slice.len()
        )));
    }
    let k_flat = validate_square_matrix(&k, n, "k")?;
    let (cand_slice, n_candidates) = validate_rect_f64_matrix(&candidates, 3, "candidates")?;
    let mut final_r = Array1::<f64>::zeros(n_candidates);
    let mut mean_corr = Array1::<f64>::zeros(n_candidates);
    let mut final_theta = Array2::<f64>::zeros((n_candidates, n));

    for candidate_index in 0..n_candidates {
        let base = 3 * candidate_index;
        let coupling_scale = cand_slice[base];
        let omega_scale = cand_slice[base + 1];
        let phase_bias = cand_slice[base + 2];
        if coupling_scale < 0.0 || omega_scale < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "candidate coupling_scale and omega_scale must be non-negative",
            ));
        }
        let mut theta: Vec<f64> = theta0_slice
            .iter()
            .map(|value| value + phase_bias)
            .collect();
        let omega_scaled: Vec<f64> = omega_slice
            .iter()
            .map(|value| value * omega_scale)
            .collect();
        let k_scaled: Vec<f64> = k_flat.iter().map(|value| value * coupling_scale).collect();

        for _ in 0..n_steps {
            let dtheta = phase_step_pairwise(&theta, &omega_scaled, &k_scaled, n);
            for i in 0..n {
                theta[i] += dt * dtheta[i];
            }
        }

        final_r[candidate_index] = order_parameter_inner(&theta);
        let mut corr_sum = 0.0;
        let mut n_pairs = 0usize;
        for i in 0..n {
            final_theta[[candidate_index, i]] = theta[i];
            for j in (i + 1)..n {
                corr_sum += (theta[i] - theta[j]).cos();
                n_pairs += 1;
            }
        }
        mean_corr[candidate_index] = if n_pairs > 0 {
            corr_sum / n_pairs as f64
        } else {
            1.0
        };
    }

    Ok((
        PyArray1::from_owned_array(py, final_r),
        PyArray1::from_owned_array(py, mean_corr),
        PyArray2::from_owned_array(py, final_theta),
    ))
}

/// Parallel classical Kuramoto trajectory.
/// Returns (times, R_values) for each timestep.
#[pyfunction]
pub fn kuramoto_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validate_positive(dt, "dt")?;
    let theta0_slice = validate_phase_vector(&theta0, "theta0")?;
    let omega_slice = validate_phase_vector(&omega, "omega")?;
    let n = theta0_slice.len();
    if omega_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "omega must have length {n}, got {}",
            omega_slice.len()
        )));
    }
    let k_flat = validate_square_matrix(&k, n, "k")?;
    let mut theta = Array1::from_vec(theta0_slice.to_vec());

    let mut times = Array1::<f64>::zeros(n_steps + 1);
    let mut r_values = Array1::<f64>::zeros(n_steps + 1);

    // Initial R
    r_values[0] = order_parameter_inner(
        theta
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("theta buffer is not contiguous"))?,
    );

    for step in 0..n_steps {
        let theta_slice = theta
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("theta buffer is not contiguous"))?;
        let dtheta = phase_step_pairwise(theta_slice, omega_slice, &k_flat, n);
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }

        times[step + 1] = (step + 1) as f64 * dt;
        r_values[step + 1] = order_parameter_inner(
            theta
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("theta buffer is not contiguous"))?,
        );
    }

    Ok((
        PyArray1::from_owned_array(py, times),
        PyArray1::from_owned_array(py, r_values),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_parameter_all_aligned() {
        let theta = vec![0.0; 8];
        let r = order_parameter_inner(&theta);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_order_parameter_opposite() {
        let theta = vec![0.0, std::f64::consts::PI];
        let r = order_parameter_inner(&theta);
        assert!(r < 1e-10);
    }

    #[test]
    fn test_order_parameter_bounded() {
        let theta = vec![0.1, 0.5, 1.2, 2.5, 3.8, 5.0];
        let r = order_parameter_inner(&theta);
        assert!((0.0..=1.0 + 1e-12).contains(&r));
    }

    #[test]
    fn test_order_parameter_single() {
        let theta = vec![2.7];
        let r = order_parameter_inner(&theta);
        assert!((r - 1.0).abs() < 1e-12, "single oscillator → R = 1");
    }

    #[test]
    fn test_order_parameter_gradient_matches_finite_difference() {
        let theta = vec![0.1, 0.5, 1.2, 2.5, 3.8, 5.0];
        let grad = order_parameter_gradient_inner(&theta);
        let h = 1e-6;
        for j in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[j] += h;
            minus[j] -= h;
            let fd = (order_parameter_inner(&plus) - order_parameter_inner(&minus)) / (2.0 * h);
            assert!((grad[j] - fd).abs() < 1e-7, "grad[{j}]={} fd={fd}", grad[j]);
        }
    }

    #[test]
    fn test_order_parameter_gradient_matches_sync_force_identity() {
        // ∂R/∂θ_j = (1/N) sin(ψ - θ_j) with ψ = atan2(S, C).
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let grad = order_parameter_gradient_inner(&theta);
        let n = theta.len() as f64;
        let cos_mean: f64 = theta.iter().map(|t| t.cos()).sum::<f64>() / n;
        let sin_mean: f64 = theta.iter().map(|t| t.sin()).sum::<f64>() / n;
        let psi = sin_mean.atan2(cos_mean);
        for (j, &t) in theta.iter().enumerate() {
            let expected = (psi - t).sin() / n;
            assert!((grad[j] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_order_parameter_gradient_sums_to_zero() {
        // A global phase shift leaves R invariant, so the gradient sums to zero.
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5, 6.0];
        let total: f64 = order_parameter_gradient_inner(&theta).iter().sum();
        assert!(total.abs() < 1e-12, "gradient sum = {total}");
    }

    #[test]
    fn test_order_parameter_gradient_aligned_is_zero() {
        let grad = order_parameter_gradient_inner(&[0.7; 8]);
        assert!(grad.iter().all(|g| g.abs() < 1e-12));
    }

    #[test]
    fn test_order_parameter_gradient_single_is_zero() {
        let grad = order_parameter_gradient_inner(&[2.7]);
        assert_eq!(grad.len(), 1);
        assert!(grad[0].abs() < 1e-12, "single oscillator → ∂R/∂θ = 0");
    }

    #[test]
    fn test_order_parameter_gradient_empty() {
        assert!(order_parameter_gradient_inner(&[]).is_empty());
    }

    #[test]
    fn test_order_parameter_hessian_matches_finite_difference_of_gradient() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let hessian = order_parameter_hessian_inner(&theta);
        let h = 1e-6;
        for i in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[i] += h;
            minus[i] -= h;
            let grad_plus = order_parameter_gradient_inner(&plus);
            let grad_minus = order_parameter_gradient_inner(&minus);
            for j in 0..theta.len() {
                let fd = (grad_plus[j] - grad_minus[j]) / (2.0 * h);
                assert!((hessian[[i, j]] - fd).abs() < 1e-6, "H[{i},{j}]");
            }
        }
    }

    #[test]
    fn test_order_parameter_hessian_is_symmetric() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5];
        let hessian = order_parameter_hessian_inner(&theta);
        for i in 0..theta.len() {
            for j in 0..theta.len() {
                assert!((hessian[[i, j]] - hessian[[j, i]]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_order_parameter_hessian_rows_sum_to_zero() {
        // A global phase shift leaves R invariant, so every Hessian row sums to zero.
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5, 6.0];
        let hessian = order_parameter_hessian_inner(&theta);
        let n = theta.len();
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| hessian[[i, j]]).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_order_parameter_hessian_aligned_curvature() {
        // Fully synchronised (R = 1): the gradient vanishes but the Hessian does not.
        // Every alignment equals 1, so H_ij = 1/N² off the diagonal and 1/N² − 1/N on it
        // — a negative semidefinite matrix, since R = 1 is the maximum.
        let n = 6usize;
        let hessian = order_parameter_hessian_inner(&[0.7; 6]);
        let count = n as f64;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j {
                    1.0 / (count * count) - 1.0 / count
                } else {
                    1.0 / (count * count)
                };
                assert!((hessian[[i, j]] - expected).abs() < 1e-12, "H[{i},{j}]");
            }
        }
    }

    #[test]
    fn test_order_parameter_hessian_single_is_zero() {
        let hessian = order_parameter_hessian_inner(&[2.7]);
        assert_eq!(hessian.shape(), &[1, 1]);
        assert!(hessian[[0, 0]].abs() < 1e-12, "single oscillator → ∂²R/∂θ² = 0");
    }

    #[test]
    fn test_order_parameter_hessian_empty() {
        assert_eq!(order_parameter_hessian_inner(&[]).shape(), &[0, 0]);
    }

    #[test]
    fn test_mean_phase_matches_atan2() {
        let theta: Vec<f64> = vec![0.3, -1.1, 2.0, 0.7];
        let n = theta.len() as f64;
        let cos_mean: f64 = theta.iter().map(|t| t.cos()).sum::<f64>() / n;
        let sin_mean: f64 = theta.iter().map(|t| t.sin()).sum::<f64>() / n;
        assert!((mean_phase_inner(&theta) - sin_mean.atan2(cos_mean)).abs() < 1e-12);
    }

    #[test]
    fn test_mean_phase_single_is_identity() {
        assert!((mean_phase_inner(&[2.7]) - 2.7).abs() < 1e-12, "ψ of one oscillator is its phase");
    }

    #[test]
    fn test_mean_phase_empty_is_zero() {
        assert_eq!(mean_phase_inner(&[]), 0.0);
    }

    #[test]
    fn test_mean_phase_gradient_matches_finite_difference() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let grad = mean_phase_gradient_inner(&theta);
        let h = 1e-6;
        for j in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[j] += h;
            minus[j] -= h;
            let fd = (mean_phase_inner(&plus) - mean_phase_inner(&minus)) / (2.0 * h);
            assert!((grad[j] - fd).abs() < 1e-7, "grad[{j}]");
        }
    }

    #[test]
    fn test_mean_phase_gradient_sums_to_one() {
        // A global phase shift advances ψ identically, so the gradient sums to one.
        let total: f64 = mean_phase_gradient_inner(&[0.1, 0.9, 2.3, 3.1, 5.5]).iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "sum = {total}");
    }

    #[test]
    fn test_mean_phase_gradient_single_is_one() {
        let grad = mean_phase_gradient_inner(&[2.7]);
        assert_eq!(grad.len(), 1);
        assert!((grad[0] - 1.0).abs() < 1e-12, "∂ψ/∂θ = 1 for one oscillator");
    }

    #[test]
    fn test_mean_phase_gradient_empty() {
        assert!(mean_phase_gradient_inner(&[]).is_empty());
    }

    #[test]
    fn test_mean_phase_hessian_matches_finite_difference_of_gradient() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let hessian = mean_phase_hessian_inner(&theta);
        let h = 1e-6;
        for i in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[i] += h;
            minus[i] -= h;
            let grad_plus = mean_phase_gradient_inner(&plus);
            let grad_minus = mean_phase_gradient_inner(&minus);
            for j in 0..theta.len() {
                let fd = (grad_plus[j] - grad_minus[j]) / (2.0 * h);
                assert!((hessian[[i, j]] - fd).abs() < 1e-6, "H[{i},{j}]");
            }
        }
    }

    #[test]
    fn test_mean_phase_hessian_is_symmetric() {
        let hessian = mean_phase_hessian_inner(&[0.1, 0.9, 2.3, 3.1, 5.5]);
        for i in 0..5 {
            for j in 0..5 {
                assert!((hessian[[i, j]] - hessian[[j, i]]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_mean_phase_hessian_rows_sum_to_zero() {
        // The second derivative along a global phase shift vanishes, so each row sums to zero.
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5, 6.0];
        let hessian = mean_phase_hessian_inner(&theta);
        let n = theta.len();
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| hessian[[i, j]]).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_mean_phase_hessian_single_is_zero() {
        let hessian = mean_phase_hessian_inner(&[2.7]);
        assert_eq!(hessian.shape(), &[1, 1]);
        assert!(hessian[[0, 0]].abs() < 1e-12, "∂²ψ/∂θ² = 0 for one oscillator");
    }

    #[test]
    fn test_mean_phase_hessian_empty() {
        assert_eq!(mean_phase_hessian_inner(&[]).shape(), &[0, 0]);
    }

    #[test]
    fn test_daido_m1_matches_order_parameter() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        assert!(
            (daido_order_parameter_inner(&theta, 1.0) - order_parameter_inner(&theta)).abs()
                < 1e-12
        );
        let daido_grad = daido_order_parameter_gradient_inner(&theta, 1.0);
        let order_grad = order_parameter_gradient_inner(&theta);
        for (a, b) in daido_grad.iter().zip(order_grad.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_daido_detects_two_clusters() {
        // Two antipodal clusters: the first harmonic cancels (r_1 = 0) but the second is
        // perfectly coherent (r_2 = 1).
        let theta = vec![0.0, 0.0, std::f64::consts::PI, std::f64::consts::PI];
        assert!(daido_order_parameter_inner(&theta, 1.0) < 1e-10);
        assert!((daido_order_parameter_inner(&theta, 2.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_daido_gradient_matches_finite_difference() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let m = 3.0;
        let grad = daido_order_parameter_gradient_inner(&theta, m);
        let h = 1e-6;
        for j in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[j] += h;
            minus[j] -= h;
            let fd = (daido_order_parameter_inner(&plus, m)
                - daido_order_parameter_inner(&minus, m))
                / (2.0 * h);
            assert!((grad[j] - fd).abs() < 1e-6, "grad[{j}]");
        }
    }

    #[test]
    fn test_daido_gradient_sums_to_zero() {
        let total: f64 = daido_order_parameter_gradient_inner(&[0.1, 0.9, 2.3, 3.1, 5.5], 2.0)
            .iter()
            .sum();
        assert!(total.abs() < 1e-12, "sum = {total}");
    }

    #[test]
    fn test_daido_empty() {
        assert_eq!(daido_order_parameter_inner(&[], 2.0), 0.0);
        assert!(daido_order_parameter_gradient_inner(&[], 2.0).is_empty());
    }

    #[test]
    fn test_daido_hessian_m1_matches_order_parameter_hessian() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let daido = daido_order_parameter_hessian_inner(&theta, 1.0);
        let order = order_parameter_hessian_inner(&theta);
        for i in 0..theta.len() {
            for j in 0..theta.len() {
                assert!((daido[[i, j]] - order[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_daido_hessian_matches_finite_difference_of_gradient() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let m = 2.0;
        let hessian = daido_order_parameter_hessian_inner(&theta, m);
        let h = 1e-6;
        for i in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[i] += h;
            minus[i] -= h;
            let grad_plus = daido_order_parameter_gradient_inner(&plus, m);
            let grad_minus = daido_order_parameter_gradient_inner(&minus, m);
            for j in 0..theta.len() {
                let fd = (grad_plus[j] - grad_minus[j]) / (2.0 * h);
                assert!((hessian[[i, j]] - fd).abs() < 1e-6, "H[{i},{j}]");
            }
        }
    }

    #[test]
    fn test_daido_hessian_is_symmetric_and_rows_sum_to_zero() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5, 6.0];
        let hessian = daido_order_parameter_hessian_inner(&theta, 3.0);
        let n = theta.len();
        for i in 0..n {
            for j in 0..n {
                assert!((hessian[[i, j]] - hessian[[j, i]]).abs() < 1e-15);
            }
            let row_sum: f64 = (0..n).map(|j| hessian[[i, j]]).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_daido_hessian_empty() {
        assert_eq!(daido_order_parameter_hessian_inner(&[], 2.0).shape(), &[0, 0]);
    }

    #[test]
    fn test_mean_field_force_matches_closed_form() {
        let theta = vec![0.3, -1.1, 2.0, 0.7];
        let k = 1.7;
        let force = mean_field_force_inner(&theta, k);
        let n = theta.len() as f64;
        let c: f64 = theta.iter().map(|t| t.cos()).sum::<f64>() / n;
        let s: f64 = theta.iter().map(|t| t.sin()).sum::<f64>() / n;
        for (j, &t) in theta.iter().enumerate() {
            let expected = k * (s * t.cos() - c * t.sin());
            assert!((force[j] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_mean_field_jacobian_matches_finite_difference_of_force() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let k = 2.5;
        let jacobian = mean_field_jacobian_inner(&theta, k);
        let h = 1e-6;
        for kk in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[kk] += h;
            minus[kk] -= h;
            let force_plus = mean_field_force_inner(&plus, k);
            let force_minus = mean_field_force_inner(&minus, k);
            for j in 0..theta.len() {
                let fd = (force_plus[j] - force_minus[j]) / (2.0 * h);
                assert!((jacobian[[j, kk]] - fd).abs() < 1e-6, "J[{j},{kk}]");
            }
        }
    }

    #[test]
    fn test_mean_field_jacobian_is_symmetric_and_rows_sum_to_zero() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5];
        let jacobian = mean_field_jacobian_inner(&theta, 1.3);
        let n = theta.len();
        for i in 0..n {
            for j in 0..n {
                assert!((jacobian[[i, j]] - jacobian[[j, i]]).abs() < 1e-15);
            }
            let row_sum: f64 = (0..n).map(|j| jacobian[[i, j]]).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_mean_field_empty() {
        assert!(mean_field_force_inner(&[], 1.0).is_empty());
        assert_eq!(mean_field_jacobian_inner(&[], 1.0).shape(), &[0, 0]);
    }

    #[test]
    fn test_order_parameter_uniform_circle() {
        // N equally spaced phases → R ≈ 0
        let n = 100;
        let theta: Vec<f64> = (0..n)
            .map(|i| 2.0 * std::f64::consts::PI * i as f64 / n as f64)
            .collect();
        let r = order_parameter_inner(&theta);
        assert!(r < 0.05, "uniform circle → R ≈ 0, got {r}");
    }

    #[test]
    fn test_phase_step_pairwise_uses_all_couplings() {
        let theta = vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI];
        let omega = vec![0.1, -0.2, 0.3];
        let k = vec![0.0, 0.5, 0.25, 0.5, 0.0, 0.75, 0.25, 0.75, 0.0];

        let dtheta = phase_step_pairwise(&theta, &omega, &k, 3);

        assert!((dtheta[0] - 0.6).abs() < 1e-12);
        assert!((dtheta[1] - 0.05).abs() < 1e-12);
        assert!((dtheta[2] - (-0.45)).abs() < 1e-12);
    }

    #[test]
    fn test_fill_time_and_r_records_grid_point() {
        let mut times = Array1::<f64>::zeros(2);
        let mut r_values = Array1::<f64>::zeros(2);
        let theta = vec![0.0, 0.0, std::f64::consts::PI];

        fill_time_and_r(&mut times, &mut r_values, 1, 0.125, &theta);

        assert!((times[1] - 0.125).abs() < 1e-12);
        assert!((r_values[1] - (1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_candidate_feature_mean_corr_bounds() {
        let theta: Vec<f64> = vec![0.0, 0.1, 0.2, 0.3];
        let mut corr_sum = 0.0_f64;
        let mut n_pairs = 0usize;
        for i in 0..theta.len() {
            for j in (i + 1)..theta.len() {
                corr_sum += (theta[i] - theta[j]).cos();
                n_pairs += 1;
            }
        }
        let mean_corr = corr_sum / n_pairs as f64;
        assert!((-1.0..=1.0).contains(&mean_corr));
        assert!(mean_corr > 0.95);
    }
}
