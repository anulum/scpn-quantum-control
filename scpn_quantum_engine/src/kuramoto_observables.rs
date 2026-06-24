// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Kuramoto synchronisation observables and their derivatives
//! Order parameter, mean phase and Daido higher-order observables (magnitude and phase)
//! with their analytic gradients and Hessians.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::kuramoto_common::validate_phase_vector;

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

/// Compute the m-th Fourier-mode phase ψ_m = atan2(⟨sin mθ⟩, ⟨cos mθ⟩).
///
/// For m = 1 this is the Kuramoto mean phase. An empty input and the incoherent mode both
/// report 0.0.
#[pyfunction]
pub fn daido_mode_phase(theta: PyReadonlyArray1<'_, f64>, m: i64) -> PyResult<f64> {
    if m < 1 {
        return Err(PyValueError::new_err(format!(
            "harmonic order m must be a positive integer, got {m}"
        )));
    }
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(daido_mode_phase_inner(theta, m as f64))
}

/// Pure Rust m-th Fourier-mode phase (no PyO3).
pub fn daido_mode_phase_inner(theta: &[f64], m: f64) -> f64 {
    if theta.is_empty() {
        return 0.0;
    }
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &t in theta {
        re += (m * t).cos();
        im += (m * t).sin();
    }
    im.atan2(re)
}

/// Compute the gradient ∂ψ_m/∂θ_j = (m / (N r_m²)) (C_m cos mθ_j + S_m sin mθ_j) of the
/// m-th Fourier-mode phase.
///
/// The components sum to m; the incoherent mode r_m = 0 returns the zero subgradient.
#[pyfunction]
pub fn daido_mode_phase_gradient<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    m: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if m < 1 {
        return Err(PyValueError::new_err(format!(
            "harmonic order m must be a positive integer, got {m}"
        )));
    }
    let theta = validate_phase_vector(&theta, "theta")?;
    let gradient = Array1::from_vec(daido_mode_phase_gradient_inner(theta, m as f64));
    Ok(PyArray1::from_owned_array(py, gradient))
}

/// Pure Rust m-th Fourier-mode phase gradient (no PyO3).
pub fn daido_mode_phase_gradient_inner(theta: &[f64], m: f64) -> Vec<f64> {
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
    let magnitude_squared = cos_mean * cos_mean + sin_mean * sin_mean;
    if magnitude_squared == 0.0 {
        return out;
    }
    let scale = m / (count * magnitude_squared);
    for (slot, &t) in out.iter_mut().zip(theta.iter()) {
        *slot = scale * (cos_mean * (m * t).cos() + sin_mean * (m * t).sin());
    }
    out
}

/// Compute the Hessian ∂²ψ_m/∂θ_i∂θ_j of the m-th Fourier-mode phase.
///
/// H_ij = m² [δ_ij s_j/(N r_m) − (s_i c_j + c_i s_j)/(N² r_m²)] with s_k = sin(ψ_m − m θ_k),
/// c_k = cos(ψ_m − m θ_k). The matrix is symmetric and every row sums to zero; for m = 1 it
/// is the mean-phase Hessian.
#[pyfunction]
pub fn daido_mode_phase_hessian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    m: i64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if m < 1 {
        return Err(PyValueError::new_err(format!(
            "harmonic order m must be a positive integer, got {m}"
        )));
    }
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(PyArray2::from_owned_array(
        py,
        daido_mode_phase_hessian_inner(theta, m as f64),
    ))
}

/// Pure Rust m-th Fourier-mode phase Hessian (no PyO3), returned row-major.
pub fn daido_mode_phase_hessian_inner(theta: &[f64], m: f64) -> Array2<f64> {
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
    let sin_aligned: Vec<f64> = theta
        .iter()
        .map(|&t| (sin_mean * (m * t).cos() - cos_mean * (m * t).sin()) / magnitude)
        .collect();
    let cos_aligned: Vec<f64> = theta
        .iter()
        .map(|&t| (cos_mean * (m * t).cos() + sin_mean * (m * t).sin()) / magnitude)
        .collect();
    let m_squared = m * m;
    let diagonal_scale = m_squared / (count * magnitude);
    let off_scale = m_squared / (count * count * magnitude * magnitude);
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] =
                -off_scale * (sin_aligned[i] * cos_aligned[j] + cos_aligned[i] * sin_aligned[j]);
        }
        out[[i, i]] += diagonal_scale * sin_aligned[i];
    }
    out
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
        assert!(
            hessian[[0, 0]].abs() < 1e-12,
            "single oscillator → ∂²R/∂θ² = 0"
        );
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
        assert!(
            (mean_phase_inner(&[2.7]) - 2.7).abs() < 1e-12,
            "ψ of one oscillator is its phase"
        );
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
        let total: f64 = mean_phase_gradient_inner(&[0.1, 0.9, 2.3, 3.1, 5.5])
            .iter()
            .sum();
        assert!((total - 1.0).abs() < 1e-12, "sum = {total}");
    }

    #[test]
    fn test_mean_phase_gradient_single_is_one() {
        let grad = mean_phase_gradient_inner(&[2.7]);
        assert_eq!(grad.len(), 1);
        assert!(
            (grad[0] - 1.0).abs() < 1e-12,
            "∂ψ/∂θ = 1 for one oscillator"
        );
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
        assert!(
            hessian[[0, 0]].abs() < 1e-12,
            "∂²ψ/∂θ² = 0 for one oscillator"
        );
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
        assert_eq!(
            daido_order_parameter_hessian_inner(&[], 2.0).shape(),
            &[0, 0]
        );
    }

    #[test]
    fn test_daido_mode_phase_m1_matches_mean_phase() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        assert!((daido_mode_phase_inner(&theta, 1.0) - mean_phase_inner(&theta)).abs() < 1e-12);
    }

    #[test]
    fn test_daido_mode_phase_gradient_matches_finite_difference() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let m = 2.0;
        let gradient = daido_mode_phase_gradient_inner(&theta, m);
        let h = 1e-6;
        for j in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[j] += h;
            minus[j] -= h;
            let mut delta = daido_mode_phase_inner(&plus, m) - daido_mode_phase_inner(&minus, m);
            // unwrap the atan2 branch cut
            delta = (delta + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
                - std::f64::consts::PI;
            let fd = delta / (2.0 * h);
            assert!((gradient[j] - fd).abs() < 1e-6, "grad[{j}]");
        }
    }

    #[test]
    fn test_daido_mode_phase_gradient_sums_to_m() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5, 6.0];
        for m in [1.0, 2.0, 3.0] {
            let sum: f64 = daido_mode_phase_gradient_inner(&theta, m).iter().sum();
            assert!((sum - m).abs() < 1e-12, "m={m} sum={sum}");
        }
    }

    #[test]
    fn test_daido_mode_phase_empty() {
        assert_eq!(daido_mode_phase_inner(&[], 2.0), 0.0);
        assert!(daido_mode_phase_gradient_inner(&[], 2.0).is_empty());
    }

    #[test]
    fn test_daido_mode_phase_hessian_m1_matches_mean_phase_hessian() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let daido = daido_mode_phase_hessian_inner(&theta, 1.0);
        let mean = mean_phase_hessian_inner(&theta);
        for i in 0..theta.len() {
            for j in 0..theta.len() {
                assert!((daido[[i, j]] - mean[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_daido_mode_phase_hessian_matches_finite_difference_of_gradient() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let m = 2.0;
        let hessian = daido_mode_phase_hessian_inner(&theta, m);
        let h = 1e-6;
        for l in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[l] += h;
            minus[l] -= h;
            let grad_plus = daido_mode_phase_gradient_inner(&plus, m);
            let grad_minus = daido_mode_phase_gradient_inner(&minus, m);
            for i in 0..theta.len() {
                let fd = (grad_plus[i] - grad_minus[i]) / (2.0 * h);
                assert!((hessian[[i, l]] - fd).abs() < 1e-6, "H[{i},{l}]");
            }
        }
    }

    #[test]
    fn test_daido_mode_phase_hessian_symmetric_and_rows_sum_to_zero() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5, 6.0];
        let hessian = daido_mode_phase_hessian_inner(&theta, 3.0);
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
    fn test_daido_mode_phase_hessian_empty() {
        assert_eq!(daido_mode_phase_hessian_inner(&[], 2.0).shape(), &[0, 0]);
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
}
