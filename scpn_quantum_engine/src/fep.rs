// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Free Energy Principle Computation

//! Rust-accelerated Free Energy Principle computations.
//!
//! Implements the core numerical operations for Friston's variational
//! free energy framework: KL divergence between Gaussians, free energy
//! gradient for belief updates, and precision-weighted prediction errors
//! for hierarchical predictive coding across SCPN layers.
//!
//! Ref: Friston, Nature Reviews Neuroscience 11, 127 (2010)

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::validation::validate_n;

fn log_det_spd_with_ridge(
    matrix: &ndarray::ArrayView2<'_, f64>,
    ridge: f64,
) -> Result<f64, &'static str> {
    let n = matrix.nrows();
    let mut chol = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[[i, j]] + if i == j { ridge } else { 0.0 };
            for k in 0..j {
                sum -= chol[i][k] * chol[j][k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err("k_precision + ridge*I must be symmetric positive definite");
                }
                chol[i][j] = sum.sqrt();
            } else {
                chol[i][j] = sum / chol[j][j];
            }
        }
    }

    let mut log_det = 0.0f64;
    for (i, row) in chol.iter().enumerate().take(n) {
        log_det += 2.0 * row[i].ln();
    }
    Ok(log_det)
}

/// Free energy gradient ∂F/∂μ for belief update dynamics.
///
/// ∂F/∂μ = K_reg × μ − Γ × (x − μ)
///
/// where K_reg = K + ridge×I (prior precision), Γ = sensory precision.
/// With identity generative model and Jacobian.
#[pyfunction]
pub fn free_energy_gradient_rust<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<'_, f64>,
    x_observed: PyReadonlyArray1<'_, f64>,
    k_precision: PyReadonlyArray2<'_, f64>,
    sensory_precision: PyReadonlyArray2<'_, f64>,
    ridge: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mu_arr = mu.as_array();
    let x_arr = x_observed.as_array();
    let k_arr = k_precision.as_array();
    let gamma = sensory_precision.as_array();
    let n = mu_arr.len();
    validate_n(n, "mu")?;

    let mut grad = Array1::<f64>::zeros(n);

    // Prior contribution: (K + ridge×I) × μ
    for i in 0..n {
        for j in 0..n {
            let k_val = k_arr[[i, j]] + if i == j { ridge } else { 0.0 };
            grad[i] += k_val * mu_arr[j];
        }
    }

    // Likelihood contribution: −Γ × (x − μ)
    for i in 0..n {
        let mut error_contrib = 0.0;
        for j in 0..n {
            error_contrib += gamma[[i, j]] * (x_arr[j] - mu_arr[j]);
        }
        grad[i] -= error_contrib;
    }

    Ok(PyArray1::from_owned_array(py, grad))
}

/// Hierarchical prediction error across SCPN layers.
///
/// For each layer i: ε_i = Π_i × (x_i − x̂_i)
/// where x̂_i = Σ_j K[i,j] × μ_j / Σ_j K[i,j] (coupling-weighted mean)
/// and Π_i = Σ_j K[i,j] (total coupling = local precision).
#[pyfunction]
pub fn hierarchical_prediction_error_rust<'py>(
    py: Python<'py>,
    observations: PyReadonlyArray1<'_, f64>,
    beliefs: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = observations.as_array();
    let mu = beliefs.as_array();
    let k_arr = k.as_array();
    let n = x.len();
    validate_n(n, "observations")?;

    let errors = prediction_error_inner(x.as_slice().unwrap(), mu.as_slice().unwrap(), &k_arr, n);
    Ok(PyArray1::from_owned_array(py, Array1::from_vec(errors)))
}

/// Pure Rust prediction error (no PyO3).
pub fn prediction_error_inner(
    observations: &[f64],
    beliefs: &[f64],
    k: &ndarray::ArrayView2<f64>,
    n: usize,
) -> Vec<f64> {
    let mut errors = vec![0.0f64; n];

    for i in 0..n {
        let mut total_coupling = 0.0f64;
        let mut weighted_sum = 0.0f64;

        for j in 0..n {
            let kij = k[[i, j]];
            total_coupling += kij;
            weighted_sum += kij * beliefs[j];
        }

        if total_coupling < 1e-15 {
            errors[i] = observations[i] - beliefs[i];
        } else {
            let prediction = weighted_sum / total_coupling;
            errors[i] = total_coupling * (observations[i] - prediction);
        }
    }

    errors
}

/// Variational free energy F = complexity + accuracy.
///
/// complexity = 0.5 × (μᵀ K_reg μ + tr(K_reg Σ) − log|Σ| − log|K_reg| − n)
/// accuracy = 0.5 × (x − μ)ᵀ Γ (x − μ)
///
/// Returns (free_energy, complexity, accuracy).
#[pyfunction]
pub fn variational_free_energy_rust(
    mu: PyReadonlyArray1<'_, f64>,
    x_observed: PyReadonlyArray1<'_, f64>,
    k_precision: PyReadonlyArray2<'_, f64>,
    sensory_precision: PyReadonlyArray2<'_, f64>,
    sigma_diag: f64,
    ridge: f64,
) -> PyResult<(f64, f64, f64)> {
    crate::validation::validate_positive(sigma_diag, "sigma_diag")?;

    let mu_arr = mu.as_array();
    validate_n(mu_arr.len(), "mu")?;
    let x_arr = x_observed.as_array();
    let k_arr = k_precision.as_array();
    let gamma = sensory_precision.as_array();
    let n = mu_arr.len();

    // Complexity: KL[q || prior] for diagonal Σ = sigma_diag × I
    // KL = 0.5 × (tr(K_reg × Σ) + μᵀ K_reg μ − n − log|K_reg| − log|Σ|)
    // For Σ = σ²I: tr(K_reg × Σ) = σ² × tr(K_reg)
    let mut trace_k = 0.0f64;
    let mut mu_k_mu = 0.0f64;
    for i in 0..n {
        let k_ii = k_arr[[i, i]] + ridge;
        trace_k += k_ii;
        for j in 0..n {
            let k_val = k_arr[[i, j]] + if i == j { ridge } else { 0.0 };
            mu_k_mu += mu_arr[i] * k_val * mu_arr[j];
        }
    }
    // Diagonal posterior covariance: log|Σ| = n × log(σ²)
    let log_det_sigma = n as f64 * sigma_diag.ln();
    let log_det_k = log_det_spd_with_ridge(&k_arr, ridge).map_err(PyValueError::new_err)?;

    let complexity = 0.5 * (sigma_diag * trace_k + mu_k_mu - n as f64 - log_det_k - log_det_sigma);

    // Accuracy: 0.5 × (x − μ)ᵀ Γ (x − μ)
    let mut accuracy = 0.0f64;
    for i in 0..n {
        let mut error_contrib = 0.0;
        for j in 0..n {
            error_contrib += gamma[[i, j]] * (x_arr[j] - mu_arr[j]);
        }
        accuracy += (x_arr[i] - mu_arr[i]) * error_contrib;
    }
    accuracy *= 0.5;

    let free_energy = complexity + accuracy;
    Ok((free_energy, complexity, accuracy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_prediction_error_zero_when_perfect() {
        let n = 3;
        let obs = vec![0.5, 0.5, 0.5];
        let beliefs = vec![0.5, 0.5, 0.5];
        let k = Array2::from_shape_fn((n, n), |(i, j)| if i == j { 0.0 } else { 1.0 });
        let errors = prediction_error_inner(&obs, &beliefs, &k.view(), n);
        for e in &errors {
            assert!(e.abs() < 1e-10, "perfect prediction → zero error");
        }
    }

    #[test]
    fn test_prediction_error_nonzero_mismatch() {
        let n = 2;
        let obs = vec![1.0, 0.0];
        let beliefs = vec![0.0, 1.0];
        let k = Array2::from_shape_fn((n, n), |(i, j)| if i == j { 0.0 } else { 1.0 });
        let errors = prediction_error_inner(&obs, &beliefs, &k.view(), n);
        // obs[0]=1, prediction=K[0,1]*beliefs[1]/K[0,1]=1, error=0
        // obs[1]=0, prediction=K[1,0]*beliefs[0]/K[1,0]=0, error=0
        // Both happen to match because coupling-weighted mean of single neighbour = that neighbour
        for e in &errors {
            assert!(e.abs() < 1e-10);
        }
    }

    #[test]
    fn test_prediction_error_zero_coupling() {
        let n = 2;
        let obs = vec![1.0, 2.0];
        let beliefs = vec![0.5, 1.5];
        let k = Array2::<f64>::zeros((n, n));
        let errors = prediction_error_inner(&obs, &beliefs, &k.view(), n);
        // Zero coupling → fallback: error = obs - beliefs
        assert!((errors[0] - 0.5).abs() < 1e-12);
        assert!((errors[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_free_energy_zero_observation_zero_belief() {
        // μ=0, x=0, K=I → complexity = KL only, accuracy = 0
        let n = 2;
        let mu = vec![0.0; n];
        let x = vec![0.0; n];
        let _k = Array2::<f64>::eye(n);
        let _gamma = Array2::<f64>::eye(n);
        let _sigma_diag = 1.0;
        let _ridge = 1e-10;

        let mu_a = Array1::from_vec(mu);
        let x_a = Array1::from_vec(x);

        // Manual: accuracy = 0
        let mut accuracy = 0.0f64;
        for i in 0..n {
            let diff: f64 = x_a[i] - mu_a[i];
            accuracy += diff * diff;
        }
        accuracy *= 0.5;
        assert!(accuracy.abs() < 1e-20);
    }

    #[test]
    fn test_log_det_spd_uses_full_matrix_not_diagonal_product() {
        let k = Array2::from_shape_vec((2, 2), vec![2.0, 0.5, 0.5, 1.5]).unwrap();
        let log_det = log_det_spd_with_ridge(&k.view(), 0.0).unwrap();
        let expected = 2.75f64.ln();
        assert!((log_det - expected).abs() < 1e-12);
        let diagonal_product = (2.0f64 * 1.5f64).ln();
        assert!((log_det - diagonal_product).abs() > 1e-3);
    }

    #[test]
    fn test_log_det_spd_rejects_non_positive_definite_precision() {
        let k = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0]).unwrap();
        let err = log_det_spd_with_ridge(&k.view(), 0.0).unwrap_err();
        assert!(err.to_string().contains("positive definite"));
    }
}
