// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — GUESS Symmetry Decay ZNE

//! Rust-accelerated GUESS (Guiding Extrapolations from Symmetry Decays).
//!
//! Batch GUESS correction: given arrays of noisy target values and
//! symmetry values, compute mitigated estimates in parallel.
//!
//! Ref: Oliva del Moral et al., arXiv:2603.13060 (2026)

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::validation::validate_positive;

/// Batch GUESS extrapolation: mitigate N observables in parallel.
///
/// ⟨O_i⟩_mit = ⟨O_i⟩_noisy × (|S_ideal / S_noisy_i|)^α
///
/// Returns array of mitigated values.
#[pyfunction]
pub fn guess_extrapolate_batch<'py>(
    py: Python<'py>,
    target_noisy: PyReadonlyArray1<'_, f64>,
    symmetry_noisy: PyReadonlyArray1<'_, f64>,
    s_ideal: f64,
    alpha: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_positive(s_ideal.abs(), "|s_ideal|")?;
    let targets = target_noisy.as_slice().unwrap();
    let sym = symmetry_noisy.as_slice().unwrap();

    let mitigated: Vec<f64> = targets
        .par_iter()
        .zip(sym.par_iter())
        .map(|(&o, &s)| {
            if s.abs() < 1e-15 {
                o // symmetry fully decayed — no correction
            } else {
                o * (s_ideal / s).abs().powf(alpha)
            }
        })
        .collect();

    Ok(PyArray1::from_vec(py, mitigated))
}

/// Fit exponential decay α from symmetry observable measurements.
///
/// Model: ⟨S⟩_g = S_ideal × exp(-α × (g - 1))
/// Fit via least-squares on log-transformed ratios.
///
/// Returns (alpha, fit_residual).
#[pyfunction]
pub fn fit_symmetry_decay(
    s_ideal: f64,
    noisy_values: PyReadonlyArray1<'_, f64>,
    noise_scales: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64)> {
    validate_positive(s_ideal.abs(), "|s_ideal|")?;
    let vals = noisy_values.as_slice().unwrap();
    let scales = noise_scales.as_slice().unwrap();
    let (alpha, residual) = fit_decay_inner(s_ideal, vals, scales);
    Ok((alpha, residual))
}

/// Pure Rust decay fitting.
pub fn fit_decay_inner(s_ideal: f64, vals: &[f64], scales: &[f64]) -> (f64, f64) {
    let n = vals.len();
    if n < 2 {
        return (0.0, 0.0);
    }

    // log(S_g / S_ideal) = -α × (g - 1)
    // Linear regression: y = -α × x where y = log(ratio), x = g-1
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    let mut count = 0usize;

    for i in 0..n {
        let ratio = (vals[i] / s_ideal).max(1e-15);
        let y = ratio.ln();
        let x = scales[i] - 1.0;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        count += 1;
    }

    let nf = count as f64;
    let denom = nf * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return (0.0, 0.0);
    }

    let slope = (nf * sum_xy - sum_x * sum_y) / denom;
    let alpha = -slope;

    // Residual
    let intercept = (sum_y - slope * sum_x) / nf;
    let mut rss = 0.0;
    for i in 0..n {
        let ratio = (vals[i] / s_ideal).max(1e-15);
        let y = ratio.ln();
        let x = scales[i] - 1.0;
        let pred = slope * x + intercept;
        rss += (y - pred) * (y - pred);
    }
    let residual = (rss / nf).sqrt();

    (alpha, residual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_decay_exact_exponential() {
        let s_ideal = 4.0;
        let alpha_true = 0.15;
        let scales: Vec<f64> = vec![1.0, 3.0, 5.0, 7.0];
        let vals: Vec<f64> = scales
            .iter()
            .map(|&g| s_ideal * (-alpha_true * (g - 1.0)).exp())
            .collect();
        let (alpha, residual) = fit_decay_inner(s_ideal, &vals, &scales);
        assert!(
            (alpha - alpha_true).abs() < 1e-10,
            "exact exponential: α={alpha}, expected {alpha_true}"
        );
        assert!(residual < 1e-10);
    }

    #[test]
    fn test_fit_decay_no_decay() {
        let (alpha, _) = fit_decay_inner(4.0, &[4.0, 4.0, 4.0], &[1.0, 3.0, 5.0]);
        assert!(alpha.abs() < 1e-10, "no decay → α ≈ 0");
    }

    #[test]
    fn test_fit_decay_single_point() {
        let (alpha, _) = fit_decay_inner(4.0, &[3.5], &[1.0]);
        assert!((alpha - 0.0).abs() < 1e-10, "single point → α = 0");
    }
}
