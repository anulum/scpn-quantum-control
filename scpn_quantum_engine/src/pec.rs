// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Probabilistic Error Cancellation

//! Probabilistic Error Cancellation (PEC) for single-qubit depolarising channel.
//!
//! Implements quasi-probability decomposition and parallel Monte Carlo sign-sampling
//! via rayon. The depolarising channel E(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ) is
//! inverted analytically to obtain quasi-probability coefficients, then sampled in
//! parallel to estimate the mitigated expectation value.

use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

/// PEC quasi-probability coefficients for single-qubit depolarising channel.
/// Returns [q_I, q_X, q_Y, q_Z].
#[pyfunction]
pub fn pec_coefficients(gate_error_rate: f64) -> [f64; 4] {
    let p = gate_error_rate;
    let denom = 4.0 - 4.0 * p;
    let q_i = 1.0 + 3.0 * p / denom;
    let q_xyz = -p / denom;
    [q_i, q_xyz, q_xyz, q_xyz]
}

/// PEC sign-sampling in parallel (rayon). Single-qubit depolarising model.
///
/// Returns (mitigated_value, overhead, sign_distribution).
/// `base_exp_z` is the noiseless <Z> expectation from the ideal circuit.
/// Each sample draws a Pauli correction per gate, accumulates the sign
/// product, and scales by gamma^n_gates. The sampled operator identity
/// affects the sign but not the base expectation — this is the single-qubit
/// approximation where all corrections act on one qubit.
#[pyfunction]
pub fn pec_sample_parallel(
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

    let cum_probs: Vec<f64> = probs
        .iter()
        .scan(0.0, |acc, &p| {
            *acc += p;
            Some(*acc)
        })
        .collect();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pec_coefficients_zero_error() {
        let [q_i, q_x, q_y, q_z] = pec_coefficients(0.0);
        assert!((q_i - 1.0).abs() < 1e-12);
        assert!(q_x.abs() < 1e-12);
        assert!(q_y.abs() < 1e-12);
        assert!(q_z.abs() < 1e-12);
    }

    #[test]
    fn test_pec_coefficients_sum() {
        let [q_i, q_x, q_y, q_z] = pec_coefficients(0.01);
        let s = q_i + q_x + q_y + q_z;
        assert!(
            (s - 1.0).abs() < 1e-10,
            "PEC coefficients should sum to 1, got {s}"
        );
    }

    #[test]
    fn test_pec_coefficients_high_error() {
        let [q_i, q_x, q_y, q_z] = pec_coefficients(0.5);
        let s = q_i + q_x + q_y + q_z;
        assert!(
            (s - 1.0).abs() < 1e-10,
            "PEC coefficients should sum to 1 even at high error"
        );
        // At p=0.5, q_xyz should be significantly negative
        assert!(q_x < 0.0);
        assert!(q_y < 0.0);
        assert!(q_z < 0.0);
    }

    #[test]
    fn test_pec_sample_parallel_deterministic() {
        let (v1, o1, _) = pec_sample_parallel(0.01, 5, 100, 0.9, 42);
        let (v2, o2, _) = pec_sample_parallel(0.01, 5, 100, 0.9, 42);
        assert!((v1 - v2).abs() < 1e-12, "same seed must give same result");
        assert!((o1 - o2).abs() < 1e-12);
    }

    #[test]
    fn test_pec_sample_parallel_zero_error() {
        let (val, overhead, _) = pec_sample_parallel(0.0, 5, 1000, 0.8, 42);
        assert!(
            (overhead - 1.0).abs() < 1e-12,
            "zero error → overhead = 1"
        );
        assert!(
            (val - 0.8).abs() < 1e-6,
            "zero error → mitigated ≈ base"
        );
    }
}
