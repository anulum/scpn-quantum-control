// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Sparse Pauli Expectation Values

//! Bitwise Pauli expectation values from statevectors.
//!
//! Computes single-qubit ⟨X⟩, ⟨Y⟩, ⟨Z⟩ and the quantum order parameter
//! R = (1/N)|Σ_i (⟨X_i⟩ + i⟨Y_i⟩)| using O(N×2^n) bit-flip operations
//! instead of constructing dense 2^n × 2^n Pauli matrices.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute quantum order parameter R from a complex statevector using
/// sparse bitwise Pauli application.
///
/// For each qubit q, computes ⟨ψ|X_q|ψ⟩ and ⟨ψ|Y_q|ψ⟩ via index
/// bit-flips (ψ[k ⊕ (1<<q)]) instead of building dense 2^n Pauli matrices.
/// O(n_osc × 2^n) time, O(2^n) memory.
#[pyfunction]
pub fn state_order_param_sparse(
    psi_re: PyReadonlyArray1<'_, f64>,
    psi_im: PyReadonlyArray1<'_, f64>,
    n_osc: usize,
) -> f64 {
    let re = psi_re.as_slice().unwrap();
    let im = psi_im.as_slice().unwrap();
    state_order_param_inner(re, im, n_osc)
}

/// Pure Rust order parameter from statevector (no PyO3).
pub fn state_order_param_inner(re: &[f64], im: &[f64], n_osc: usize) -> f64 {
    let dim = re.len();
    let mut z_re = 0.0_f64;
    let mut z_im = 0.0_f64;

    for q in 0..n_osc {
        let mask: usize = 1 << q;
        let mut exp_x = 0.0_f64;
        let mut exp_y = 0.0_f64;

        for k in 0..dim {
            let flipped = k ^ mask;
            let prod_re = re[k] * re[flipped] + im[k] * im[flipped];

            exp_x += prod_re;

            let bit = ((k >> q) & 1) as f64;
            let sign = 1.0 - 2.0 * bit;
            let prod_im = re[k] * im[flipped] - im[k] * re[flipped];
            exp_y += sign * (-prod_im);
        }

        z_re += exp_x;
        z_im += exp_y;
    }

    z_re /= n_osc as f64;
    z_im /= n_osc as f64;
    (z_re * z_re + z_im * z_im).sqrt()
}

/// Compute single-qubit Pauli expectation ⟨ψ|P_qubit|ψ⟩ using bitwise ops.
///
/// pauli: 0=X, 1=Y, 2=Z
#[pyfunction]
pub fn expectation_pauli_fast(
    psi_re: PyReadonlyArray1<'_, f64>,
    psi_im: PyReadonlyArray1<'_, f64>,
    _n: usize,
    qubit: usize,
    pauli: usize,
) -> f64 {
    let re = psi_re.as_slice().unwrap();
    let im = psi_im.as_slice().unwrap();
    let dim = re.len();

    match pauli {
        0 => {
            let mask = 1usize << qubit;
            let mut result = 0.0;
            for k in 0..dim {
                let f = k ^ mask;
                result += re[k] * re[f] + im[k] * im[f];
            }
            result
        }
        1 => {
            let mask = 1usize << qubit;
            let mut result = 0.0;
            for k in 0..dim {
                let f = k ^ mask;
                let bit = ((k >> qubit) & 1) as f64;
                let sign = 2.0 * bit - 1.0;
                let prod_im = re[k] * im[f] - im[k] * re[f];
                result += sign * (-prod_im);
            }
            result
        }
        _ => {
            let mut result = 0.0;
            for k in 0..dim {
                let bit = ((k >> qubit) & 1) as f64;
                let sign = 1.0 - 2.0 * bit;
                result += sign * (re[k] * re[k] + im[k] * im[k]);
            }
            result
        }
    }
}

/// Batch compute per-qubit X and Y expectations for all n qubits in one call.
///
/// Returns (exp_x[n], exp_y[n]). Avoids 2n FFI roundtrips vs calling
/// expectation_pauli_fast individually per qubit per Pauli.
#[pyfunction]
pub fn all_xy_expectations<'py>(
    py: Python<'py>,
    psi_re: PyReadonlyArray1<'_, f64>,
    psi_im: PyReadonlyArray1<'_, f64>,
    n_osc: usize,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let re = psi_re.as_slice().unwrap();
    let im = psi_im.as_slice().unwrap();
    let dim = re.len();

    let mut exp_x = vec![0.0f64; n_osc];
    let mut exp_y = vec![0.0f64; n_osc];

    for q in 0..n_osc {
        let mask = 1usize << q;
        let mut ex = 0.0;
        let mut ey = 0.0;

        for k in 0..dim {
            let f = k ^ mask;
            ex += re[k] * re[f] + im[k] * im[f];

            let bit = ((k >> q) & 1) as f64;
            let sign = 2.0 * bit - 1.0;
            let prod_im = re[k] * im[f] - im[k] * re[f];
            ey += sign * (-prod_im);
        }

        exp_x[q] = ex;
        exp_y[q] = ey;
    }

    (
        PyArray1::from_vec(py, exp_x),
        PyArray1::from_vec(py, exp_y),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_order_param_product_state() {
        // |00⟩ = [1, 0, 0, 0]: ⟨X⟩ = 0, ⟨Y⟩ = 0, R = 0 for product state
        // Actually ⟨Z_i⟩ = 1 for all i, but R uses X,Y only
        let re = vec![1.0, 0.0, 0.0, 0.0];
        let im = vec![0.0; 4];
        let r = state_order_param_inner(&re, &im, 2);
        // |00⟩ has ⟨X_i⟩ = 0 for all i → R = 0
        assert!(r < 1e-10, "|00⟩ → R = 0, got {r}");
    }

    #[test]
    fn test_state_order_param_plus_state() {
        // |+⟩ = [1/√2, 1/√2] for 1 qubit: ⟨X⟩ = 1 → R = 1
        let s = 1.0 / 2.0_f64.sqrt();
        let re = vec![s, s];
        let im = vec![0.0; 2];
        let r = state_order_param_inner(&re, &im, 1);
        assert!((r - 1.0).abs() < 1e-10, "|+⟩ → R = 1, got {r}");
    }

    #[test]
    fn test_expectation_z_ground() {
        // |0⟩ = [1, 0]: ⟨Z⟩ = 1
        let re = vec![1.0, 0.0];
        let im = vec![0.0; 2];
        // Z on qubit 0, pauli=2
        let z_exp = expectation_pauli_fast_inner(&re, &im, 0, 2);
        assert!((z_exp - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_expectation_z_excited() {
        // |1⟩ = [0, 1]: ⟨Z⟩ = −1
        let re = vec![0.0, 1.0];
        let im = vec![0.0; 2];
        let z_exp = expectation_pauli_fast_inner(&re, &im, 0, 2);
        assert!((z_exp - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_expectation_x_plus() {
        // |+⟩ = [1/√2, 1/√2]: ⟨X⟩ = 1
        let s = 1.0 / 2.0_f64.sqrt();
        let re = vec![s, s];
        let im = vec![0.0; 2];
        let x_exp = expectation_pauli_fast_inner(&re, &im, 0, 0);
        assert!((x_exp - 1.0).abs() < 1e-10);
    }

    /// Helper: pure Rust single-qubit Pauli expectation.
    fn expectation_pauli_fast_inner(re: &[f64], im: &[f64], qubit: usize, pauli: usize) -> f64 {
        let dim = re.len();
        match pauli {
            0 => {
                let mask = 1usize << qubit;
                let mut result = 0.0;
                for k in 0..dim {
                    let f = k ^ mask;
                    result += re[k] * re[f] + im[k] * im[f];
                }
                result
            }
            1 => {
                let mask = 1usize << qubit;
                let mut result = 0.0;
                for k in 0..dim {
                    let f = k ^ mask;
                    let bit = ((k >> qubit) & 1) as f64;
                    let sign = 2.0 * bit - 1.0;
                    let prod_im = re[k] * im[f] - im[k] * re[f];
                    result += sign * (-prod_im);
                }
                result
            }
            _ => {
                let mut result = 0.0;
                for k in 0..dim {
                    let bit = ((k >> qubit) & 1) as f64;
                    let sign = 1.0 - 2.0 * bit;
                    result += sign * (re[k] * re[k] + im[k] * im[k]);
                }
                result
            }
        }
    }
}
