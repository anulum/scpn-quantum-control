// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Koopman Generator

//! Dense Koopman generator for the finite Kuramoto observable lift.
//!
//! The observable ordering matches `analysis/koopman.py`:
//! theta coordinates, pairwise cosines, then pairwise sines.

use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_contiguous_slice<'a>(values: &'a [f64], name: &str) -> PyResult<&'a [f64]> {
    validate_finite_slice(values, name)?;
    Ok(values)
}

fn validate_finite_slice(values: &[f64], name: &str) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

/// Build the dense Koopman generator for a Kuramoto coupling matrix.
#[pyfunction]
pub fn koopman_generator<'py>(
    py: Python<'py>,
    k: PyReadonlyArray2<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    theta_ref: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let k_arr = k.as_array();
    let omega_arr = omega.as_array();
    let theta_arr = theta_ref.as_array();
    if k_arr.ndim() != 2 || k_arr.shape()[0] != k_arr.shape()[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "k must be a square 2-D matrix, got shape {:?}",
            k_arr.shape()
        )));
    }
    let n = k_arr.shape()[0];
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "k must have at least one oscillator",
        ));
    }
    if omega_arr.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "omega must have length {n}, got {}",
            omega_arr.len()
        )));
    }
    if theta_arr.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "theta_ref must have length {n}, got {}",
            theta_arr.len()
        )));
    }
    let k_slice = k_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("k must be a C-contiguous NumPy array"))?;
    let omega_slice = omega_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("omega must be a C-contiguous NumPy array"))?;
    let theta_slice = theta_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("theta_ref must be a C-contiguous NumPy array"))?;
    let k_slice = validate_contiguous_slice(k_slice, "k")?;
    let omega_slice = validate_contiguous_slice(omega_slice, "omega")?;
    let theta_slice = validate_contiguous_slice(theta_slice, "theta_ref")?;

    let generator = koopman_generator_inner(k_slice, omega_slice, theta_slice, n);
    Ok(PyArray2::from_owned_array(py, generator))
}

/// Pure Rust dense Koopman generator using the Python-compatible convention.
pub fn koopman_generator_inner(
    k: &[f64],
    omega: &[f64],
    theta_ref: &[f64],
    n: usize,
) -> Array2<f64> {
    let n_pairs = n * (n - 1) / 2;
    let dim = n + 2 * n_pairs;
    let mut generator = Array2::<f64>::zeros((dim, dim));
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }

    for i in 0..n {
        for (pair_index, (a, b)) in pairs.iter().enumerate() {
            let sin_index = n + n_pairs + pair_index;
            if *b == i {
                generator[[i, sin_index]] += k[*a * n + i];
            } else if *a == i {
                generator[[i, sin_index]] -= k[*b * n + i];
            }
        }
    }

    for (pair_index, (a, b)) in pairs.iter().enumerate() {
        let delta_omega = omega[*b] - omega[*a];
        let cos_index = n + pair_index;
        let sin_index = n + n_pairs + pair_index;
        generator[[cos_index, sin_index]] = -delta_omega;
        generator[[sin_index, cos_index]] = delta_omega;
    }

    for (pair_index, (a, b)) in pairs.iter().enumerate() {
        let cos_index = n + pair_index;
        let sin_index = n + n_pairs + pair_index;
        let delta = theta_ref[*b] - theta_ref[*a];
        let sin_delta = delta.sin();

        for m in 0..n {
            if m == *a || m == *b {
                continue;
            }
            let coupling_a = k[m * n + *a];
            let coupling_b = k[m * n + *b];
            generator[[cos_index, cos_index]] += -(coupling_b - coupling_a) * sin_delta * 0.5;
            generator[[sin_index, sin_index]] += (coupling_b - coupling_a) * sin_delta * 0.5;
        }
    }

    generator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_oscillator_generator_matches_reference_entries() {
        let k = [0.0, 0.5, 0.5, 0.0];
        let omega = [1.0, 1.5];
        let theta_ref = [0.1, -0.2];
        let generator = koopman_generator_inner(&k, &omega, &theta_ref, 2);

        assert_eq!(generator.shape(), &[4, 4]);
        assert!((generator[[0, 3]] + 0.5).abs() < 1e-12);
        assert!((generator[[1, 3]] - 0.5).abs() < 1e-12);
        assert!((generator[[2, 3]] + 0.5).abs() < 1e-12);
        assert!((generator[[3, 2]] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_three_oscillator_generator_is_finite() {
        let k = [0.0, 0.5, 0.2, 0.5, 0.0, 0.3, 0.2, 0.3, 0.0];
        let omega = [1.0, 1.5, 2.0];
        let theta_ref = [0.1, -0.2, 0.3];
        let generator = koopman_generator_inner(&k, &omega, &theta_ref, 3);

        assert_eq!(generator.shape(), &[9, 9]);
        assert!(generator.iter().all(|value| value.is_finite()));
    }
}
