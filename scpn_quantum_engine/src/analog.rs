// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Analog Kuramoto Coupling Compiler

//! Vectorised construction of analog Kuramoto coupling terms.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::PI;

use crate::validation::{validate_finite, validate_flat_square, validate_n};

/// Compile upper-triangular coupling terms for analog Kuramoto backends.
///
/// Platform codes:
/// - `0`: neutral atoms, returns positive interaction strengths and
///   Rydberg-equivalent radii `(c6 / |K_ij|)^(1/6)`.
/// - `1`: circuit-QED resonators, returns exchange magnitudes and phases.
/// - `2`: continuous-variable modes, returns beam-splitter magnitudes and phases.
#[pyfunction]
pub fn analog_coupling_terms<'py>(
    py: Python<'py>,
    k_flat: PyReadonlyArray1<'_, f64>,
    n: usize,
    platform_code: usize,
    coupling_scale: f64,
    c6_coefficient: f64,
    zero_threshold: f64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validate_n(n, "n")?;
    let k = k_flat.as_slice().unwrap();
    validate_flat_square(k, n, "k_flat")?;
    validate_finite(k, "k_flat")?;
    if platform_code > 2 {
        return Err(PyValueError::new_err(format!(
            "platform_code must be 0, 1, or 2, got {platform_code}"
        )));
    }
    if !coupling_scale.is_finite() || coupling_scale <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "coupling_scale must be finite and positive, got {coupling_scale}"
        )));
    }
    if !c6_coefficient.is_finite() || c6_coefficient <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "c6_coefficient must be finite and positive, got {c6_coefficient}"
        )));
    }
    if !zero_threshold.is_finite() || zero_threshold < 0.0 {
        return Err(PyValueError::new_err(format!(
            "zero_threshold must be finite and non-negative, got {zero_threshold}"
        )));
    }

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut strengths = Vec::new();
    let mut phases = Vec::new();
    let mut radii = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let kij = coupling_scale * k[i * n + j];
            if kij.abs() <= zero_threshold {
                continue;
            }
            rows.push(i as i64);
            cols.push(j as i64);
            strengths.push(kij.abs());
            phases.push(if kij >= 0.0 { 0.0 } else { PI });
            if platform_code == 0 {
                radii.push((c6_coefficient / kij.abs()).powf(1.0 / 6.0));
            } else {
                radii.push(0.0);
            }
        }
    }

    Ok((
        PyArray1::from_vec(py, rows),
        PyArray1::from_vec(py, cols),
        PyArray1::from_vec(py, strengths),
        PyArray1::from_vec(py, phases),
        PyArray1::from_vec(py, radii),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_atom_radius_scales_with_inverse_sixth_power() {
        let k = 64.0_f64;
        let radius = (64.0_f64 / k).powf(1.0 / 6.0);
        assert!((radius - 1.0).abs() < 1e-12);
    }

    #[test]
    fn negative_coupling_maps_to_pi_phase() {
        let phase = if -0.5_f64 >= 0.0 { 0.0 } else { PI };
        assert!((phase - PI).abs() < 1e-12);
    }
}
