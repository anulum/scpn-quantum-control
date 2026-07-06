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
use std::cmp::Ordering;
use std::f64::consts::PI;

use crate::validation::{
    validate_contiguous_slice, validate_finite, validate_flat_square, validate_n,
};

type AnalogCouplingTermsResult<'py> = PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)>;

type HybridCouplingPartitionResult<'py> = PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)>;

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
) -> AnalogCouplingTermsResult<'py> {
    validate_n(n, "n")?;
    let k = validate_contiguous_slice(&k_flat, "k_flat")?;
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

/// Split Kuramoto couplings into analog-native and digital-residual blocks.
///
/// The selector is deterministic: couplings with `|K_ij| > zero_threshold`
/// and `|K_ij| >= analog_threshold` are ranked by descending magnitude, then
/// by `(i, j)`. The first `analog_budget` couplings are routed to the analog
/// block; remaining non-zero couplings become the digital residual.
#[pyfunction]
pub fn hybrid_coupling_partition<'py>(
    py: Python<'py>,
    k_flat: PyReadonlyArray1<'_, f64>,
    n: usize,
    analog_budget: usize,
    analog_threshold: f64,
    zero_threshold: f64,
) -> HybridCouplingPartitionResult<'py> {
    validate_n(n, "n")?;
    let k = validate_contiguous_slice(&k_flat, "k_flat")?;
    validate_flat_square(k, n, "k_flat")?;
    validate_finite(k, "k_flat")?;
    if !analog_threshold.is_finite() || analog_threshold < 0.0 {
        return Err(PyValueError::new_err(format!(
            "analog_threshold must be finite and non-negative, got {analog_threshold}"
        )));
    }
    if !zero_threshold.is_finite() || zero_threshold < 0.0 {
        return Err(PyValueError::new_err(format!(
            "zero_threshold must be finite and non-negative, got {zero_threshold}"
        )));
    }

    let mut candidates: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let magnitude = k[i * n + j].abs();
            if magnitude > zero_threshold && magnitude >= analog_threshold {
                candidates.push((i, j, magnitude));
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .2
            .partial_cmp(&left.2)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
    });

    let mut selected = vec![false; n * n];
    for (i, j, _) in candidates.into_iter().take(analog_budget) {
        selected[i * n + j] = true;
        selected[j * n + i] = true;
    }

    let mut analog = vec![0.0; n * n];
    let mut digital = vec![0.0; n * n];
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut route_codes = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let kij = k[i * n + j];
            if kij.abs() <= zero_threshold {
                continue;
            }
            rows.push(i as i64);
            cols.push(j as i64);
            if selected[i * n + j] {
                analog[i * n + j] = kij;
                analog[j * n + i] = kij;
                route_codes.push(1);
            } else {
                digital[i * n + j] = kij;
                digital[j * n + i] = kij;
                route_codes.push(0);
            }
        }
    }

    Ok((
        PyArray1::from_vec(py, analog),
        PyArray1::from_vec(py, digital),
        PyArray1::from_vec(py, rows),
        PyArray1::from_vec(py, cols),
        PyArray1::from_vec(py, route_codes),
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

    #[test]
    fn hybrid_selection_prefers_largest_magnitude_then_indices() {
        let mut edges = [(0_usize, 2_usize, 0.75_f64), (0, 1, 0.75), (1, 2, 0.5)];
        edges.sort_by(|left, right| {
            right
                .2
                .partial_cmp(&left.2)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
                .then_with(|| left.1.cmp(&right.1))
        });
        assert_eq!(edges[0], (0, 1, 0.75));
        assert_eq!(edges[1], (0, 2, 0.75));
    }
}
