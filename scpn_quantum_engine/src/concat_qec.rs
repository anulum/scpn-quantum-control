// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Concatenated QEC Threshold Computation

//! Rust-accelerated concatenated QEC logical error rate computation.
//!
//! Computes the iterative logical error rate across concatenation levels:
//! p_L(k) = A × (p_L(k−1) / p_th)^((d_k + 1) / 2)
//!
//! Also computes average K_nm coupling between SCPN domains for
//! syndrome flow analysis.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::validation::{validate_domain_range, validate_positive, validate_range};

/// Surface code logical error rate at given distance and physical rate.
/// p_L = A × (p_phys / p_th)^((d+1)/2)
fn logical_error_rate(
    code_distance: usize,
    p_physical: f64,
    p_threshold: f64,
    prefactor: f64,
) -> f64 {
    let ratio = p_physical / p_threshold;
    if ratio >= 1.0 {
        return 1.0;
    }
    let exponent = (code_distance as f64 + 1.0) / 2.0;
    prefactor * ratio.powf(exponent)
}

/// Concatenated logical error rate across multiple levels.
///
/// Returns array of p_L values, one per level.
/// Level 0: p_L(0) = A × (p_phys / p_th)^((d_0+1)/2)
/// Level k: p_L(k) = A × (p_L(k-1) / p_th)^((d_k+1)/2)
#[pyfunction]
pub fn concatenated_logical_rate_rust<'py>(
    py: Python<'py>,
    p_physical: f64,
    distances: PyReadonlyArray1<'_, i64>,
    p_threshold: f64,
    prefactor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_range(p_physical, 0.0, 1.0, "p_physical")?;
    validate_positive(p_threshold, "p_threshold")?;
    validate_positive(prefactor, "prefactor")?;
    let dists = distances.as_slice().unwrap();
    let rates = concatenated_rates_inner(p_physical, dists, p_threshold, prefactor);
    Ok(PyArray1::from_vec(py, rates))
}

/// Pure Rust implementation (no PyO3).
pub fn concatenated_rates_inner(
    p_physical: f64,
    distances: &[i64],
    p_threshold: f64,
    prefactor: f64,
) -> Vec<f64> {
    let mut rates = Vec::with_capacity(distances.len());
    let mut p_current = p_physical;
    for &d in distances {
        let p_logical = logical_error_rate(d as usize, p_current, p_threshold, prefactor);
        rates.push(p_logical);
        p_current = p_logical;
    }
    rates
}

/// Average K_nm coupling between two SCPN domain ranges.
///
/// Computes mean coupling across all (i,j) pairs where i ∈ [a_start, a_end]
/// and j ∈ [b_start, b_end], excluding diagonal.
#[pyfunction]
pub fn knm_domain_coupling(
    k: PyReadonlyArray2<'_, f64>,
    a_start: usize,
    a_end: usize,
    b_start: usize,
    b_end: usize,
) -> PyResult<f64> {
    let k_arr = k.as_array();
    let n = k_arr.nrows();
    validate_domain_range(a_start, a_end, n, "domain_a")?;
    validate_domain_range(b_start, b_end, n, "domain_b")?;

    let mut total = 0.0f64;
    let mut count = 0usize;

    for i in a_start..=a_end {
        for j in b_start..=b_end {
            if i != j && i < n && j < n {
                total += k_arr[[i, j]];
                count += 1;
            }
        }
    }

    Ok(if count > 0 {
        total / count as f64
    } else {
        0.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logical_error_rate_below_threshold() {
        let r = logical_error_rate(3, 0.003, 0.01, 0.1);
        // (0.003/0.01)^2 = 0.3^2 = 0.09, × 0.1 = 0.009
        assert!((r - 0.009).abs() < 1e-10);
    }

    #[test]
    fn test_logical_error_rate_above_threshold() {
        let r = logical_error_rate(3, 0.05, 0.01, 0.1);
        assert!((r - 1.0).abs() < 1e-12, "above threshold → 1.0");
    }

    #[test]
    fn test_concatenated_rates_monotonic() {
        let rates = concatenated_rates_inner(0.001, &[5, 5, 5], 0.01, 0.1);
        assert_eq!(rates.len(), 3);
        for i in 0..rates.len() - 1 {
            assert!(
                rates[i + 1] < rates[i],
                "below threshold: each level must reduce error"
            );
        }
    }

    #[test]
    fn test_concatenated_rates_empty() {
        let rates = concatenated_rates_inner(0.003, &[], 0.01, 0.1);
        assert!(rates.is_empty());
    }

    #[test]
    fn test_concatenated_rates_single() {
        let rates = concatenated_rates_inner(0.003, &[5], 0.01, 0.1);
        assert_eq!(rates.len(), 1);
        let expected = logical_error_rate(5, 0.003, 0.01, 0.1);
        assert!((rates[0] - expected).abs() < 1e-15);
    }

    #[test]
    fn test_concatenated_double_exponential() {
        let rates = concatenated_rates_inner(0.0001, &[7, 7, 7], 0.01, 0.1);
        // Very low p → each level squares (approximately) the log
        assert!(rates[2] < 1e-30, "three levels of d=7 at p=0.0001 → tiny rate");
    }

    #[test]
    fn test_d1_no_correction() {
        // d=1: exponent = 1, p_L = A × (p/p_th)^1
        let r = logical_error_rate(1, 0.003, 0.01, 0.1);
        let expected = 0.1 * 0.003 / 0.01;
        assert!((r - expected).abs() < 1e-12);
    }
}
