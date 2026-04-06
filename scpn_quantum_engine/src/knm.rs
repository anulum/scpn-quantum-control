// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — K_nm Coupling Matrix Construction

//! K_nm coupling matrix for the SCPN hierarchy.
//!
//! Builds the inter-layer coupling matrix from Paper 27 parameters:
//! K_nm = K_base × exp(-α × |n - m|) with calibration anchors from
//! Table 2 and cross-hierarchy boosts for L1↔L16 and L5↔L7.

use ndarray::Array2;
use numpy::PyArray2;
use pyo3::prelude::*;

/// Build K_nm coupling matrix from Paper 27 parameters.
/// K_nm = K_base × exp(-alpha × |n - m|) with calibration anchors.
#[pyfunction]
pub fn build_knm<'py>(
    py: Python<'py>,
    n: usize,
    k_base: f64,
    alpha: f64,
) -> Bound<'py, PyArray2<f64>> {
    let k = build_knm_inner(n, k_base, alpha);
    PyArray2::from_owned_array(py, k)
}

/// Pure Rust implementation (no PyO3) for testing and internal use.
pub fn build_knm_inner(n: usize, k_base: f64, alpha: f64) -> Array2<f64> {
    let mut k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            k[[i, j]] = k_base * (-alpha * (i as f64 - j as f64).abs()).exp();
        }
    }

    // Calibration anchors (Paper 27 Table 2)
    let anchors: [(usize, usize, f64); 4] = [
        (0, 1, 0.302),
        (1, 2, 0.201),
        (2, 3, 0.252),
        (3, 4, 0.154),
    ];
    for &(i, j, val) in &anchors {
        if i < n && j < n {
            k[[i, j]] = val;
            k[[j, i]] = val;
        }
    }

    // Cross-hierarchy boosts (max preserves exponential if already larger)
    if n > 15 {
        k[[0, 15]] = k[[0, 15]].max(0.05);
        k[[15, 0]] = k[[15, 0]].max(0.05);
    }
    if n > 6 {
        k[[4, 6]] = k[[4, 6]].max(0.15);
        k[[6, 4]] = k[[6, 4]].max(0.15);
    }

    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_knm_symmetric() {
        let n = 4;
        let k = build_knm_inner(n, 0.45, 0.3);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (k[[i, j]] - k[[j, i]]).abs() < 1e-12,
                    "K_nm must be symmetric: K[{i},{j}]={} != K[{j},{i}]={}",
                    k[[i, j]],
                    k[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_build_knm_anchors() {
        let n = 5;
        let k = build_knm_inner(n, 0.45, 0.3);
        assert!((k[[0, 1]] - 0.302).abs() < 1e-12, "anchor K[0,1]=0.302");
        assert!((k[[1, 2]] - 0.201).abs() < 1e-12, "anchor K[1,2]=0.201");
        assert!((k[[2, 3]] - 0.252).abs() < 1e-12, "anchor K[2,3]=0.252");
        assert!((k[[3, 4]] - 0.154).abs() < 1e-12, "anchor K[3,4]=0.154");
    }

    #[test]
    fn test_build_knm_exponential_decay() {
        let n = 8;
        let k = build_knm_inner(n, 0.45, 0.3);
        // Beyond anchored region, exponential decay holds
        assert!(k[[5, 6]] > k[[5, 7]]);
    }

    #[test]
    fn test_build_knm_cross_hierarchy() {
        let n = 16;
        let k = build_knm_inner(n, 0.45, 0.3);
        assert!(k[[0, 15]] >= 0.05, "L1↔L16 cross-hierarchy boost");
        assert!(k[[4, 6]] >= 0.15, "L5↔L7 cross-hierarchy boost");
    }

    #[test]
    fn test_build_knm_positive() {
        let n = 8;
        let k = build_knm_inner(n, 0.45, 0.3);
        for i in 0..n {
            for j in 0..n {
                assert!(k[[i, j]] >= 0.0, "K_nm must be non-negative");
            }
        }
    }
}
