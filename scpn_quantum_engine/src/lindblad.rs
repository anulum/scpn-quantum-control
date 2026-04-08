// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Lindblad Jump Operators

//! Lindblad jump operator construction for open quantum system simulation.
//!
//! Builds jump operators L_k for excitation transfer between coupled oscillators:
//! L_k: |…0_i…1_j…⟩ ← |…1_i…0_j…⟩ (transfers excitation from i to j).
//! Output is COO sparse format for scipy.sparse construction.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::validation::validate_n;

/// Lindblad jump operator COO data + anti-Hermitian diagonal.
///
/// Builds all jump operators L_k for pairs (i,j) where |K[i,j]| > threshold.
/// Returns (rows, cols, op_starts, n_ops) where op_starts[k] is the first
/// index in rows/cols belonging to operator k. op_starts has length n_ops+1.
#[allow(clippy::type_complexity)]
#[pyfunction]
pub fn lindblad_jump_ops_coo<'py>(
    py: Python<'py>,
    k_flat: PyReadonlyArray1<'_, f64>,
    n: usize,
    threshold: f64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    usize,
)> {
    validate_n(n, "n")?;
    if n > 20 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n={n} too large for jump operators (max 20)"
        )));
    }
    crate::validation::validate_flat_square(k_flat.as_slice().unwrap(), n, "k_flat")?;
    let k = k_flat.as_slice().unwrap();
    let dim = 1usize << n;

    let mut rows: Vec<i64> = Vec::new();
    let mut cols: Vec<i64> = Vec::new();
    let mut op_starts: Vec<i64> = Vec::new();
    let mut op_count = 0usize;

    for i in 0..n {
        for j in 0..n {
            if i != j && k[i * n + j].abs() > threshold {
                op_starts.push(rows.len() as i64);
                for idx in 0..dim {
                    if ((idx >> i) & 1) == 1 && ((idx >> j) & 1) == 0 {
                        let flipped = idx ^ ((1 << i) | (1 << j));
                        rows.push(flipped as i64);
                        cols.push(idx as i64);
                    }
                }
                op_count += 1;
            }
        }
    }
    // Sentinel: marks end of last operator
    op_starts.push(rows.len() as i64);

    Ok((
        PyArray1::from_vec(py, rows),
        PyArray1::from_vec(py, cols),
        PyArray1::from_vec(py, op_starts),
        op_count,
    ))
}

/// Anti-Hermitian diagonal for Lindblad trajectory path.
///
/// diag[idx] = number of active jump channels that can fire from state |idx⟩.
#[pyfunction]
pub fn lindblad_anti_hermitian_diag<'py>(
    py: Python<'py>,
    k_flat: PyReadonlyArray1<'_, f64>,
    n: usize,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_n(n, "n")?;
    let k = k_flat.as_slice().unwrap();
    let dim = 1usize << n;
    let mut diag = vec![0.0f64; dim];

    for i in 0..n {
        for j in 0..n {
            if i != j && k[i * n + j].abs() > threshold {
                for (idx, d) in diag.iter_mut().enumerate().take(dim) {
                    if ((idx >> i) & 1) == 1 && ((idx >> j) & 1) == 0 {
                        *d += 1.0;
                    }
                }
            }
        }
    }

    Ok(PyArray1::from_vec(py, diag))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_jump_ops_count_two_qubits() {
        // 2 qubits, K[0,1]=K[1,0]=1 → 2 jump operators (0→1 and 1→0)
        let n = 2;
        let k: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];
        let threshold = 0.01;

        let mut op_count = 0usize;
        for i in 0..n {
            for j in 0..n {
                if i != j && k[i * n + j].abs() > threshold {
                    op_count += 1;
                }
            }
        }
        assert_eq!(op_count, 2, "2 coupled qubits → 2 jump operators");
    }

    #[test]
    fn test_anti_hermitian_diag_ground() {
        // |00⟩ = idx 0: no excitation to transfer → diag[0] = 0
        let n = 2;
        let k: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];
        let dim = 1usize << n;
        let threshold = 0.01;
        let mut diag = vec![0.0f64; dim];

        for i in 0..n {
            for j in 0..n {
                if i != j && k[i * n + j].abs() > threshold {
                    for (idx, d) in diag.iter_mut().enumerate().take(dim) {
                        if ((idx >> i) & 1) == 1 && ((idx >> j) & 1) == 0 {
                            *d += 1.0;
                        }
                    }
                }
            }
        }

        assert!(diag[0].abs() < 1e-12, "|00⟩ has no active channels");
        assert!(diag[3].abs() < 1e-12, "|11⟩ has no active channels (both excited)");
        assert!(diag[1] > 0.0, "|01⟩ has active channel");
        assert!(diag[2] > 0.0, "|10⟩ has active channel");
    }
}
