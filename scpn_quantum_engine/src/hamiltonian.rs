// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — XY Hamiltonian Construction

//! Dense and sparse XY Hamiltonian construction via bitwise flip-flop.
//!
//! H = −Σ_{i<j} K[i,j](X_iX_j + Y_iY_j) − Σ_i ω_i Z_i
//!
//! Uses the identity (XX+YY)|↑↓⟩ = 2|↓↑⟩, zero when same spin, to construct
//! the Hamiltonian directly from bit patterns without Qiskit SparsePauliOp.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Build dense XY Hamiltonian directly from K coupling and ω frequencies.
///
/// Returns flat real array (XY Hamiltonian is real in computational basis).
/// Eliminates Qiskit SparsePauliOp construction + to_matrix() overhead.
#[pyfunction]
pub fn build_xy_hamiltonian_dense<'py>(
    py: Python<'py>,
    k_flat: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    n: usize,
) -> Bound<'py, PyArray1<f64>> {
    let k = k_flat.as_slice().unwrap();
    let w = omega.as_slice().unwrap();
    let dim = 1usize << n;
    let mut h = vec![0.0f64; dim * dim];

    for idx in 0..dim {
        // Diagonal: −ω_i Z_i, where Z eigenvalue = 1−2×bit
        let mut diag = 0.0;
        for (i, &wi) in w.iter().enumerate().take(n) {
            let bit = ((idx >> i) & 1) as f64;
            diag -= wi * (1.0 - 2.0 * bit);
        }
        h[idx * dim + idx] = diag;

        // Off-diagonal: −K[i,j]×(XX+YY) flip-flop
        for i in 0..n {
            for j in (i + 1)..n {
                let kij = k[i * n + j];
                if kij.abs() < 1e-15 {
                    continue;
                }
                let bi = (idx >> i) & 1;
                let bj = (idx >> j) & 1;
                if bi != bj {
                    let flipped = idx ^ ((1 << i) | (1 << j));
                    h[idx * dim + flipped] -= 2.0 * kij;
                }
            }
        }
    }

    PyArray1::from_vec(py, h)
}

/// Build sparse XY Hamiltonian as COO triplets (rows, cols, vals).
///
/// Same bitwise flip-flop as dense version but outputs sparse format
/// for scipy.sparse.csc_matrix construction.
#[allow(clippy::type_complexity)]
#[pyfunction]
pub fn build_sparse_xy_hamiltonian<'py>(
    py: Python<'py>,
    k_flat: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    n: usize,
) -> (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
) {
    let k = k_flat.as_slice().unwrap();
    let om = omega.as_slice().unwrap();
    let dim = 1usize << n;

    let mut rows: Vec<i64> = Vec::new();
    let mut cols: Vec<i64> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // Diagonal: −Σ ω_i (1 − 2×b_i(s))
    for s in 0..dim {
        let mut diag = 0.0f64;
        for (i, &omi) in om.iter().enumerate().take(n) {
            let bi = ((s >> i) & 1) as f64;
            diag -= omi * (1.0 - 2.0 * bi);
        }
        rows.push(s as i64);
        cols.push(s as i64);
        vals.push(diag);
    }

    // Off-diagonal: XY flip-flop
    for i in 0..n {
        for j in (i + 1)..n {
            let kij = k[i * n + j];
            if kij.abs() < 1e-15 {
                continue;
            }
            let mask = (1usize << i) | (1usize << j);
            let val = -2.0 * kij;
            for s in 0..dim {
                let bi = (s >> i) & 1;
                let bj = (s >> j) & 1;
                if bi != bj {
                    let s_flip = s ^ mask;
                    rows.push(s as i64);
                    cols.push(s_flip as i64);
                    vals.push(val);
                }
            }
        }
    }

    (
        PyArray1::from_vec(py, rows),
        PyArray1::from_vec(py, cols),
        PyArray1::from_vec(py, vals),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_dense_hamiltonian_hermitian() {
        // 2-qubit XY with K[0,1]=1, ω=[0,0]
        let n = 2;
        let dim = 1usize << n;
        let k = vec![0.0, 1.0, 1.0, 0.0]; // K[0,1]=K[1,0]=1
        let w = vec![0.0, 0.0];
        let mut h = vec![0.0f64; dim * dim];

        for idx in 0..dim {
            let mut diag = 0.0;
            for (i, &wi) in w.iter().enumerate().take(n) {
                let bit = ((idx >> i) & 1) as f64;
                diag -= wi * (1.0 - 2.0 * bit);
            }
            h[idx * dim + idx] = diag;

            for i in 0..n {
                for j in (i + 1)..n {
                    let kij: f64 = k[i * n + j];
                    if kij.abs() < 1e-15 {
                        continue;
                    }
                    let bi = (idx >> i) & 1;
                    let bj = (idx >> j) & 1;
                    if bi != bj {
                        let flipped = idx ^ ((1 << i) | (1 << j));
                        h[idx * dim + flipped] -= 2.0 * kij;
                    }
                }
            }
        }

        // Check symmetry (Hermitian for real matrix)
        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (h[i * dim + j] - h[j * dim + i]).abs() < 1e-12,
                    "H must be symmetric: H[{i},{j}]={} != H[{j},{i}]={}",
                    h[i * dim + j],
                    h[j * dim + i]
                );
            }
        }
    }

    #[test]
    fn test_dense_hamiltonian_flipflop() {
        // 2 qubits, K[0,1]=1: H should connect |01⟩↔|10⟩ with −2
        let n = 2;
        let dim = 1usize << n;
        let k = vec![0.0, 1.0, 1.0, 0.0];
        let mut h = vec![0.0f64; dim * dim];

        for idx in 0..dim {
            for i in 0..n {
                for j in (i + 1)..n {
                    let kij: f64 = k[i * n + j];
                    let bi = (idx >> i) & 1;
                    let bj = (idx >> j) & 1;
                    if bi != bj {
                        let flipped = idx ^ ((1 << i) | (1 << j));
                        h[idx * dim + flipped] -= 2.0 * kij;
                    }
                }
            }
        }

        // |01⟩ = index 1 (bit0=1, bit1=0), |10⟩ = index 2 (bit0=0, bit1=1)
        assert!((h[1 * dim + 2] - (-2.0)).abs() < 1e-12, "flip-flop element");
        assert!((h[2 * dim + 1] - (-2.0)).abs() < 1e-12, "symmetric flip-flop");
        // |00⟩↔|11⟩ should be zero (same spin → no flip-flop)
        assert!(h[0 * dim + 3].abs() < 1e-12, "no flip for same spin");
    }
}
