// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Dynamical Lie Algebra Computation

//! Dynamical Lie Algebra (DLA) dimension computation via commutator closure.
//!
//! For the SCPN XY Hamiltonian with heterogeneous frequencies, the DLA follows
//! DLA(N) = 2^(2N−1) − 2 = su(even) ⊕ su(odd), where Z₂ parity is the only
//! symmetry constraint. This module computes the DLA dimension by iteratively
//! generating commutators and checking linear independence via Gram-Schmidt.
//!
//! The commutator computation is parallelised via rayon; independence filtering
//! remains serial for correctness (order-dependent projection).

use pyo3::prelude::*;
use rayon::prelude::*;

/// DLA: compute dynamical Lie algebra dimension via commutator closure.
///
/// Takes a flat array of generator matrices (each dim×dim, row-major)
/// and computes the closure under commutation. Returns the DLA dimension.
///
/// This is the hot path that takes 27 min in Python for N=4. In Rust
/// with vectorised matrix ops, target is <30s.
#[pyfunction]
pub fn dla_dimension(
    generators_flat: numpy::PyReadonlyArray1<'_, f64>,
    dim: usize,
    n_generators: usize,
    max_iterations: usize,
    max_dimension: usize,
    tol: f64,
) -> usize {
    let data = generators_flat.as_slice().unwrap();
    let mat_size = dim * dim;

    let mut basis: Vec<Vec<f64>> = Vec::new();
    for g in 0..n_generators {
        let start = g * mat_size;
        let mat: Vec<f64> = data[start..start + mat_size].to_vec();
        if is_independent_fast(&mat, &basis, tol) {
            basis.push(mat);
        }
    }

    for _iter in 0..max_iterations {
        let n_basis = basis.len();
        if n_basis >= max_dimension {
            break;
        }

        let pairs: Vec<(usize, usize)> = (0..n_basis)
            .flat_map(|i| ((i + 1)..n_basis).map(move |j| (i, j)))
            .collect();

        let candidates: Vec<Vec<f64>> = pairs
            .par_iter()
            .filter_map(|&(i, j)| {
                let comm = commutator_dense(&basis[i], &basis[j], dim);
                let norm: f64 = comm.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < tol {
                    None
                } else {
                    Some(comm)
                }
            })
            .collect();

        let mut new_ops: Vec<Vec<f64>> = Vec::new();
        for comm in candidates {
            let mut combined = basis.clone();
            combined.extend(new_ops.iter().cloned());
            if is_independent_fast(&comm, &combined, tol) {
                new_ops.push(comm);
                if basis.len() + new_ops.len() >= max_dimension {
                    break;
                }
            }
        }

        if new_ops.is_empty() {
            break;
        }
        basis.extend(new_ops);
    }

    basis.len()
}

/// Dense matrix commutator [A, B] = AB − BA (row-major).
pub fn commutator_dense(a: &[f64], b: &[f64], dim: usize) -> Vec<f64> {
    let mut result = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut ab = 0.0;
            let mut ba = 0.0;
            for k in 0..dim {
                ab += a[i * dim + k] * b[k * dim + j];
                ba += b[i * dim + k] * a[k * dim + j];
            }
            result[i * dim + j] = ab - ba;
        }
    }
    result
}

/// Fast linear independence check via Gram-Schmidt projection.
pub fn is_independent_fast(new_op: &[f64], basis: &[Vec<f64>], tol: f64) -> bool {
    let new_norm: f64 = new_op.iter().map(|x| x * x).sum::<f64>().sqrt();
    if new_norm < tol {
        return false;
    }
    if basis.is_empty() {
        return true;
    }

    let mut residual: Vec<f64> = new_op.to_vec();
    for b in basis {
        let b_norm_sq: f64 = b.iter().map(|x| x * x).sum();
        if b_norm_sq < tol * tol {
            continue;
        }
        let dot: f64 = residual.iter().zip(b.iter()).map(|(r, bi)| r * bi).sum();
        let coeff = dot / b_norm_sq;
        for (r, bi) in residual.iter_mut().zip(b.iter()) {
            *r -= coeff * bi;
        }
    }

    let res_norm: f64 = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
    res_norm > tol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commutator_antisymmetric() {
        let a = vec![0.0, 1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 1.0, 0.0];
        let ab = commutator_dense(&a, &b, 2);
        let ba = commutator_dense(&b, &a, 2);
        for i in 0..4 {
            assert!((ab[i] + ba[i]).abs() < 1e-12, "[A,B] != -[B,A] at {i}");
        }
    }

    #[test]
    fn test_commutator_diagonal_zero() {
        let a = vec![1.0, 0.0, 0.0, 2.0];
        let result = commutator_dense(&a, &a, 2);
        for v in &result {
            assert!(v.abs() < 1e-12, "[A,A] must be zero");
        }
    }

    #[test]
    fn test_commutator_pauli_xz() {
        let x = vec![0.0, 1.0, 1.0, 0.0];
        let z = vec![1.0, 0.0, 0.0, -1.0];
        let c = commutator_dense(&x, &z, 2);
        assert!((c[0]).abs() < 1e-12);
        assert!((c[1] - (-2.0)).abs() < 1e-12);
        assert!((c[2] - 2.0).abs() < 1e-12);
        assert!((c[3]).abs() < 1e-12);
    }

    #[test]
    fn test_independent_empty_basis() {
        let op = vec![1.0, 0.0, 0.0, 1.0];
        assert!(is_independent_fast(&op, &[], 1e-10));
    }

    #[test]
    fn test_independent_zero_op() {
        let op = vec![0.0, 0.0, 0.0, 0.0];
        assert!(!is_independent_fast(&op, &[], 1e-10));
    }

    #[test]
    fn test_independent_duplicate() {
        let op = vec![1.0, 0.0, 0.0, 1.0];
        assert!(!is_independent_fast(&op, &[op.clone()], 1e-10));
    }

    #[test]
    fn test_independent_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 1.0];
        assert!(is_independent_fast(&b, &[a], 1e-10));
    }

    #[test]
    fn test_independent_scaled_duplicate() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![2.0, 0.0, 0.0, 2.0]; // 2×a → dependent
        assert!(!is_independent_fast(&b, &[a], 1e-10));
    }
}
