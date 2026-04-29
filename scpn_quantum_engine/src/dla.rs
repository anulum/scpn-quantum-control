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

use ndarray::Axis;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::validation::{validate_n, validate_positive};

/// DLA: compute dynamical Lie algebra dimension via commutator closure.
///
/// Takes a flat array of generator matrices (each dim×dim, row-major)
/// and computes the closure under commutation. Returns the DLA dimension.
///
/// This is the hot path that takes 27 min in Python for N=4. In Rust
/// with vectorised matrix ops, target is <30s.
#[pyfunction]
pub fn dla_dimension(
    generators_flat: PyReadonlyArray1<'_, f64>,
    dim: usize,
    n_generators: usize,
    max_iterations: usize,
    max_dimension: usize,
    tol: f64,
) -> PyResult<usize> {
    validate_n(dim, "dim")?;
    validate_n(n_generators, "n_generators")?;
    validate_n(max_iterations, "max_iterations")?;
    validate_n(max_dimension, "max_dimension")?;
    validate_positive(tol, "tol")?;

    let data = generators_flat.as_slice().unwrap();
    let mat_size = dim * dim;
    let expected_len = n_generators * mat_size;
    if data.len() < expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "generators_flat too short: {} < {} (n_generators={} × dim²={})",
            data.len(),
            expected_len,
            n_generators,
            mat_size
        )));
    }

    Ok(dla_dimension_inner(
        data,
        dim,
        n_generators,
        max_iterations,
        max_dimension,
        tol,
    ))
}

/// Build the fixed-parity repetition-code memory mask.
///
/// The physical layout is contiguous blocks of `code_distance` qubits per
/// logical oscillator. A basis state is inside the memory manifold when every
/// block is either all-zero or all-one. For odd `code_distance`, the block
/// parity equals the logical bit, so the target global parity selects a
/// DLA-invariant logical sector.
#[pyfunction]
pub fn dla_protected_memory_mask<'py>(
    py: Python<'py>,
    n_logical: usize,
    code_distance: usize,
    target_parity: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let total_qubits = validate_memory_shape(n_logical, code_distance, target_parity)?;
    let dim = 1usize << total_qubits;
    let mut mask = Vec::with_capacity(dim);
    for state in 0..dim {
        let (in_code, logical_parity) =
            memory_code_and_logical_parity(state, n_logical, code_distance);
        mask.push(in_code && logical_parity == target_parity);
    }
    Ok(PyArray1::from_vec(py, mask))
}

/// Score probability weight in the protected memory, code, and parity sectors.
#[pyfunction]
pub fn dla_protected_memory_metrics(
    probabilities: PyReadonlyArray1<'_, f64>,
    n_logical: usize,
    code_distance: usize,
    target_parity: usize,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let total_qubits = validate_memory_shape(n_logical, code_distance, target_parity)?;
    let dim = 1usize << total_qubits;
    let probs = probabilities.as_slice().unwrap();
    if probs.len() != dim {
        return Err(PyValueError::new_err(format!(
            "probabilities length must be 2^(n_logical*code_distance) = {dim}, got {}",
            probs.len()
        )));
    }

    memory_metrics_inner(probs, n_logical, code_distance, target_parity)
        .map_err(PyValueError::new_err)
}

/// Score a time series of probability vectors in the protected DLA sectors.
#[pyfunction]
pub fn dla_protected_trajectory_metrics<'py>(
    py: Python<'py>,
    probabilities: PyReadonlyArray2<'_, f64>,
    n_logical: usize,
    code_distance: usize,
    target_parity: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let total_qubits = validate_memory_shape(n_logical, code_distance, target_parity)?;
    let dim = 1usize << total_qubits;
    let probs = probabilities.as_array();
    if probs.shape()[1] != dim {
        return Err(PyValueError::new_err(format!(
            "probabilities second dimension must be 2^(n_logical*code_distance) = {dim}, got {}",
            probs.shape()[1]
        )));
    }

    let n_times = probs.shape()[0];
    let mut protected = Vec::with_capacity(n_times);
    let mut code = Vec::with_capacity(n_times);
    let mut target = Vec::with_capacity(n_times);
    let mut opposite = Vec::with_capacity(n_times);
    let mut total = Vec::with_capacity(n_times);

    for row in probs.axis_iter(Axis(0)) {
        let row_slice = row.as_slice().ok_or_else(|| {
            PyValueError::new_err("probability trajectory rows must be contiguous")
        })?;
        let (p, c, t, o, sum) =
            memory_metrics_inner(row_slice, n_logical, code_distance, target_parity)
                .map_err(PyValueError::new_err)?;
        protected.push(p);
        code.push(c);
        target.push(t);
        opposite.push(o);
        total.push(sum);
    }

    Ok((
        PyArray1::from_vec(py, protected),
        PyArray1::from_vec(py, code),
        PyArray1::from_vec(py, target),
        PyArray1::from_vec(py, opposite),
        PyArray1::from_vec(py, total),
    ))
}

/// Pure Rust DLA dimension computation (testable without Python).
pub fn dla_dimension_inner(
    data: &[f64],
    dim: usize,
    n_generators: usize,
    max_iterations: usize,
    max_dimension: usize,
    tol: f64,
) -> usize {
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

fn memory_metrics_inner(
    probs: &[f64],
    n_logical: usize,
    code_distance: usize,
    target_parity: usize,
) -> Result<(f64, f64, f64, f64, f64), String> {
    let mut protected_weight = 0.0;
    let mut code_weight = 0.0;
    let mut target_parity_weight = 0.0;
    let mut opposite_parity_weight = 0.0;
    let mut total_weight = 0.0;

    for (state, &probability) in probs.iter().enumerate() {
        if !probability.is_finite() || probability < 0.0 {
            return Err("probabilities must contain finite non-negative values".to_string());
        }
        total_weight += probability;
        let physical_parity = state.count_ones() as usize % 2;
        if physical_parity == target_parity {
            target_parity_weight += probability;
        } else {
            opposite_parity_weight += probability;
        }
        let (in_code, logical_parity) =
            memory_code_and_logical_parity(state, n_logical, code_distance);
        if in_code {
            code_weight += probability;
            if logical_parity == target_parity {
                protected_weight += probability;
            }
        }
    }

    Ok((
        protected_weight,
        code_weight,
        target_parity_weight,
        opposite_parity_weight,
        total_weight,
    ))
}

fn validate_memory_shape(
    n_logical: usize,
    code_distance: usize,
    target_parity: usize,
) -> PyResult<usize> {
    validate_n(n_logical, "n_logical")?;
    validate_n(code_distance, "code_distance")?;
    if code_distance % 2 == 0 {
        return Err(PyValueError::new_err(format!(
            "code_distance must be odd, got {code_distance}"
        )));
    }
    if target_parity > 1 {
        return Err(PyValueError::new_err(format!(
            "target_parity must be 0 or 1, got {target_parity}"
        )));
    }
    let total_qubits = n_logical
        .checked_mul(code_distance)
        .ok_or_else(|| PyValueError::new_err("n_logical*code_distance overflows usize"))?;
    if total_qubits > 24 {
        return Err(PyValueError::new_err(format!(
            "n_logical*code_distance must be <= 24 for dense masks, got {total_qubits}"
        )));
    }
    Ok(total_qubits)
}

fn memory_code_and_logical_parity(
    state: usize,
    n_logical: usize,
    code_distance: usize,
) -> (bool, usize) {
    let block_mask = (1usize << code_distance) - 1;
    let mut logical_parity = 0usize;
    for logical in 0..n_logical {
        let shift = logical * code_distance;
        let block = (state >> shift) & block_mask;
        if block == 0 {
            continue;
        }
        if block == block_mask {
            logical_parity ^= 1;
        } else {
            return (false, logical_parity);
        }
    }
    (true, logical_parity)
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

    #[test]
    fn test_dla_dimension_inner_pauli() {
        // Pauli X and Z generate su(2), dimension 3
        let x = vec![0.0, 1.0, 1.0, 0.0];
        let z = vec![1.0, 0.0, 0.0, -1.0];
        let mut data = Vec::new();
        data.extend_from_slice(&x);
        data.extend_from_slice(&z);
        let dim = dla_dimension_inner(&data, 2, 2, 100, 100, 1e-10);
        assert_eq!(dim, 3, "Pauli X,Z should generate su(2) with dim=3");
    }

    #[test]
    fn test_memory_code_logical_parity_for_odd_repetition_blocks() {
        assert_eq!(memory_code_and_logical_parity(0b000_111, 2, 3), (true, 1));
        assert_eq!(memory_code_and_logical_parity(0b111_111, 2, 3), (true, 0));
        assert_eq!(memory_code_and_logical_parity(0b010_111, 2, 3), (false, 1));
    }

    #[test]
    fn test_memory_metrics_inner_tracks_code_and_parity_weights() {
        let mut probs = vec![0.0; 64];
        probs[0b000_000] = 0.45;
        probs[0b111_111] = 0.40;
        probs[0b000_111] = 0.10;
        probs[0b010_111] = 0.05;

        let (protected, code, target, opposite, total) =
            memory_metrics_inner(&probs, 2, 3, 0).unwrap();

        assert!((protected - 0.85).abs() < 1e-12);
        assert!((code - 0.95).abs() < 1e-12);
        assert!((target - 0.90).abs() < 1e-12);
        assert!((opposite - 0.10).abs() < 1e-12);
        assert!((total - 1.0).abs() < 1e-12);
    }
}
