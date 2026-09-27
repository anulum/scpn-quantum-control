// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — XY Hamiltonian Construction

//! Dense and sparse XY Hamiltonian construction via bitwise flip-flop.
//!
//! `H = −Σ_{i<j} K[i,j](X_iX_j + Y_iY_j) − Σ_i ω_i Z_i`
//!
//! Uses the identity (XX+YY)|↑↓⟩ = 2|↓↑⟩, zero when same spin, to construct
//! the Hamiltonian directly from bit patterns without Qiskit SparsePauliOp.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

use crate::validation::{
    validate_contiguous_slice, validate_finite, validate_flat_square, validate_n,
};

fn validate_frequency_vector(omega: &[f64], n: usize) -> PyResult<()> {
    if omega.len() != n {
        return Err(PyValueError::new_err(format!(
            "omega length {} != n {n}",
            omega.len()
        )));
    }
    validate_finite(omega, "omega")
}

fn hamiltonian_dimension(n: usize) -> PyResult<usize> {
    let exponent = u32::try_from(n)
        .map_err(|_| PyValueError::new_err("Hamiltonian exponent exceeds native addressability"))?;
    1usize
        .checked_shl(exponent)
        .ok_or_else(|| PyValueError::new_err("Hamiltonian dimension exceeds native addressability"))
}

fn reserve_native_memory<'py>(
    py: Python<'py>,
    shape: (usize, usize),
    dtype: &str,
    count: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let memory = py.import("scpn_quantum_control.execution_memory")?;
    let buffer = memory.getattr("ExecutionBuffer")?.call1((
        "native_hamiltonian_output",
        "dense_output",
        shape,
        dtype,
        count,
    ))?;
    let plan = memory.getattr("ExecutionMemoryPlan")?.call1(((buffer,),))?;
    let manager = py
        .import("scpn_quantum_control.execution_reservations")?
        .getattr("reserve_execution_memory")?
        .call1((plan,))?;
    let reservation = manager.call_method0("__enter__")?;
    Ok((manager, reservation))
}

/// Build dense XY Hamiltonian directly from K coupling and ω frequencies.
///
/// Returns flat real array (XY Hamiltonian is real in computational basis).
/// Eliminates Qiskit SparsePauliOp construction + to_matrix() overhead.
/// Direct callers use the installed control package's shared memory/lifecycle policy.
#[pyfunction]
pub fn build_xy_hamiltonian_dense<'py>(
    py: Python<'py>,
    k_flat: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_n(n, "n")?;
    let dim = hamiltonian_dimension(n)?;
    let elements = dim
        .checked_mul(dim)
        .filter(|count| *count <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| PyValueError::new_err("Hamiltonian bytes exceed native addressability"))?;
    let k = validate_contiguous_slice(&k_flat, "k_flat")?;
    let w = validate_contiguous_slice(&omega, "omega")?;
    validate_flat_square(k, n, "k_flat")?;
    validate_finite(k, "k_flat")?;
    validate_frequency_vector(w, n)?;
    let (manager, reservation) = reserve_native_memory(py, (dim, dim), "float64", 1)?;
    let result = (|| {
        reservation.call_method0("checkpoint")?;
        let mut h = Vec::new();
        h.try_reserve_exact(elements)
            .map_err(|error| PyMemoryError::new_err(error.to_string()))?;
        h.resize(elements, 0.0f64);

        for idx in 0..dim {
            if idx & 255 == 0 {
                reservation.call_method0("checkpoint")?;
            }
            // Diagonal: −ω_i Z_i, where Z eigenvalue = 1−2×bit
            let mut diag = 0.0;
            for (i, &wi) in w.iter().enumerate().take(n) {
                if wi.abs() > 1e-15 {
                    let bit = ((idx >> i) & 1) as f64;
                    diag -= wi * (1.0 - 2.0 * bit);
                }
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

        validate_finite(&h, "Hamiltonian output")?;
        reservation.call_method0("checkpoint")?;
        Ok(PyArray1::from_vec(py, h))
    })();
    manager.call_method1("__exit__", (py.None(), py.None(), py.None()))?;
    result
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
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validate_n(n, "n")?;
    let dim = hamiltonian_dimension(n)?;
    let k = validate_contiguous_slice(&k_flat, "k_flat")?;
    let om = validate_contiguous_slice(&omega, "omega")?;
    validate_flat_square(k, n, "k_flat")?;
    validate_finite(k, "k_flat")?;
    validate_frequency_vector(om, n)?;
    let mut pairs = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if k[i * n + j].abs() >= 1e-15 {
                pairs += 1;
            }
        }
    }
    let entries = pairs
        .checked_mul(dim / 2)
        .and_then(|off_diagonal| off_diagonal.checked_add(dim))
        .ok_or_else(|| PyValueError::new_err("Hamiltonian entries exceed native addressability"))?;
    let bytes = entries
        .checked_mul(std::mem::size_of::<i64>().max(std::mem::size_of::<f64>()))
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| PyValueError::new_err("Hamiltonian bytes exceed native addressability"))?;
    let (manager, reservation) = reserve_native_memory(py, (bytes, 1), "uint8", 3)?;
    let result = (|| {
        reservation.call_method0("checkpoint")?;
        let mut rows: Vec<i64> = Vec::new();
        let mut cols: Vec<i64> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        rows.try_reserve_exact(entries)
            .map_err(|error| PyMemoryError::new_err(error.to_string()))?;
        cols.try_reserve_exact(entries)
            .map_err(|error| PyMemoryError::new_err(error.to_string()))?;
        vals.try_reserve_exact(entries)
            .map_err(|error| PyMemoryError::new_err(error.to_string()))?;

        // Diagonal: −Σ ω_i (1 − 2×b_i(s))
        for s in 0..dim {
            if s & 255 == 0 {
                reservation.call_method0("checkpoint")?;
            }
            let mut diag = 0.0f64;
            for (i, &omi) in om.iter().enumerate().take(n) {
                if omi.abs() > 1e-15 {
                    let bi = ((s >> i) & 1) as f64;
                    diag -= omi * (1.0 - 2.0 * bi);
                }
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
                    if s & 255 == 0 {
                        reservation.call_method0("checkpoint")?;
                    }
                    let bi = (s >> i) & 1;
                    let bj = (s >> j) & 1;
                    if bi != bj {
                        if rows.len() >= entries {
                            return Err(PyValueError::new_err(
                                "couplings changed during Hamiltonian construction",
                            ));
                        }
                        let s_flip = s ^ mask;
                        rows.push(s as i64);
                        cols.push(s_flip as i64);
                        vals.push(val);
                    }
                }
            }
        }

        if rows.len() != entries {
            return Err(PyValueError::new_err(
                "couplings changed during Hamiltonian construction",
            ));
        }
        validate_finite(&vals, "Hamiltonian output")?;
        reservation.call_method0("checkpoint")?;
        Ok((
            PyArray1::from_vec(py, rows),
            PyArray1::from_vec(py, cols),
            PyArray1::from_vec(py, vals),
        ))
    })();
    manager.call_method1("__exit__", (py.None(), py.None(), py.None()))?;
    result
}

#[expect(
    clippy::identity_op,
    clippy::erasing_op,
    reason = "row-major indices are written out as `row * dim + col` so each \
              assertion names the matrix element it checks; collapsing \
              `0 * dim` or `1 * dim` would hide which element that is"
)]
#[cfg(test)]
mod tests {
    #[test]
    fn test_dense_hamiltonian_hermitian() {
        // 2-qubit XY with K[0,1]=1, ω=[0,0]
        let n = 2;
        let dim = 1usize << n;
        let k = [0.0, 1.0, 1.0, 0.0]; // K[0,1]=K[1,0]=1
        let w = [0.0, 0.0];
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
        let k = [0.0, 1.0, 1.0, 0.0];
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
        assert!(
            (h[2 * dim + 1] - (-2.0)).abs() < 1e-12,
            "symmetric flip-flop"
        );
        // |00⟩↔|11⟩ should be zero (same spin → no flip-flop)
        assert!(h[0 * dim + 3].abs() < 1e-12, "no flip for same spin");
    }
}
