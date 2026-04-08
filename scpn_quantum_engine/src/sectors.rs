// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Symmetry Sectors, Correlation, and Parity

//! Magnetisation sectors, order parameter from statevector, XY correlation
//! matrix, and Z�� parity filtering.
//!
//! The XY Hamiltonian preserves total magnetisation M = N − 2×popcount(k),
//! enabling block-diagonal analysis. The correlation matrix C[i,j] = ⟨XX_ij + YY_ij⟩
//! drives Hebbian learning in DynamicCouplingEngine.

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::validation::validate_n;

fn validate_statevec(len: usize, n: usize, name: &str) -> PyResult<()> {
    crate::validation::to_pyresult(crate::validation::check_statevec_len(len, n, name))
}

/// Magnetisation labels: result[k] = M of basis state |k⟩.
/// M = n − 2 × popcount(k). Uses hardware popcount instruction.
#[pyfunction]
pub fn magnetisation_labels<'py>(py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<i32>>> {
    validate_n(n, "n")?;
    if n > 30 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n={n} too large (max 30, would allocate 2^{n} entries)"
        )));
    }
    let dim = 1usize << n;
    let mut labels = Vec::with_capacity(dim);
    let n_i32 = n as i32;
    for k in 0..dim {
        let popcount = (k as u64).count_ones() as i32;
        labels.push(n_i32 - 2 * popcount);
    }
    Ok(PyArray1::from_vec(py, labels))
}

/// Order parameter R from complex statevector.
/// R = (1/N)|Σ_i (⟨X_i⟩ + i⟨Y_i⟩)| computed via bitwise Pauli.
///
/// Same logic as state_order_param_sparse but for complex statevectors
/// used in tensor_jump MCWF trajectories.
#[pyfunction]
pub fn order_param_from_statevector(
    psi_re: PyReadonlyArray1<'_, f64>,
    psi_im: PyReadonlyArray1<'_, f64>,
    n: usize,
) -> PyResult<f64> {
    validate_n(n, "n")?;
    let re = psi_re.as_slice().unwrap();
    let im = psi_im.as_slice().unwrap();
    validate_statevec(re.len(), n, "psi_re")?;
    let dim = 1usize << n;

    let mut z_re = 0.0f64;
    let mut z_im = 0.0f64;

    for i in 0..n {
        let mut exp_x = 0.0f64;
        let mut exp_y = 0.0f64;
        let mask = 1usize << i;
        for k in 0..dim {
            let k_flip = k ^ mask;
            let re_prod = re[k] * re[k_flip] + im[k] * im[k_flip];
            let im_prod = re[k] * im[k_flip] - im[k] * re[k_flip];
            exp_x += re_prod;
            exp_y += im_prod;
        }
        z_re += exp_x;
        z_im += exp_y;
    }

    z_re /= n as f64;
    z_im /= n as f64;
    Ok((z_re * z_re + z_im * z_im).sqrt())
}

/// XY correlation matrix C[i,j] = ⟨XX_ij + YY_ij⟩ from statevector.
/// Used by DynamicCouplingEngine for Hebbian learning.
/// Parallelised over qubit pairs via rayon.
#[pyfunction]
pub fn correlation_matrix_xy<'py>(
    py: Python<'py>,
    psi_re: PyReadonlyArray1<'_, f64>,
    psi_im: PyReadonlyArray1<'_, f64>,
    n_osc: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    validate_n(n_osc, "n_osc")?;
    let re = psi_re.as_slice().unwrap();
    let im = psi_im.as_slice().unwrap();
    validate_statevec(re.len(), n_osc, "psi_re")?;
    let dim = 1usize << n_osc;

    let pairs: Vec<(usize, usize)> = (0..n_osc)
        .flat_map(|i| ((i + 1)..n_osc).map(move |j| (i, j)))
        .collect();

    let results: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mask = (1usize << i) | (1usize << j);
            let mut corr = 0.0f64;
            for k in 0..dim {
                let bi = (k >> i) & 1;
                let bj = (k >> j) & 1;
                if bi != bj {
                    let k_flip = k ^ mask;
                    corr += 2.0 * (re[k] * re[k_flip] + im[k] * im[k_flip]);
                }
            }
            (i, j, corr)
        })
        .collect();

    let mut c = Array2::<f64>::zeros((n_osc, n_osc));
    for (i, j, corr) in results {
        c[(i, j)] = corr;
        c[(j, i)] = corr;
    }

    Ok(PyArray2::from_owned_array(py, c))
}

/// Z₂ parity filter for measurement counts (compound mitigation).
///
/// Takes a flat array of bitstring values and returns a boolean mask
/// indicating which bitstrings match the expected parity.
#[pyfunction]
pub fn parity_filter_mask<'py>(
    py: Python<'py>,
    bitstrings: PyReadonlyArray1<'_, u64>,
    expected_parity: u8,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    if expected_parity > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected_parity must be 0 or 1, got {expected_parity}"
        )));
    }
    let bs = bitstrings.as_slice().unwrap();
    let mask: Vec<bool> = bs
        .par_iter()
        .map(|&val| (val.count_ones() as u8 % 2) == expected_parity)
        .collect();
    Ok(PyArray1::from_vec(py, mask))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_magnetisation_labels_2qubit() {
        let n = 2;
        let dim = 1usize << n;
        let mut labels = Vec::with_capacity(dim);
        let n_i32 = n as i32;
        for k in 0..dim {
            let popcount = (k as u64).count_ones() as i32;
            labels.push(n_i32 - 2 * popcount);
        }
        // |00⟩=+2, |01⟩=0, |10⟩=0, |11⟩=−2
        assert_eq!(labels, vec![2, 0, 0, -2]);
    }

    #[test]
    fn test_parity_filter() {
        let bitstrings: Vec<u64> = vec![0b00, 0b01, 0b10, 0b11];
        let expected_parity = 0u8; // even parity
        let mask: Vec<bool> = bitstrings
            .iter()
            .map(|&val| (val.count_ones() as u8 % 2) == expected_parity)
            .collect();
        // |00⟩ even, |01⟩ odd, |10⟩ odd, |11⟩ even
        assert_eq!(mask, vec![true, false, false, true]);
    }

    #[test]
    fn test_parity_filter_odd() {
        let bitstrings: Vec<u64> = vec![0b00, 0b01, 0b10, 0b11];
        let expected_parity = 1u8;
        let mask: Vec<bool> = bitstrings
            .iter()
            .map(|&val| (val.count_ones() as u8 % 2) == expected_parity)
            .collect();
        assert_eq!(mask, vec![false, true, true, false]);
    }
}
