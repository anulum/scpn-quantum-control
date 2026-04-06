// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Out-of-Time-Order Correlator (OTOC)

//! OTOC F(t) computation via eigendecomposition, parallel across time points.
//!
//! F(t) = Re(⟨ψ| W†(t) V† W(t) V |ψ⟩) where W(t) = e^{iHt} W e^{−iHt}.
//!
//! Diagonalise H once (in Python via numpy.linalg.eigh), pass eigenvalues +
//! eigenvectors here. Each time point: O(d²) phase rotation + mat-vec products.
//! Avoids 2× scipy.expm (O(d³) Padé) per time point.
//!
//! Ref: Swingle, Nature Physics 14, 988 (2018)

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::complex_utils::{c64, cmat_from_flat, conj_transpose, ct_matvec, cvec_from_parts, C64};

/// OTOC F(t) via eigendecomposition, parallel across time points (rayon).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn otoc_from_eigendecomp<'py>(
    py: Python<'py>,
    eigenvalues: PyReadonlyArray1<'_, f64>,
    eigvecs_re: PyReadonlyArray1<'_, f64>,
    eigvecs_im: PyReadonlyArray1<'_, f64>,
    w_re: PyReadonlyArray1<'_, f64>,
    w_im: PyReadonlyArray1<'_, f64>,
    v_re: PyReadonlyArray1<'_, f64>,
    v_im: PyReadonlyArray1<'_, f64>,
    psi_re: PyReadonlyArray1<'_, f64>,
    psi_im: PyReadonlyArray1<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
    dim: usize,
) -> Bound<'py, PyArray1<f64>> {
    let evals = eigenvalues.as_slice().unwrap();
    let u = cmat_from_flat(
        eigvecs_re.as_slice().unwrap(),
        eigvecs_im.as_slice().unwrap(),
        dim,
    );
    let u_h = conj_transpose(&u);
    let w_mat = cmat_from_flat(w_re.as_slice().unwrap(), w_im.as_slice().unwrap(), dim);
    let v_mat = cmat_from_flat(v_re.as_slice().unwrap(), v_im.as_slice().unwrap(), dim);
    let psi = cvec_from_parts(psi_re.as_slice().unwrap(), psi_im.as_slice().unwrap());
    let t_arr = times.as_slice().unwrap();

    // Transform to eigenbasis (done once)
    let w_eig = u_h.dot(&w_mat).dot(&u);
    let v_eig = u_h.dot(&v_mat).dot(&u);
    let psi_e = u_h.dot(&psi);
    let state0 = v_eig.dot(&psi_e); // V|ψ⟩ in eigenbasis

    let results: Vec<f64> = t_arr
        .par_iter()
        .map(|&t| {
            // W(t)[i,j] = exp(i(E_i − E_j)t) × W_eig[i,j]
            let mut w_t = Array2::<C64>::zeros((dim, dim));
            for i in 0..dim {
                for j in 0..dim {
                    let ph = (evals[i] - evals[j]) * t;
                    w_t[[i, j]] = c64(ph.cos(), ph.sin()) * w_eig[[i, j]];
                }
            }
            // F(t) = Re(⟨ψ_e| W_t† V_e† W_t |state0⟩)
            let s1 = w_t.dot(&state0);
            let s2 = ct_matvec(&v_eig, &s1);
            let s3 = ct_matvec(&w_t, &s2);
            psi_e
                .iter()
                .zip(s3.iter())
                .map(|(&p, &s)| (p.conj() * s).re)
                .sum::<f64>()
        })
        .collect();

    PyArray1::from_owned_array(py, Array1::from_vec(results))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otoc_t_zero_identity_w() {
        // W = V = I, F(0) = |⟨ψ|ψ⟩|² = 1
        let dim = 2;
        let u_re = vec![1.0, 0.0, 0.0, 1.0]; // identity eigvecs
        let u_im = vec![0.0; 4];
        let w_re = vec![1.0, 0.0, 0.0, 1.0]; // W = I
        let w_im = vec![0.0; 4];
        let v_re = vec![1.0, 0.0, 0.0, 1.0]; // V = I
        let v_im = vec![0.0; 4];
        let psi_re = vec![1.0, 0.0]; // |0⟩
        let psi_im = vec![0.0; 2];

        let u = cmat_from_flat(&u_re, &u_im, dim);
        let u_h = conj_transpose(&u);
        let w_mat = cmat_from_flat(&w_re, &w_im, dim);
        let v_mat = cmat_from_flat(&v_re, &v_im, dim);
        let psi = cvec_from_parts(&psi_re, &psi_im);

        let w_eig = u_h.dot(&w_mat).dot(&u);
        let v_eig = u_h.dot(&v_mat).dot(&u);
        let psi_e = u_h.dot(&psi);
        let state0 = v_eig.dot(&psi_e);

        // At t=0: W(0) = W_eig (no phase rotation)
        let s1 = w_eig.dot(&state0);
        let s2 = ct_matvec(&v_eig, &s1);
        let s3 = ct_matvec(&w_eig, &s2);
        let f0: f64 = psi_e
            .iter()
            .zip(s3.iter())
            .map(|(&p, &s)| (p.conj() * s).re)
            .sum();

        assert!((f0 - 1.0).abs() < 1e-10, "F(0) with W=V=I should be 1, got {f0}");
    }
}
