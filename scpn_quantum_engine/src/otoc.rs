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
//! eigenvectors here. Per time point: O(d) phase rotation + 3× O(d²) mat-vec.
//! Avoids constructing full W(t) matrix and 2× scipy.expm (O(d³) Padé).
//!
//! Ref: Swingle, Nature Physics 14, 988 (2018)

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::complex_utils::{c64, cmat_from_flat, conj_transpose, ct_matvec, cvec_from_parts, C64};

/// OTOC F(t) via eigendecomposition, parallel across time points (rayon).
/// Highly optimized O(d) phase applications + O(d²) mat-vec.
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
    // W_e = U† W U, V_e = U† V U
    let w_eig = u_h.dot(&w_mat).dot(&u);
    let v_eig = u_h.dot(&v_mat).dot(&u);
    let w_eig_h = conj_transpose(&w_eig);
    let psi_e = u_h.dot(&psi);
    let state0 = v_eig.dot(&psi_e); // V|ψ⟩ in eigenbasis

    let results: Vec<f64> = t_arr
        .par_iter()
        .map(|&t| {
            // W(t) = D(t) W_e D†(t), where D(t) = diag(exp(i E_j t))
            // We apply rotations to vectors to keep it O(d²)
            
            // 1. phases[j] = exp(i E_j t)
            let mut phases = Vec::with_capacity(dim);
            for &e in evals {
                let ph = e * t;
                phases.push(c64(ph.cos(), ph.sin()));
            }

            // 2. s1 = W(t) |state0⟩ = D(t) W_e D†(t) |state0⟩
            let mut temp = state0.clone();
            for j in 0..dim { temp[j] *= phases[j].conj(); }
            let mut s1 = w_eig.dot(&temp);
            for j in 0..dim { s1[j] *= phases[j]; }

            // 3. s2 = V_e† s1
            let s2 = ct_matvec(&v_eig, &s1);

            // 4. s3 = W†(t) s2 = D(t) W_e† D†(t) s2
            let mut temp2 = s2;
            for j in 0..dim { temp2[j] *= phases[j].conj(); }
            let mut s3 = w_eig_h.dot(&temp2);
            for j in 0..dim { s3[j] *= phases[j]; }

            // 5. F(t) = Re(⟨ψ_e| s3⟩)
            let mut ft = 0.0;
            for j in 0..dim {
                ft += (psi_e[j].conj() * s3[j]).re;
            }
            ft
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
        let evals: Vec<f64> = vec![0.0, 1.0];
        let w_eig = ndarray::Array2::eye(dim);
        let v_eig = ndarray::Array2::eye(dim);
        let psi_e = ndarray::Array1::from_vec(vec![c64(1.0, 0.0), c64(0.0, 0.0)]);
        let state0 = v_eig.dot(&psi_e);
        let t = 0.0;

        let mut phases = Vec::with_capacity(dim);
        for &e in &evals {
            let ph = e * t;
            phases.push(c64(ph.cos(), ph.sin()));
        }

        let mut temp = state0.clone();
        for j in 0..dim { temp[j] *= phases[j].conj(); }
        let mut s1 = w_eig.dot(&temp);
        for j in 0..dim { s1[j] *= phases[j]; }

        let s2 = ct_matvec(&v_eig, &s1);

        let w_eig_h = conj_transpose(&w_eig);
        let mut temp2 = s2;
        for j in 0..dim { temp2[j] *= phases[j].conj(); }
        let mut s3 = w_eig_h.dot(&temp2);
        for j in 0..dim { s3[j] *= phases[j]; }

        let mut f0 = 0.0;
        for j in 0..dim {
            f0 += (psi_e[j].conj() * s3[j]).re;
        }

        assert!((f0 - 1.0).abs() < 1e-10, "F(0) with W=V=I should be 1, got {f0}");
    }
}
