// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — NV-centre ODMR spectrum kernel

//! Lorentzian CW-ODMR photoluminescence spectrum.
//!
//! Bit-true with the NumPy reference in
//! `scpn_quantum_control.sensing.nv_magnetometry_20T._lorentzian_dip`:
//! `spectrum[i] = 1 - sum_c contrast * (half^2 / ((f_i - c)^2 + half^2))`,
//! `half = fwhm / 2`.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Normalised ODMR spectrum (a Lorentzian dip per resonance centre).
#[pyfunction]
pub fn nv_odmr_spectrum<'py>(
    py: Python<'py>,
    freqs: PyReadonlyArray1<'_, f64>,
    centers: PyReadonlyArray1<'_, f64>,
    fwhm: f64,
    contrast: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !fwhm.is_finite() || fwhm <= 0.0 {
        return Err(PyValueError::new_err("fwhm must be finite and positive"));
    }
    if !contrast.is_finite() {
        return Err(PyValueError::new_err("contrast must be finite"));
    }
    let f = freqs.as_slice()?;
    let c = centers.as_slice()?;
    let half = fwhm / 2.0;
    let half_sq = half * half;
    let mut out = vec![0.0_f64; f.len()];
    for (i, &freq) in f.iter().enumerate() {
        // Sequential per-centre subtraction matches NumPy's `spectrum -= term`
        // exactly (1 - t0 - t1 differs from 1 - (t0 + t1) in IEEE-754).
        let mut s = 1.0_f64;
        for &center in c {
            let delta = freq - center;
            s -= contrast * half_sq / (delta * delta + half_sq);
        }
        out[i] = s;
    }
    Ok(PyArray1::from_vec(py, out))
}
