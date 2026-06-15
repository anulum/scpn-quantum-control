// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — quantum/classical co-simulation classical substep

//! One explicit-Euler Kuramoto substep for the classical bath of a
//! quantum/classical co-simulation.
//!
//! The classical oscillators feel their internal pairwise coupling plus a
//! mean-field drive from the quantum-strong core. The quantum drive on node `i`
//! is `cos(theta_i) * a_i - sin(theta_i) * b_i`, where `a_i`/`b_i` are the
//! sine/cosine-weighted quantum-core moments precomputed by the Python engine
//! (constant across the substep). Zero couplings are skipped, so a sparse
//! classical bath costs far less than a dense NumPy sweep. Agrees with the
//! NumPy reference in `scpn_quantum_control.cosimulation.quantum_classical` to
//! floating-point rounding (libm vs. std differences only).

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::validation::{validate_contiguous_slice, validate_positive};

/// Advance the classical phases by one Euler step `dt`.
///
/// Inputs are NumPy buffers (no per-call list marshalling): `theta`, `omega`,
/// `drive_a`, `drive_b` are length `n`; `k_classical` is the `n x n` coupling.
#[pyfunction]
pub fn cosim_classical_substep<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k_classical: PyReadonlyArray2<'_, f64>,
    drive_a: PyReadonlyArray1<'_, f64>,
    drive_b: PyReadonlyArray1<'_, f64>,
    dt: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_positive(dt, "dt")?;
    let theta = validate_contiguous_slice(&theta, "theta")?;
    let omega = validate_contiguous_slice(&omega, "omega")?;
    let drive_a = validate_contiguous_slice(&drive_a, "drive_a")?;
    let drive_b = validate_contiguous_slice(&drive_b, "drive_b")?;
    let n = theta.len();
    if omega.len() != n || drive_a.len() != n || drive_b.len() != n {
        return Err(PyValueError::new_err(
            "theta, omega, drive_a, drive_b must share the same length",
        ));
    }
    let k = k_classical.as_array();
    if k.dim() != (n, n) {
        return Err(PyValueError::new_err(
            "k_classical must be an n x n matrix matching theta",
        ));
    }
    for (name, slice) in [
        ("theta", theta),
        ("omega", omega),
        ("drive_a", drive_a),
        ("drive_b", drive_b),
    ] {
        if slice.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(format!("{name} must be finite")));
        }
    }

    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let theta_i = theta[i];
        let mut acc = omega[i];
        for j in 0..n {
            let kij = k[[i, j]];
            if kij != 0.0 {
                acc += kij * (theta[j] - theta_i).sin();
            }
        }
        acc += theta_i.cos() * drive_a[i] - theta_i.sin() * drive_b[i];
        out[i] = theta_i + dt * acc;
    }
    Ok(PyArray1::from_owned_array(py, out))
}
