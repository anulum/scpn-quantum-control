// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! Rust tier for the adaptive Dormand–Prince (DOPRI5) networked-Kuramoto forward trajectory.
//!
//! Separate from the fixed-grid RK4/Euler autodiff module: adaptive, error-controlled integration
//! is its own responsibility. Reuses the networked coupling force from that module.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::kuramoto_autodiff::networked_force_into;
use crate::kuramoto_common::validate_phase_vector;

/// Pure Rust adaptive Dormand–Prince (DOPRI5) forward trajectory of the networked phase flow.
///
/// Error-controlled embedded 4/5 pair with the standard elementary step controller, mirroring the
/// Python floor (including its `t_end / 100` initial-step guess) so the two agree to the requested
/// tolerance and walk the same realised grid on well-conditioned problems — an adaptive scheme's
/// grid is tolerance-parity, not bit-identical, because marginal accept/reject decisions depend on
/// last-ULP float ordering. Returns the accepted times, the phases at those times (row-major
/// `(M + 1, N)`) and the realised step sizes.
#[allow(clippy::too_many_arguments)]
pub fn kuramoto_dopri_trajectory_inner(
    theta0: &[f64],
    omega: &[f64],
    coupling_flat: &[f64],
    n: usize,
    t_end: f64,
    rtol: f64,
    atol: f64,
    safety: f64,
    min_factor: f64,
    max_factor: f64,
    max_steps: usize,
) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
    const TABLEAU: [[f64; 6]; 7] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
        [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0],
        [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0],
        [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
    ];
    const FIFTH: [f64; 7] = [
        35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0,
    ];
    const ERROR: [f64; 7] = [
        35.0 / 384.0 - 5179.0 / 57600.0,
        0.0,
        500.0 / 1113.0 - 7571.0 / 16695.0,
        125.0 / 192.0 - 393.0 / 640.0,
        -2187.0 / 6784.0 + 92097.0 / 339200.0,
        11.0 / 84.0 - 187.0 / 2100.0,
        -1.0 / 40.0,
    ];
    let mut y = theta0.to_vec();
    let mut deriv: Vec<Vec<f64>> = (0..7).map(|_| vec![0.0_f64; n]).collect();
    let mut stage = vec![0.0_f64; n];
    let mut proposed = vec![0.0_f64; n];
    networked_force_into(&y, coupling_flat, n, &mut deriv[0]);
    for j in 0..n {
        deriv[0][j] += omega[j];
    }
    // Initial step mirrors the Python floor's ``t_end / 100`` guess so the adaptive controller
    // walks the same realised grid on well-conditioned problems.
    let mut step = t_end / 100.0;
    let mut time = 0.0_f64;
    let mut times = vec![0.0_f64];
    let mut phases_flat = y.clone();
    let mut steps: Vec<f64> = Vec::new();
    let mut accepted = 0usize;
    while time < t_end && accepted < max_steps {
        if step > t_end - time {
            step = t_end - time;
        }
        for s in 1..7 {
            for i in 0..n {
                let mut acc = y[i];
                for (p, deriv_p) in deriv.iter().enumerate().take(s) {
                    acc += step * TABLEAU[s][p] * deriv_p[i];
                }
                stage[i] = acc;
            }
            networked_force_into(&stage, coupling_flat, n, &mut deriv[s]);
            for j in 0..n {
                deriv[s][j] += omega[j];
            }
        }
        let mut error = 0.0_f64;
        for i in 0..n {
            let mut fifth = 0.0_f64;
            let mut embedded = 0.0_f64;
            for s in 0..7 {
                fifth += FIFTH[s] * deriv[s][i];
                embedded += ERROR[s] * deriv[s][i];
            }
            proposed[i] = y[i] + step * fifth;
            let scale = atol + rtol * y[i].abs().max(proposed[i].abs());
            let scaled = (step * embedded) / scale;
            error += scaled * scaled;
        }
        error = (error / n as f64).sqrt();
        if error <= 1.0 {
            time += step;
            y.copy_from_slice(&proposed);
            for j in 0..n {
                deriv[0][j] = deriv[6][j];
            }
            times.push(time);
            phases_flat.extend_from_slice(&y);
            steps.push(step);
            accepted += 1;
        }
        let factor = if error == 0.0 {
            max_factor
        } else {
            safety * error.powf(-0.2)
        };
        step *= factor.clamp(min_factor, max_factor);
    }
    (
        Array1::from_vec(times),
        Array2::from_shape_vec((accepted + 1, n), phases_flat)
            .expect("dopri phases buffer matches (M+1, n)"),
        Array1::from_vec(steps),
    )
}

/// Adaptive Dormand–Prince (DOPRI5) forward trajectory of the networked Kuramoto flow (PyO3 tier).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn kuramoto_dopri_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    t_end: f64,
    rtol: f64,
    atol: f64,
    safety: f64,
    min_factor: f64,
    max_factor: f64,
    max_steps: i64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let theta0 = validate_phase_vector(&theta0, "theta0")?;
    let frequencies = validate_phase_vector(&omega, "omega")?;
    let matrix = coupling.as_array();
    let n = theta0.len();
    if frequencies.len() != n {
        return Err(PyValueError::new_err(format!(
            "omega must have shape ({n},), got ({},)",
            frequencies.len()
        )));
    }
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "coupling must have shape ({n}, {n}), got {:?}",
            matrix.shape()
        )));
    }
    if t_end < 0.0 {
        return Err(PyValueError::new_err(format!(
            "t_end must be non-negative, got {t_end}"
        )));
    }
    if max_steps < 1 {
        return Err(PyValueError::new_err(format!(
            "max_steps must be positive, got {max_steps}"
        )));
    }
    let coupling_flat = matrix
        .as_slice()
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| matrix.iter().copied().collect());
    let (times, phases, steps) = kuramoto_dopri_trajectory_inner(
        theta0,
        frequencies,
        &coupling_flat,
        n,
        t_end,
        rtol,
        atol,
        safety,
        min_factor,
        max_factor,
        max_steps as usize,
    );
    Ok((
        PyArray1::from_owned_array(py, times),
        PyArray2::from_owned_array(py, phases),
        PyArray1::from_owned_array(py, steps),
    ))
}
