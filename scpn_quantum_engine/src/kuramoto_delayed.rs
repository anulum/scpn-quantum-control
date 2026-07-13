// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — kuramoto delayed Rust module
//! Rust tier for the time-delayed (method-of-steps) networked-Kuramoto forward trajectory.
//!
//! The coupling carries a finite propagation delay ``τ``, so each oscillator feels the *delayed*
//! phases of its neighbours while its own response is instantaneous:
//! ``θ̇_j(t) = ω_j + Σ_k K_jk sin(θ_k(t−τ) − θ_j(t))``. This is a delay-differential equation whose
//! state is the whole phase history on ``[−τ, 0]``; it is advanced by the **method of steps** with a
//! delay-aware fixed-step RK4. The running phase grid doubles as the history buffer, and the delayed
//! argument at a sub-stage time ``t + c·dt`` is read from the buffer at grid position ``t + c·dt − τ``
//! (linearly interpolated for the ``c = 1/2`` stages), which is always an already-computed sample
//! because ``τ`` is an integer number of steps. The delayed coupling reads two phase vectors, so it
//! owns its own force helper rather than reusing the single-vector networked force of the autodiff
//! module.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::kuramoto_common::validate_phase_vector;

/// Evaluate the delayed networked coupling force into ``out`` (length ``n``).
///
/// ``F_j = Σ_k K_jk sin(θ_k(t−τ) − θ_j(t))`` reads the delayed phases ``lagged`` for every neighbour
/// while the self-phase ``current[j]`` is instantaneous — the networked Kuramoto force evaluated
/// across the two phase vectors ``θ(t)`` and ``θ(t−τ)``.
fn networked_delayed_force_into(
    current: &[f64],
    lagged: &[f64],
    coupling_flat: &[f64],
    n: usize,
    out: &mut [f64],
) {
    for j in 0..n {
        let current_j = current[j];
        let row = &coupling_flat[j * n..j * n + n];
        let mut acc = 0.0_f64;
        for k in 0..n {
            acc += row[k] * (lagged[k] - current_j).sin();
        }
        out[j] = acc;
    }
}

/// Read the phase sample at fractional grid index ``position`` into ``out`` (linear interpolation).
///
/// ``buffer`` is the running phase grid flattened row-major, ``n`` phases per row; grid index ``g``
/// holds ``θ`` at time ``(g − delay_steps)·dt``. An integer ``position`` copies the row exactly (the
/// zero-weight branch, mirroring the Python floor); a fractional one blends the two bracketing rows,
/// which reproduces the half-step (``c = 1/2``) Runge–Kutta sub-stage reads.
fn lagged_into(buffer: &[f64], position: f64, n: usize, out: &mut [f64]) {
    let lower = position.floor() as usize;
    let weight = position - lower as f64;
    if weight == 0.0 {
        out.copy_from_slice(&buffer[lower * n..lower * n + n]);
    } else {
        let low_row = &buffer[lower * n..lower * n + n];
        let high_row = &buffer[(lower + 1) * n..(lower + 1) * n + n];
        for i in 0..n {
            out[i] = (1.0 - weight) * low_row[i] + weight * high_row[i];
        }
    }
}

/// Pure-Rust method-of-steps RK4 forward trajectory of the delayed networked Kuramoto flow.
///
/// ``history_flat`` supplies ``θ`` on ``[−τ, 0]`` as ``delay_steps + 1`` rows (row ``delay_steps`` is
/// ``θ(0)``) flattened row-major; the running grid is appended in place so each step's delayed reads
/// hit already-computed samples. Advances the phases by the classical four-stage Runge–Kutta rule
/// applied to ``θ̇(t) = ω + F(θ(t), θ(t−τ))``, mirroring the Python floor's arithmetic so the two agree
/// to the requested tolerance (the only difference is the coupling-force summation order — a scalar
/// loop here versus NumPy's vectorised reduction — so the tiers are tolerance-parity, ~1e-11, not
/// bit-identical). Returns the ``(M + 1,)`` sample times and the ``(M + 1, N)`` phases, ``M = n_steps``.
pub fn kuramoto_delayed_trajectory_inner(
    history_flat: &[f64],
    omega: &[f64],
    coupling_flat: &[f64],
    n: usize,
    delay_steps: usize,
    dt: f64,
    n_steps: usize,
) -> (Array1<f64>, Array2<f64>) {
    let mut buffer: Vec<f64> = Vec::with_capacity((delay_steps + 1 + n_steps) * n);
    buffer.extend_from_slice(history_flat);

    let mut theta = vec![0.0_f64; n];
    let mut lagged = vec![0.0_f64; n];
    let mut k1 = vec![0.0_f64; n];
    let mut k2 = vec![0.0_f64; n];
    let mut k3 = vec![0.0_f64; n];
    let mut k4 = vec![0.0_f64; n];
    let mut stage = vec![0.0_f64; n];
    let mut next = vec![0.0_f64; n];

    let mut phases_flat: Vec<f64> = Vec::with_capacity((n_steps + 1) * n);
    phases_flat.extend_from_slice(&buffer[delay_steps * n..delay_steps * n + n]);

    for step in 0..n_steps {
        let base = (delay_steps + step) * n;
        theta.copy_from_slice(&buffer[base..base + n]);
        let position = step as f64;

        // Delayed grid position for a sub-stage at t_n + c·dt is (step + c); see the module note.
        lagged_into(&buffer, position, n, &mut lagged);
        networked_delayed_force_into(&theta, &lagged, coupling_flat, n, &mut k1);
        for j in 0..n {
            k1[j] += omega[j];
            stage[j] = theta[j] + 0.5 * dt * k1[j];
        }
        lagged_into(&buffer, position + 0.5, n, &mut lagged);
        networked_delayed_force_into(&stage, &lagged, coupling_flat, n, &mut k2);
        for j in 0..n {
            k2[j] += omega[j];
            stage[j] = theta[j] + 0.5 * dt * k2[j];
        }
        lagged_into(&buffer, position + 0.5, n, &mut lagged);
        networked_delayed_force_into(&stage, &lagged, coupling_flat, n, &mut k3);
        for j in 0..n {
            k3[j] += omega[j];
            stage[j] = theta[j] + dt * k3[j];
        }
        lagged_into(&buffer, position + 1.0, n, &mut lagged);
        networked_delayed_force_into(&stage, &lagged, coupling_flat, n, &mut k4);
        for j in 0..n {
            k4[j] += omega[j];
            next[j] = theta[j] + (dt / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }
        buffer.extend_from_slice(&next);
        phases_flat.extend_from_slice(&next);
    }

    let times: Vec<f64> = (0..=n_steps).map(|step| dt * step as f64).collect();
    (
        Array1::from_vec(times),
        Array2::from_shape_vec((n_steps + 1, n), phases_flat)
            .expect("delayed phases buffer matches (M+1, n)"),
    )
}

/// Time-delayed method-of-steps RK4 forward trajectory of the networked Kuramoto flow (PyO3).
#[pyfunction]
pub fn kuramoto_delayed_trajectory<'py>(
    py: Python<'py>,
    initial_history: PyReadonlyArray2<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: i64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let frequencies = validate_phase_vector(&omega, "omega")?;
    let n = frequencies.len();
    let history = initial_history.as_array();
    if history.ncols() != n {
        return Err(PyValueError::new_err(format!(
            "initial_history must have {n} columns, got {}",
            history.ncols()
        )));
    }
    if history.nrows() < 2 {
        return Err(PyValueError::new_err(format!(
            "initial_history must have at least two rows (delay_steps + 1), got {}",
            history.nrows()
        )));
    }
    let delay_steps = history.nrows() - 1;
    let matrix = coupling.as_array();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "coupling must have shape ({n}, {n}), got {:?}",
            matrix.shape()
        )));
    }
    if dt <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "dt must be positive, got {dt}"
        )));
    }
    if n_steps < 1 {
        return Err(PyValueError::new_err(format!(
            "n_steps must be positive, got {n_steps}"
        )));
    }
    let history_flat: Vec<f64> = history
        .as_slice()
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| history.iter().copied().collect());
    let coupling_flat = matrix
        .as_slice()
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| matrix.iter().copied().collect());
    let (times, phases) = kuramoto_delayed_trajectory_inner(
        &history_flat,
        frequencies,
        &coupling_flat,
        n,
        delay_steps,
        dt,
        n_steps as usize,
    );
    Ok((
        PyArray1::from_owned_array(py, times),
        PyArray2::from_owned_array(py, phases),
    ))
}
