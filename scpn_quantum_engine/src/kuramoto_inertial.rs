// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — kuramoto inertial Rust module
//! Rust tier for the inertial (second-order / swing-equation) networked-Kuramoto forward trajectory.
//!
//! Separate from the first-order fixed-grid RK4/Euler autodiff module: the second-order swing
//! equation ``m θ̈ + γ θ̇ = ω + F(θ)`` integrates a ``(θ, v)`` phase-space state, its own
//! responsibility. Reuses the networked coupling force from the autodiff module.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::kuramoto_autodiff::networked_force_into;
use crate::kuramoto_common::validate_phase_vector;

/// Evaluate the inertial phase-space vector field ``[θ̇, v̇]`` into ``out`` (length ``2n``).
///
/// The concatenated state is ``[θ (n), v (n)]``; the swing equation gives ``θ̇ = v`` and
/// ``v̇ = (ω + F(θ) − γ v) / m`` with ``F`` the networked coupling force. ``force_buf`` is a
/// scratch buffer of length ``n`` reused across the four Runge–Kutta stages to avoid allocation.
#[allow(clippy::too_many_arguments)]
fn inertial_field_into(
    state: &[f64],
    omega: &[f64],
    coupling_flat: &[f64],
    n: usize,
    mass: f64,
    damping: f64,
    force_buf: &mut [f64],
    out: &mut [f64],
) {
    networked_force_into(&state[..n], coupling_flat, n, force_buf);
    out[..n].copy_from_slice(&state[n..]);
    for j in 0..n {
        out[n + j] = (omega[j] + force_buf[j] - damping * state[n + j]) / mass;
    }
}

/// Pure Rust fixed-step RK4 forward trajectory of the inertial (second-order) networked flow.
///
/// Advances the concatenated ``[θ, v]`` state by the classical four-stage Runge–Kutta rule applied
/// to [`inertial_field_into`], mirroring the Python floor's arithmetic so the two agree to the
/// requested tolerance (the only difference is the coupling-force summation order — a scalar loop
/// here versus NumPy's vectorised reduction — so the tiers are tolerance-parity, ~1e-11, not
/// bit-identical). Samples ``θ`` and ``v`` at every step. Returns the ``(M + 1,)`` sample times,
/// the ``(M + 1, N)`` phases and the ``(M + 1, N)`` velocities, with ``M = n_steps``.
#[allow(clippy::too_many_arguments)]
pub fn kuramoto_inertial_trajectory_inner(
    theta0: &[f64],
    v0: &[f64],
    omega: &[f64],
    coupling_flat: &[f64],
    n: usize,
    mass: f64,
    damping: f64,
    dt: f64,
    n_steps: usize,
) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    let dim = 2 * n;
    let mut current = vec![0.0_f64; dim];
    current[..n].copy_from_slice(theta0);
    current[n..].copy_from_slice(v0);

    let mut k1 = vec![0.0_f64; dim];
    let mut k2 = vec![0.0_f64; dim];
    let mut k3 = vec![0.0_f64; dim];
    let mut k4 = vec![0.0_f64; dim];
    let mut stage = vec![0.0_f64; dim];
    let mut force_buf = vec![0.0_f64; n];

    let mut phases_flat: Vec<f64> = Vec::with_capacity((n_steps + 1) * n);
    let mut velocities_flat: Vec<f64> = Vec::with_capacity((n_steps + 1) * n);
    phases_flat.extend_from_slice(&current[..n]);
    velocities_flat.extend_from_slice(&current[n..]);

    for _ in 0..n_steps {
        inertial_field_into(
            &current,
            omega,
            coupling_flat,
            n,
            mass,
            damping,
            &mut force_buf,
            &mut k1,
        );
        for i in 0..dim {
            stage[i] = current[i] + 0.5 * dt * k1[i];
        }
        inertial_field_into(
            &stage,
            omega,
            coupling_flat,
            n,
            mass,
            damping,
            &mut force_buf,
            &mut k2,
        );
        for i in 0..dim {
            stage[i] = current[i] + 0.5 * dt * k2[i];
        }
        inertial_field_into(
            &stage,
            omega,
            coupling_flat,
            n,
            mass,
            damping,
            &mut force_buf,
            &mut k3,
        );
        for i in 0..dim {
            stage[i] = current[i] + dt * k3[i];
        }
        inertial_field_into(
            &stage,
            omega,
            coupling_flat,
            n,
            mass,
            damping,
            &mut force_buf,
            &mut k4,
        );
        for i in 0..dim {
            current[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        phases_flat.extend_from_slice(&current[..n]);
        velocities_flat.extend_from_slice(&current[n..]);
    }

    let times: Vec<f64> = (0..=n_steps).map(|step| dt * step as f64).collect();
    (
        Array1::from_vec(times),
        Array2::from_shape_vec((n_steps + 1, n), phases_flat)
            .expect("inertial phases buffer matches (M+1, n)"),
        Array2::from_shape_vec((n_steps + 1, n), velocities_flat)
            .expect("inertial velocities buffer matches (M+1, n)"),
    )
}

/// Inertial (second-order) fixed-step RK4 forward trajectory of the networked Kuramoto flow (PyO3).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn kuramoto_inertial_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    velocities: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    mass: f64,
    damping: f64,
    dt: f64,
    n_steps: i64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let theta0 = validate_phase_vector(&theta0, "theta0")?;
    let speed = validate_phase_vector(&velocities, "velocities")?;
    let frequencies = validate_phase_vector(&omega, "omega")?;
    let matrix = coupling.as_array();
    let n = theta0.len();
    if speed.len() != n {
        return Err(PyValueError::new_err(format!(
            "velocities must have shape ({n},), got ({},)",
            speed.len()
        )));
    }
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
    if mass <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "mass must be positive, got {mass}"
        )));
    }
    if damping < 0.0 {
        return Err(PyValueError::new_err(format!(
            "damping must be non-negative, got {damping}"
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
    let coupling_flat = matrix
        .as_slice()
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| matrix.iter().copied().collect());
    let (times, phases, velocities_out) = kuramoto_inertial_trajectory_inner(
        theta0,
        speed,
        frequencies,
        &coupling_flat,
        n,
        mass,
        damping,
        dt,
        n_steps as usize,
    );
    Ok((
        PyArray1::from_owned_array(py, times),
        PyArray2::from_owned_array(py, phases),
        PyArray2::from_owned_array(py, velocities_out),
    ))
}
