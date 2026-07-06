// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! Rust tier for the symplectic (velocity-Verlet) inertial networked-Kuramoto forward trajectory.
//!
//! Structure-preserving integration of the swing equation ``m θ̈ + γ θ̇ = ω + F(θ)`` by a damped
//! velocity-Verlet (leapfrog) scheme with Strang splitting of the linear damping — its own
//! responsibility, distinct from the fixed-grid RK4 inertial integrator (bounded, not secular,
//! energy error). Reuses the networked coupling force from the autodiff module.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::kuramoto_autodiff::networked_force_into;
use crate::kuramoto_common::validate_phase_vector;

/// Pure Rust damped velocity-Verlet forward trajectory of the inertial networked flow.
///
/// Advances ``(θ, v)`` by a half-step exponential velocity decay ``e^{-γ dt/(2m)}``, a
/// velocity-Verlet kick–drift–kick of the Hamiltonian part with acceleration ``a(θ) = (ω + F(θ))/m``,
/// and a second half-step decay — the Strang splitting that is exactly symplectic at ``γ = 0``.
/// Mirrors the Python floor's arithmetic so the two agree to machine precision (the only
/// cross-language difference is the coupling-force summation order, so the tiers are
/// tolerance-parity, not guaranteed bit-identical). Samples ``θ`` and ``v`` at every step. Returns
/// the ``(M + 1,)`` times, the ``(M + 1, N)`` phases and the ``(M + 1, N)`` velocities.
#[allow(clippy::too_many_arguments)]
pub fn kuramoto_symplectic_inertial_trajectory_inner(
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
    let decay = (-damping * dt / (2.0 * mass)).exp();
    let mut position = theta0.to_vec();
    let mut momentum = v0.to_vec();
    let mut force_buf = vec![0.0_f64; n];

    let mut phases_flat: Vec<f64> = Vec::with_capacity((n_steps + 1) * n);
    let mut velocities_flat: Vec<f64> = Vec::with_capacity((n_steps + 1) * n);
    phases_flat.extend_from_slice(&position);
    velocities_flat.extend_from_slice(&momentum);

    for _ in 0..n_steps {
        for value in momentum.iter_mut() {
            *value *= decay;
        }
        networked_force_into(&position, coupling_flat, n, &mut force_buf);
        for j in 0..n {
            momentum[j] += 0.5 * dt * (omega[j] + force_buf[j]) / mass;
        }
        for j in 0..n {
            position[j] += dt * momentum[j];
        }
        networked_force_into(&position, coupling_flat, n, &mut force_buf);
        for j in 0..n {
            momentum[j] += 0.5 * dt * (omega[j] + force_buf[j]) / mass;
        }
        for value in momentum.iter_mut() {
            *value *= decay;
        }
        phases_flat.extend_from_slice(&position);
        velocities_flat.extend_from_slice(&momentum);
    }

    let times: Vec<f64> = (0..=n_steps).map(|step| dt * step as f64).collect();
    (
        Array1::from_vec(times),
        Array2::from_shape_vec((n_steps + 1, n), phases_flat)
            .expect("symplectic phases buffer matches (M+1, n)"),
        Array2::from_shape_vec((n_steps + 1, n), velocities_flat)
            .expect("symplectic velocities buffer matches (M+1, n)"),
    )
}

/// Symplectic (velocity-Verlet) inertial forward trajectory of the networked Kuramoto flow (PyO3).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn kuramoto_symplectic_inertial_trajectory<'py>(
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
        return Err(PyValueError::new_err(format!("dt must be positive, got {dt}")));
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
    let (times, phases, velocities_out) = kuramoto_symplectic_inertial_trajectory_inner(
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
