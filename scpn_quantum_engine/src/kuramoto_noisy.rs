// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! Rust tier for the stochastic (noisy) networked-Kuramoto Euler–Maruyama forward trajectory.
//!
//! Each oscillator obeys the Langevin equation ``dθ_j = (ω_j + F_j(θ)) dt + √(2D) dW_j`` with a
//! common diffusion (noise intensity) ``D`` and the instantaneous networked coupling force
//! ``F_j = Σ_k K_jk sin(θ_k − θ_j)``. The Euler–Maruyama step is
//! ``θ ← θ + (ω + F(θ)) dt + √(2 D dt) ξ`` with ``ξ`` a standard-normal increment. Cross-language
//! RNG reproduction is deliberately avoided: the caller pre-generates the whole ``(n_steps, N)``
//! standard-normal array (numpy's seeded PCG64) and passes it in, so every tier consumes the exact
//! same Wiener increments and only the coupling-force summation order differs. Reuses the
//! instantaneous networked force of the autodiff module — the drift is the ordinary Kuramoto force.

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::kuramoto_autodiff::networked_force_into;
use crate::kuramoto_common::validate_phase_vector;

/// Kuramoto order parameter ``r = |⟨e^{iθ}⟩| = √((Σcos)² + (Σsin)²) / N`` of a phase vector.
fn order_parameter(theta: &[f64]) -> f64 {
    let mut cos_sum = 0.0_f64;
    let mut sin_sum = 0.0_f64;
    for &value in theta {
        cos_sum += value.cos();
        sin_sum += value.sin();
    }
    (cos_sum * cos_sum + sin_sum * sin_sum).sqrt() / theta.len() as f64
}

/// Pure-Rust seeded Euler–Maruyama forward trajectory of the noisy networked Kuramoto flow.
///
/// Advances the phases by ``θ ← θ + (ω + F(θ)) dt + √(2 D dt) ξ`` for the supplied standard-normal
/// increments ``noise_flat`` (``n_steps`` rows of ``n`` flattened row-major), recording the order
/// parameter after every step. Mirrors the Python floor's arithmetic — including the
/// ``(θ + drift) + scale·ξ`` addition order — so the two agree to the requested tolerance (the only
/// difference is the coupling-force summation order, a scalar loop here versus NumPy's vectorised
/// reduction, so the tiers are tolerance-parity, not bit-identical). Returns the ``(n_steps,)`` order
/// parameter series and the ``(N,)`` terminal phases.
#[allow(clippy::too_many_arguments)]
pub fn kuramoto_noisy_trajectory_inner(
    theta0: &[f64],
    omega: &[f64],
    coupling_flat: &[f64],
    n: usize,
    diffusion: f64,
    dt: f64,
    noise_flat: &[f64],
    n_steps: usize,
) -> (Array1<f64>, Array1<f64>) {
    let scale = (2.0 * diffusion * dt).sqrt();
    let mut theta = theta0.to_vec();
    let mut force = vec![0.0_f64; n];
    let mut series: Vec<f64> = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        networked_force_into(&theta, coupling_flat, n, &mut force);
        let noise_row = &noise_flat[step * n..step * n + n];
        for j in 0..n {
            let drift = (omega[j] + force[j]) * dt;
            theta[j] = theta[j] + drift + scale * noise_row[j];
        }
        series.push(order_parameter(&theta));
    }
    (Array1::from_vec(series), Array1::from_vec(theta))
}

/// Stochastic (noisy) Euler–Maruyama forward trajectory of the networked Kuramoto flow (PyO3).
#[pyfunction]
pub fn kuramoto_noisy_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    diffusion: f64,
    dt: f64,
    noise: PyReadonlyArray2<'_, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let phases = validate_phase_vector(&theta0, "theta0")?;
    let frequencies = validate_phase_vector(&omega, "omega")?;
    let n = phases.len();
    if frequencies.len() != n {
        return Err(PyValueError::new_err(format!(
            "omega must have shape ({n},), got ({},)",
            frequencies.len()
        )));
    }
    let matrix = coupling.as_array();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "coupling must have shape ({n}, {n}), got {:?}",
            matrix.shape()
        )));
    }
    if diffusion < 0.0 {
        return Err(PyValueError::new_err(format!(
            "diffusion must be non-negative, got {diffusion}"
        )));
    }
    if dt <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "dt must be positive, got {dt}"
        )));
    }
    let noise_view = noise.as_array();
    if noise_view.ncols() != n {
        return Err(PyValueError::new_err(format!(
            "noise must have {n} columns, got {}",
            noise_view.ncols()
        )));
    }
    if noise_view.nrows() < 1 {
        return Err(PyValueError::new_err(format!(
            "noise must have at least one row (n_steps), got {}",
            noise_view.nrows()
        )));
    }
    let n_steps = noise_view.nrows();
    let coupling_flat = matrix
        .as_slice()
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| matrix.iter().copied().collect());
    let noise_flat: Vec<f64> = noise_view
        .as_slice()
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| noise_view.iter().copied().collect());
    let (series, terminal) = kuramoto_noisy_trajectory_inner(
        phases,
        frequencies,
        &coupling_flat,
        n,
        diffusion,
        dt,
        &noise_flat,
        n_steps,
    );
    Ok((
        PyArray1::from_owned_array(py, series),
        PyArray1::from_owned_array(py, terminal),
    ))
}
