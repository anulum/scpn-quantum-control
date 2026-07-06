// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Real-Time Feedback Policy Kernel

//! Vectorised feedback policy updates for live-shot synchronisation control.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::Instant;

use crate::validation::{validate_contiguous_slice, validate_finite, validate_range};

type FeedbackPolicyBatchResult<'py> = PyResult<(
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)>;

type RealtimeFeedbackLoopResult<'py> = PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)>;

/// Convert measured order parameters into action codes, gains, and errors.
///
/// Action codes:
/// - `-1`: release coupling because synchrony is above the target band
/// - `0`: hold the current controller setting
/// - `1`: increase synchronising coupling because synchrony is below target
#[pyfunction]
pub fn feedback_policy_batch<'py>(
    py: Python<'py>,
    r_values: PyReadonlyArray1<'_, f64>,
    target_r: f64,
    deadband: f64,
    base_gain: f64,
    max_gain: f64,
) -> FeedbackPolicyBatchResult<'py> {
    validate_range(target_r, 0.0, 1.0, "target_r")?;
    validate_range(deadband, 0.0, 1.0, "deadband")?;
    if !base_gain.is_finite() || base_gain < 0.0 {
        return Err(PyValueError::new_err(format!(
            "base_gain must be finite and non-negative, got {base_gain}"
        )));
    }
    if !max_gain.is_finite() || max_gain < 1.0 {
        return Err(PyValueError::new_err(format!(
            "max_gain must be finite and at least 1.0, got {max_gain}"
        )));
    }

    let r = validate_contiguous_slice(&r_values, "r_values")?;
    validate_finite(r, "r_values")?;

    let mut actions = Vec::with_capacity(r.len());
    let mut gains = Vec::with_capacity(r.len());
    let mut errors = Vec::with_capacity(r.len());

    for &value in r {
        let (action, gain, error) =
            feedback_policy_inner(value, target_r, deadband, base_gain, max_gain);
        actions.push(action);
        gains.push(gain);
        errors.push(error);
    }

    Ok((
        PyArray1::from_vec(py, actions),
        PyArray1::from_vec(py, gains),
        PyArray1::from_vec(py, errors),
    ))
}

/// Execute the complete realtime control loop inside Rust.
///
/// This path keeps the full step loop (state update + order parameter +
/// feedback policy + coupling-scale update) on the Rust side so Python does not
/// participate in per-step scheduling.
#[pyfunction]
#[expect(
    clippy::too_many_arguments,
    reason = "public PyO3 ABI mirrors the Python realtime feedback wrapper"
)]
pub fn run_realtime_feedback_loop<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    target_r: f64,
    deadband: f64,
    base_gain: f64,
    max_gain: f64,
    dt: f64,
    n_steps: usize,
) -> RealtimeFeedbackLoopResult<'py> {
    validate_range(target_r, 0.0, 1.0, "target_r")?;
    validate_range(deadband, 0.0, 1.0, "deadband")?;
    if !base_gain.is_finite() || base_gain < 0.0 {
        return Err(PyValueError::new_err(format!(
            "base_gain must be finite and non-negative, got {base_gain}"
        )));
    }
    if !max_gain.is_finite() || max_gain < 1.0 {
        return Err(PyValueError::new_err(format!(
            "max_gain must be finite and at least 1.0, got {max_gain}"
        )));
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "dt must be finite and positive, got {dt}"
        )));
    }
    if n_steps < 1 {
        return Err(PyValueError::new_err("n_steps must be a positive integer"));
    }

    let theta0_slice = validate_contiguous_slice(&theta0, "theta0")?;
    validate_finite(theta0_slice, "theta0")?;
    let omega_slice = validate_contiguous_slice(&omega, "omega")?;
    validate_finite(omega_slice, "omega")?;
    let n = theta0_slice.len();
    if omega_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "omega must have length {n}, got {}",
            omega_slice.len()
        )));
    }
    let k_array = k.as_array();
    if k_array.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "k must have shape ({n}, {n}), got {:?}",
            k_array.shape()
        )));
    }
    let k_slice = k_array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("k must be a C-contiguous NumPy array"))?;
    validate_finite(k_slice, "k")?;

    let mut theta = theta0_slice.to_vec();
    let mut coupling_scale = 1.0_f64;
    let mut times = Vec::with_capacity(n_steps);
    let mut r_values = Vec::with_capacity(n_steps);
    let mut applied_scales = Vec::with_capacity(n_steps);
    let mut next_scales = Vec::with_capacity(n_steps);
    let mut actions = Vec::with_capacity(n_steps);
    let mut errors = Vec::with_capacity(n_steps);
    let mut tick_latency_ms = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let tick_started = Instant::now();
        let mut dtheta = vec![0.0_f64; n];
        for i in 0..n {
            let theta_i = theta[i];
            let mut value = omega_slice[i];
            let row_offset = i * n;
            for j in 0..n {
                let kij = coupling_scale * k_slice[row_offset + j];
                value += kij * (theta[j] - theta_i).sin();
            }
            dtheta[i] = value;
        }
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }

        let r_value = order_parameter_inner(&theta);
        let (action, gain, error) =
            feedback_policy_inner(r_value, target_r, deadband, base_gain, max_gain);
        let applied_scale = coupling_scale;
        coupling_scale = gain;

        times.push(step as f64 * dt);
        r_values.push(r_value);
        applied_scales.push(applied_scale);
        next_scales.push(coupling_scale);
        actions.push(action);
        errors.push(error);
        tick_latency_ms.push(tick_started.elapsed().as_secs_f64() * 1_000.0);
    }

    Ok((
        PyArray1::from_vec(py, times),
        PyArray1::from_vec(py, r_values),
        PyArray1::from_vec(py, applied_scales),
        PyArray1::from_vec(py, next_scales),
        PyArray1::from_vec(py, actions),
        PyArray1::from_vec(py, errors),
        PyArray1::from_vec(py, tick_latency_ms),
    ))
}

pub fn feedback_policy_inner(
    r_value: f64,
    target_r: f64,
    deadband: f64,
    base_gain: f64,
    max_gain: f64,
) -> (i32, f64, f64) {
    let error = target_r - r_value;
    if error.abs() <= deadband {
        return (0, 1.0, error);
    }

    if error > 0.0 {
        let gain = (1.0 + base_gain * error).min(max_gain);
        (1, gain, error)
    } else {
        let gain = (1.0 + base_gain * error).clamp(1.0 / max_gain, 1.0);
        (-1, gain, error)
    }
}

fn order_parameter_inner(theta: &[f64]) -> f64 {
    let n = theta.len() as f64;
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for &value in theta {
        re += value.cos();
        im += value.sin();
    }
    (re * re + im * im).sqrt() / n
}

#[cfg(test)]
mod tests {
    use super::feedback_policy_inner;

    #[test]
    fn low_synchrony_increases_gain() {
        let (action, gain, error) = feedback_policy_inner(0.25, 0.8, 0.05, 0.5, 2.0);
        assert_eq!(action, 1);
        assert!(error > 0.0);
        assert!(gain > 1.0);
    }

    #[test]
    fn target_band_holds_gain() {
        let (action, gain, error) = feedback_policy_inner(0.77, 0.8, 0.05, 0.5, 2.0);
        assert_eq!(action, 0);
        assert!(error.abs() <= 0.05);
        assert_eq!(gain, 1.0);
    }

    #[test]
    fn high_synchrony_releases_gain() {
        let (action, gain, error) = feedback_policy_inner(0.95, 0.8, 0.05, 0.5, 2.0);
        assert_eq!(action, -1);
        assert!(error < 0.0);
        assert!(gain < 1.0);
    }
}
