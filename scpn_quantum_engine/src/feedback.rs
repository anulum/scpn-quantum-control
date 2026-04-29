// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Real-Time Feedback Policy Kernel

//! Vectorised feedback policy updates for live-shot synchronisation control.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::validation::{validate_finite, validate_range};

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
) -> PyResult<(
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
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

    let r = r_values.as_slice().unwrap();
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
