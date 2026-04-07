// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Classical Kuramoto ODE Integration

//! Classical Kuramoto oscillator dynamics.
//!
//! Vectorised Euler integration of the coupled phase equation:
//! dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i)
//!
//! Includes order parameter R = (1/N)|��_i exp(iθ_i)| computation
//! and full trajectory recording.

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::validation::validate_positive;

/// Classical Kuramoto ODE step (vectorised, no Python overhead).
/// θ' = ω + K @ sin(��_outer − θ_inner)
/// Returns new θ after n_steps of Euler integration.
#[pyfunction]
pub fn kuramoto_euler<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_positive(dt, "dt")?;
    let mut theta = theta0.as_array().to_owned();
    let omega = omega.as_array();
    let k = k.as_array();
    let n = theta.len();

    for _ in 0..n_steps {
        let mut dtheta = Array1::<f64>::zeros(n);
        for i in 0..n {
            dtheta[i] = omega[i];
            for j in 0..n {
                dtheta[i] += k[[i, j]] * (theta[j] - theta[i]).sin();
            }
        }
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }
    }

    Ok(PyArray1::from_owned_array(py, theta))
}

/// Compute Kuramoto order parameter R from phase array.
/// R = (1/N) |Σ_i exp(i × θ_i)|
#[pyfunction]
pub fn order_parameter(theta: PyReadonlyArray1<'_, f64>) -> f64 {
    let theta = theta.as_array();
    order_parameter_inner(theta.as_slice().unwrap())
}

/// Pure Rust order parameter (no PyO3).
pub fn order_parameter_inner(theta: &[f64]) -> f64 {
    let n = theta.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for &t in theta {
        re += t.cos();
        im += t.sin();
    }
    (re * re + im * im).sqrt() / n
}

/// Parallel classical Kuramoto trajectory.
/// Returns (times, R_values) for each timestep.
#[pyfunction]
pub fn kuramoto_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validate_positive(dt, "dt")?;
    let mut theta = theta0.as_array().to_owned();
    let omega_arr = omega.as_array();
    let k_arr = k.as_array();
    let n = theta.len();

    let mut times = Array1::<f64>::zeros(n_steps + 1);
    let mut r_values = Array1::<f64>::zeros(n_steps + 1);

    // Initial R
    r_values[0] = order_parameter_inner(theta.as_slice().unwrap());

    for step in 0..n_steps {
        let mut dtheta = Array1::<f64>::zeros(n);
        for i in 0..n {
            dtheta[i] = omega_arr[i];
            for j in 0..n {
                dtheta[i] += k_arr[[i, j]] * (theta[j] - theta[i]).sin();
            }
        }
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }

        times[step + 1] = (step + 1) as f64 * dt;
        r_values[step + 1] = order_parameter_inner(theta.as_slice().unwrap());
    }

    Ok((
        PyArray1::from_owned_array(py, times),
        PyArray1::from_owned_array(py, r_values),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_parameter_all_aligned() {
        let theta = vec![0.0; 8];
        let r = order_parameter_inner(&theta);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_order_parameter_opposite() {
        let theta = vec![0.0, std::f64::consts::PI];
        let r = order_parameter_inner(&theta);
        assert!(r < 1e-10);
    }

    #[test]
    fn test_order_parameter_bounded() {
        let theta = vec![0.1, 0.5, 1.2, 2.5, 3.8, 5.0];
        let r = order_parameter_inner(&theta);
        assert!((0.0..=1.0 + 1e-12).contains(&r));
    }

    #[test]
    fn test_order_parameter_single() {
        let theta = vec![2.7];
        let r = order_parameter_inner(&theta);
        assert!((r - 1.0).abs() < 1e-12, "single oscillator → R = 1");
    }

    #[test]
    fn test_order_parameter_uniform_circle() {
        // N equally spaced phases → R ≈ 0
        let n = 100;
        let theta: Vec<f64> = (0..n)
            .map(|i| 2.0 * std::f64::consts::PI * i as f64 / n as f64)
            .collect();
        let r = order_parameter_inner(&theta);
        assert!(r < 0.05, "uniform circle → R ≈ 0, got {r}");
    }
}
