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

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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

fn validate_kuramoto_shapes(n: usize, omega_len: usize, k_shape: &[usize]) -> PyResult<()> {
    if omega_len != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "omega must have length {n}, got {omega_len}"
        )));
    }
    if k_shape != [n, n] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "k must have shape ({n}, {n}), got {k_shape:?}"
        )));
    }
    Ok(())
}

fn validate_finite_slice(values: &[f64], name: &str) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

fn phase_step_pairwise(theta: &[f64], omega: &[f64], k_flat: &[f64], n: usize) -> Vec<f64> {
    let mut dtheta = vec![0.0; n];
    for i in 0..n {
        dtheta[i] = omega[i];
        for j in 0..n {
            dtheta[i] += k_flat[i * n + j] * (theta[j] - theta[i]).sin();
        }
    }
    dtheta
}

fn fill_time_and_r(
    times: &mut Array1<f64>,
    r_values: &mut Array1<f64>,
    step: usize,
    dt: f64,
    theta: &[f64],
) {
    times[step] = step as f64 * dt;
    r_values[step] = order_parameter_inner(theta);
}

/// Higher-order simplicial Kuramoto trajectory.
///
/// Pairwise term: K_ij sin(theta_j - theta_i).
/// Anchored triadic term: B_a sin(theta_j + theta_k - 2 theta_i) for each
/// hyperedge row (i, j, k).
#[pyfunction]
pub fn higher_order_kuramoto_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    hyperedges: PyReadonlyArray2<'_, i64>,
    hyper_weights: PyReadonlyArray1<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    validate_positive(dt, "dt")?;
    let mut theta = theta0.as_array().to_owned();
    let n = theta.len();
    let omega_arr = omega.as_array();
    let k_arr = k.as_array();
    validate_kuramoto_shapes(n, omega_arr.len(), k_arr.shape())?;
    validate_finite_slice(theta.as_slice().unwrap(), "theta0")?;
    validate_finite_slice(omega_arr.as_slice().unwrap(), "omega")?;
    validate_finite_slice(k_arr.as_slice().unwrap(), "k")?;

    let edge_arr = hyperedges.as_array();
    let weights = hyper_weights.as_array();
    if edge_arr.ndim() != 2 || edge_arr.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "hyperedges must have shape (n_edges, 3)",
        ));
    }
    if weights.len() != edge_arr.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hyper_weights must have length {}, got {}",
            edge_arr.shape()[0],
            weights.len()
        )));
    }
    validate_finite_slice(weights.as_slice().unwrap(), "hyper_weights")?;
    for edge in edge_arr.rows() {
        for &index in edge.iter() {
            if index < 0 || index as usize >= n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "hyperedge index {index} is outside [0, {n})"
                )));
            }
        }
    }

    let k_flat = k_arr.as_slice().unwrap();
    let omega_slice = omega_arr.as_slice().unwrap();
    let edge_slice = edge_arr.as_slice().unwrap();
    let weight_slice = weights.as_slice().unwrap();
    let mut times = Array1::<f64>::zeros(n_steps + 1);
    let mut r_values = Array1::<f64>::zeros(n_steps + 1);
    fill_time_and_r(&mut times, &mut r_values, 0, dt, theta.as_slice().unwrap());

    for step in 0..n_steps {
        let theta_slice = theta.as_slice().unwrap();
        let mut dtheta = phase_step_pairwise(theta_slice, omega_slice, k_flat, n);
        for edge_index in 0..weight_slice.len() {
            let base = 3 * edge_index;
            let i = edge_slice[base] as usize;
            let j = edge_slice[base + 1] as usize;
            let l = edge_slice[base + 2] as usize;
            dtheta[i] += weight_slice[edge_index]
                * (theta_slice[j] + theta_slice[l] - 2.0 * theta_slice[i]).sin();
        }
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }
        fill_time_and_r(
            &mut times,
            &mut r_values,
            step + 1,
            dt,
            theta.as_slice().unwrap(),
        );
    }

    Ok((
        PyArray1::from_owned_array(py, times),
        PyArray1::from_owned_array(py, r_values),
    ))
}

/// Monitored Kuramoto trajectory with deterministic measurement-feedback closure.
///
/// The instantaneous readout is R_m = (1-strength)R + strength*target_R.  The
/// feedback term g(target_R - R_m) sin(psi - theta_i) pulls phases toward the
/// measured mean phase without changing the pairwise Kuramoto law.
#[pyfunction]
pub fn monitored_kuramoto_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    target_r: f64,
    monitor_gain: f64,
    measurement_strength: f64,
    dt: f64,
    n_steps: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validate_positive(dt, "dt")?;
    if !target_r.is_finite() || !(0.0..=1.0).contains(&target_r) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_r must be finite and in [0, 1]",
        ));
    }
    if !monitor_gain.is_finite() || monitor_gain < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "monitor_gain must be finite and non-negative",
        ));
    }
    if !measurement_strength.is_finite() || !(0.0..=1.0).contains(&measurement_strength) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "measurement_strength must be finite and in [0, 1]",
        ));
    }

    let mut theta = theta0.as_array().to_owned();
    let n = theta.len();
    let omega_arr = omega.as_array();
    let k_arr = k.as_array();
    validate_kuramoto_shapes(n, omega_arr.len(), k_arr.shape())?;
    validate_finite_slice(theta.as_slice().unwrap(), "theta0")?;
    validate_finite_slice(omega_arr.as_slice().unwrap(), "omega")?;
    validate_finite_slice(k_arr.as_slice().unwrap(), "k")?;

    let k_flat = k_arr.as_slice().unwrap();
    let omega_slice = omega_arr.as_slice().unwrap();
    let mut times = Array1::<f64>::zeros(n_steps + 1);
    let mut r_values = Array1::<f64>::zeros(n_steps + 1);
    let mut readouts = Array1::<f64>::zeros(n_steps + 1);
    let mut feedback = Array1::<f64>::zeros(n_steps + 1);

    for step in 0..=n_steps {
        let theta_slice = theta.as_slice().unwrap();
        let r = order_parameter_inner(theta_slice);
        let readout = (1.0 - measurement_strength) * r + measurement_strength * target_r;
        times[step] = step as f64 * dt;
        r_values[step] = r;
        readouts[step] = readout;
        feedback[step] = monitor_gain * (target_r - readout);
        if step == n_steps {
            break;
        }
        let (mut re, mut im) = (0.0, 0.0);
        for &phase in theta_slice {
            re += phase.cos();
            im += phase.sin();
        }
        let mean_phase = im.atan2(re);
        let mut dtheta = phase_step_pairwise(theta_slice, omega_slice, k_flat, n);
        for i in 0..n {
            dtheta[i] += feedback[step] * (mean_phase - theta_slice[i]).sin();
        }
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }
    }

    Ok((
        PyArray1::from_owned_array(py, times),
        PyArray1::from_owned_array(py, r_values),
        PyArray1::from_owned_array(py, readouts),
        PyArray1::from_owned_array(py, feedback),
    ))
}

/// PT-symmetric complex Kuramoto trajectory with balanced gain/loss.
///
/// The complex oscillator evolves as z_i' = (gain_i + i dtheta_i) z_i and is
/// renormalised after each Euler step so the returned R isolates phase locking
/// while pt_norm and imbalance expose the non-Hermitian gain/loss channel.
#[pyfunction]
pub fn pt_symmetric_kuramoto_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    gain_loss: PyReadonlyArray1<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    validate_positive(dt, "dt")?;
    let theta_arr = theta0.as_array();
    let n = theta_arr.len();
    let omega_arr = omega.as_array();
    let k_arr = k.as_array();
    let gain_arr = gain_loss.as_array();
    validate_kuramoto_shapes(n, omega_arr.len(), k_arr.shape())?;
    if gain_arr.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gain_loss must have length {n}, got {}",
            gain_arr.len()
        )));
    }
    validate_finite_slice(theta_arr.as_slice().unwrap(), "theta0")?;
    validate_finite_slice(omega_arr.as_slice().unwrap(), "omega")?;
    validate_finite_slice(k_arr.as_slice().unwrap(), "k")?;
    validate_finite_slice(gain_arr.as_slice().unwrap(), "gain_loss")?;
    let gain_sum: f64 = gain_arr.iter().sum();
    if gain_sum.abs() > 1e-10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "gain_loss must sum to zero for balanced PT symmetry",
        ));
    }

    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for &phase in theta_arr.iter() {
        re.push(phase.cos());
        im.push(phase.sin());
    }
    let k_flat = k_arr.as_slice().unwrap();
    let omega_slice = omega_arr.as_slice().unwrap();
    let gain_slice = gain_arr.as_slice().unwrap();
    let mut theta = theta_arr.to_owned();
    let mut times = Array1::<f64>::zeros(n_steps + 1);
    let mut r_values = Array1::<f64>::zeros(n_steps + 1);
    let mut pt_norm = Array1::<f64>::zeros(n_steps + 1);
    let mut imbalance = Array1::<f64>::zeros(n_steps + 1);

    for step in 0..=n_steps {
        let mut total_re = 0.0;
        let mut total_im = 0.0;
        let mut norm = 0.0;
        let mut signed_power = 0.0;
        for i in 0..n {
            total_re += re[i];
            total_im += im[i];
            let power = re[i] * re[i] + im[i] * im[i];
            norm += power;
            signed_power += gain_slice[i] * power;
            theta[i] = im[i].atan2(re[i]);
        }
        times[step] = step as f64 * dt;
        r_values[step] =
            (total_re * total_re + total_im * total_im).sqrt() / norm.sqrt() / (n as f64).sqrt();
        pt_norm[step] = norm / n as f64;
        imbalance[step] = signed_power;
        if step == n_steps {
            break;
        }
        let dtheta = phase_step_pairwise(theta.as_slice().unwrap(), omega_slice, k_flat, n);
        for i in 0..n {
            let z_re = re[i];
            let z_im = im[i];
            let gain = gain_slice[i];
            let freq = dtheta[i];
            re[i] += dt * (gain * z_re - freq * z_im);
            im[i] += dt * (freq * z_re + gain * z_im);
        }
        let norm_after = re
            .iter()
            .zip(im.iter())
            .map(|(x, y)| x * x + y * y)
            .sum::<f64>()
            .sqrt();
        if norm_after > 0.0 {
            let scale = (n as f64).sqrt() / norm_after;
            for i in 0..n {
                re[i] *= scale;
                im[i] *= scale;
            }
        }
    }

    Ok((
        PyArray1::from_owned_array(py, times),
        PyArray1::from_owned_array(py, r_values),
        PyArray1::from_owned_array(py, pt_norm),
        PyArray1::from_owned_array(py, imbalance),
    ))
}

/// Batch candidate features for automated Kuramoto witness discovery.
///
/// Candidate columns are `(coupling_scale, omega_scale, phase_bias)`.  The
/// kernel integrates the classical Kuramoto dynamics and returns final R,
/// mean pairwise cos(theta_i-theta_j), and final phases for downstream witness
/// scoring in Python.
#[pyfunction]
pub fn kuramoto_witness_candidate_features<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    candidates: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    validate_positive(dt, "dt")?;
    let theta0_arr = theta0.as_array();
    let omega_arr = omega.as_array();
    let k_arr = k.as_array();
    let cand_arr = candidates.as_array();
    let n = theta0_arr.len();
    validate_kuramoto_shapes(n, omega_arr.len(), k_arr.shape())?;
    if cand_arr.ndim() != 2 || cand_arr.shape()[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "candidates must have shape (n_candidates, 3)",
        ));
    }
    validate_finite_slice(theta0_arr.as_slice().unwrap(), "theta0")?;
    validate_finite_slice(omega_arr.as_slice().unwrap(), "omega")?;
    validate_finite_slice(k_arr.as_slice().unwrap(), "k")?;
    validate_finite_slice(cand_arr.as_slice().unwrap(), "candidates")?;

    let n_candidates = cand_arr.shape()[0];
    let mut final_r = Array1::<f64>::zeros(n_candidates);
    let mut mean_corr = Array1::<f64>::zeros(n_candidates);
    let mut final_theta = Array2::<f64>::zeros((n_candidates, n));
    let omega_slice = omega_arr.as_slice().unwrap();
    let k_flat = k_arr.as_slice().unwrap();

    for (candidate_index, candidate) in cand_arr.rows().into_iter().enumerate() {
        let coupling_scale = candidate[0];
        let omega_scale = candidate[1];
        let phase_bias = candidate[2];
        if coupling_scale < 0.0 || omega_scale < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "candidate coupling_scale and omega_scale must be non-negative",
            ));
        }
        let mut theta: Vec<f64> = theta0_arr.iter().map(|value| value + phase_bias).collect();
        let omega_scaled: Vec<f64> = omega_slice
            .iter()
            .map(|value| value * omega_scale)
            .collect();
        let k_scaled: Vec<f64> = k_flat.iter().map(|value| value * coupling_scale).collect();

        for _ in 0..n_steps {
            let dtheta = phase_step_pairwise(&theta, &omega_scaled, &k_scaled, n);
            for i in 0..n {
                theta[i] += dt * dtheta[i];
            }
        }

        final_r[candidate_index] = order_parameter_inner(&theta);
        let mut corr_sum = 0.0;
        let mut n_pairs = 0usize;
        for i in 0..n {
            final_theta[[candidate_index, i]] = theta[i];
            for j in (i + 1)..n {
                corr_sum += (theta[i] - theta[j]).cos();
                n_pairs += 1;
            }
        }
        mean_corr[candidate_index] = if n_pairs > 0 {
            corr_sum / n_pairs as f64
        } else {
            1.0
        };
    }

    Ok((
        PyArray1::from_owned_array(py, final_r),
        PyArray1::from_owned_array(py, mean_corr),
        PyArray2::from_owned_array(py, final_theta),
    ))
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

    #[test]
    fn test_phase_step_pairwise_uses_all_couplings() {
        let theta = vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI];
        let omega = vec![0.1, -0.2, 0.3];
        let k = vec![0.0, 0.5, 0.25, 0.5, 0.0, 0.75, 0.25, 0.75, 0.0];

        let dtheta = phase_step_pairwise(&theta, &omega, &k, 3);

        assert!((dtheta[0] - 0.6).abs() < 1e-12);
        assert!((dtheta[1] - 0.05).abs() < 1e-12);
        assert!((dtheta[2] - (-0.45)).abs() < 1e-12);
    }

    #[test]
    fn test_fill_time_and_r_records_grid_point() {
        let mut times = Array1::<f64>::zeros(2);
        let mut r_values = Array1::<f64>::zeros(2);
        let theta = vec![0.0, 0.0, std::f64::consts::PI];

        fill_time_and_r(&mut times, &mut r_values, 1, 0.125, &theta);

        assert!((times[1] - 0.125).abs() < 1e-12);
        assert!((r_values[1] - (1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_candidate_feature_mean_corr_bounds() {
        let theta: Vec<f64> = vec![0.0, 0.1, 0.2, 0.3];
        let mut corr_sum = 0.0_f64;
        let mut n_pairs = 0usize;
        for i in 0..theta.len() {
            for j in (i + 1)..theta.len() {
                corr_sum += (theta[i] - theta[j]).cos();
                n_pairs += 1;
            }
        }
        let mean_corr = corr_sum / n_pairs as f64;
        assert!((-1.0..=1.0).contains(&mean_corr));
        assert!(mean_corr > 0.95);
    }
}
