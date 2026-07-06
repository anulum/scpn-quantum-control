// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Differentiable networked-Kuramoto Euler integrator and adjoint
//! Forward explicit-Euler integration of the networked Kuramoto dynamics with a full-trajectory
//! record, and the reverse-mode adjoint that turns the simulation into a differentiable program:
//! gradients of a terminal objective with respect to the initial phases, natural frequencies and
//! coupling matrix flow back through the dynamics via the networked stability Jacobian.

use ndarray::{Array1, Array2, ArrayView2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::kuramoto_common::validate_phase_vector;

type KuramotoGradientResult<'py> = PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
)>;

/// Compute the forward networked-Kuramoto Euler trajectory θ_{n+1} = θ_n + dt (ω + F(θ_n)).
///
/// F_j(θ) = Σ_k K_jk sin(θ_k − θ_j). Returns the (n_steps + 1, N) phase trajectory, row 0 being
/// the initial state.
#[pyfunction]
pub fn kuramoto_euler_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: i64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
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
    if n_steps < 0 {
        return Err(PyValueError::new_err(format!(
            "n_steps must be non-negative, got {n_steps}"
        )));
    }
    Ok(PyArray2::from_owned_array(
        py,
        kuramoto_euler_trajectory_inner(theta0, frequencies, &matrix, dt, n_steps as usize),
    ))
}

/// Pure Rust forward Euler trajectory (no PyO3), returned row-major as (n_steps + 1, N).
///
/// The coupling force is computed inline into the trajectory row each step (no per-step
/// allocation), so the hot loop stays allocation-free.
pub fn kuramoto_euler_trajectory_inner(
    theta0: &[f64],
    omega: &[f64],
    coupling: &ArrayView2<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> Array2<f64> {
    let n = theta0.len();
    let coupling_flat = coupling
        .as_slice()
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| coupling.iter().copied().collect());
    let mut flat = vec![0.0_f64; (n_steps + 1) * n];
    flat[..n].copy_from_slice(theta0);
    let mut current = theta0.to_vec();
    for step in 0..n_steps {
        let base = (step + 1) * n;
        for j in 0..n {
            let theta_j = current[j];
            let row = &coupling_flat[j * n..j * n + n];
            let mut force = 0.0_f64;
            for k in 0..n {
                force += row[k] * (current[k] - theta_j).sin();
            }
            flat[base + j] = theta_j + dt * (omega[j] + force);
        }
        current.copy_from_slice(&flat[base..base + n]);
    }
    Array2::from_shape_vec((n_steps + 1, n), flat)
        .expect("trajectory buffer matches (n_steps + 1, n)")
}

/// Compute the reverse-mode adjoint of the networked-Kuramoto Euler integrator.
///
/// Given the forward ``trajectory`` and a cotangent λ_N = ∂L/∂θ_N on the final state, returns
/// (∂L/∂θ₀, ∂L/∂ω, ∂L/∂K) via λ_n = λ_{n+1} + dt J(θ_n)ᵀ λ_{n+1}, with the per-step
/// ∂L/∂ω and ∂L/∂K accumulations.
#[pyfunction]
pub fn kuramoto_euler_vjp<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray2<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    dt: f64,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> KuramotoGradientResult<'py> {
    let path = trajectory.as_array();
    if path.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "trajectory must be two-dimensional, got shape {:?}",
            path.shape()
        )));
    }
    let n = path.shape()[1];
    let matrix = coupling.as_array();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "coupling must have shape ({n}, {n}), got {:?}",
            matrix.shape()
        )));
    }
    let seed = validate_phase_vector(&cotangent, "cotangent")?;
    if seed.len() != n {
        return Err(PyValueError::new_err(format!(
            "cotangent must have shape ({n},), got ({},)",
            seed.len()
        )));
    }
    let (grad_theta0, grad_omega, grad_coupling) =
        kuramoto_euler_vjp_inner(&path, &matrix, dt, seed);
    Ok((
        PyArray1::from_owned_array(py, grad_theta0),
        PyArray1::from_owned_array(py, grad_omega),
        PyArray2::from_owned_array(py, grad_coupling),
    ))
}

/// Pure Rust reverse-mode adjoint (no PyO3). Returns (∂L/∂θ₀, ∂L/∂ω, ∂L/∂K).
///
/// The adjoint product `J(θ_n)ᵀ λ` is accumulated inline with reused buffers (no per-step
/// Jacobian matrix is materialised), keeping the reverse hot loop allocation-free.
pub fn kuramoto_euler_vjp_inner(
    trajectory: &ArrayView2<'_, f64>,
    coupling: &ArrayView2<'_, f64>,
    dt: f64,
    cotangent: &[f64],
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let n = trajectory.shape()[1];
    let n_steps = trajectory.shape()[0] - 1;
    let traj_flat = trajectory
        .as_slice()
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| trajectory.iter().copied().collect());
    let coupling_flat = coupling
        .as_slice()
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| coupling.iter().copied().collect());
    let mut adjoint: Vec<f64> = cotangent.to_vec();
    let mut next: Vec<f64> = vec![0.0; n];
    let mut grad_omega = vec![0.0_f64; n];
    let mut grad_coupling = vec![0.0_f64; n * n];
    for step in (0..n_steps).rev() {
        let phases = &traj_flat[step * n..step * n + n];
        // ∂L/∂ω += dt λ_{n+1}; ∂L/∂K_pq += dt λ_{n+1,p} sin(θ_q − θ_p).
        for p in 0..n {
            grad_omega[p] += dt * adjoint[p];
            let theta_p = phases[p];
            let lambda_p = adjoint[p];
            let g_row = &mut grad_coupling[p * n..p * n + n];
            for q in 0..n {
                g_row[q] += dt * lambda_p * (phases[q] - theta_p).sin();
            }
        }
        // λ_n = λ_{n+1} + dt J(θ_n)ᵀ λ_{n+1}, with J_jl = K_jl cos(θ_l − θ_j) (l ≠ j) and
        // J_jj = −Σ_{k≠j} K_jk cos(θ_k − θ_j). Accumulate (Jᵀ λ)_l inline by scattering each
        // row j's contributions, avoiding a materialised Jacobian.
        next.copy_from_slice(&adjoint);
        for j in 0..n {
            let theta_j = phases[j];
            let lambda_j = adjoint[j];
            let row = &coupling_flat[j * n..j * n + n];
            let mut diagonal = 0.0_f64;
            for l in 0..n {
                if l != j {
                    let entry = row[l] * (phases[l] - theta_j).cos();
                    next[l] += dt * entry * lambda_j;
                    diagonal += entry;
                }
            }
            next[j] -= dt * diagonal * lambda_j;
        }
        std::mem::swap(&mut adjoint, &mut next);
    }
    (
        Array1::from_vec(adjoint),
        Array1::from_vec(grad_omega),
        Array2::from_shape_vec((n, n), grad_coupling).expect("grad_coupling buffer matches (n, n)"),
    )
}

/// Write the networked force F_j = Σ_k K_jk sin(θ_k − θ_j) into ``out`` (flat coupling).
pub(crate) fn networked_force_into(theta: &[f64], coupling_flat: &[f64], n: usize, out: &mut [f64]) {
    for j in 0..n {
        let theta_j = theta[j];
        let row = &coupling_flat[j * n..j * n + n];
        let mut acc = 0.0_f64;
        for k in 0..n {
            acc += row[k] * (theta[k] - theta_j).sin();
        }
        out[j] = acc;
    }
}

/// Add ``dt (Jᵀ λ)_l`` into ``out`` for the networked Jacobian at ``theta`` (flat coupling).
fn add_dt_jacobian_transpose_product(
    theta: &[f64],
    coupling_flat: &[f64],
    n: usize,
    dt: f64,
    lambda: &[f64],
    out: &mut [f64],
) {
    for j in 0..n {
        let theta_j = theta[j];
        let lambda_j = lambda[j];
        let row = &coupling_flat[j * n..j * n + n];
        let mut diagonal = 0.0_f64;
        for l in 0..n {
            if l != j {
                let entry = row[l] * (theta[l] - theta_j).cos();
                out[l] += dt * entry * lambda_j;
                diagonal += entry;
            }
        }
        out[j] -= dt * diagonal * lambda_j;
    }
}

/// Accumulate one stage's ∂L/∂K_pq += λ_p sin(s_q − s_p) into ``grad_coupling`` (flat).
fn accumulate_coupling_gradient(
    stage: &[f64],
    stage_cotangent: &[f64],
    n: usize,
    grad_coupling: &mut [f64],
) {
    for p in 0..n {
        let theta_p = stage[p];
        let lambda_p = stage_cotangent[p];
        let g_row = &mut grad_coupling[p * n..p * n + n];
        for q in 0..n {
            g_row[q] += lambda_p * (stage[q] - theta_p).sin();
        }
    }
}

/// Compute the forward networked-Kuramoto RK4 trajectory θ_{n+1} = θ_n + (dt/6)(k1+2k2+2k3+k4).
///
/// Classical fourth-order Runge–Kutta of θ̇ = ω + F(θ). Returns the (n_steps + 1, N) trajectory.
#[pyfunction]
pub fn kuramoto_rk4_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: i64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
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
    if n_steps < 0 {
        return Err(PyValueError::new_err(format!(
            "n_steps must be non-negative, got {n_steps}"
        )));
    }
    Ok(PyArray2::from_owned_array(
        py,
        kuramoto_rk4_trajectory_inner(theta0, frequencies, &matrix, dt, n_steps as usize),
    ))
}

/// Pure Rust forward RK4 trajectory (no PyO3), returned row-major as (n_steps + 1, N).
pub fn kuramoto_rk4_trajectory_inner(
    theta0: &[f64],
    omega: &[f64],
    coupling: &ArrayView2<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> Array2<f64> {
    let n = theta0.len();
    let coupling_flat = coupling
        .as_slice()
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| coupling.iter().copied().collect());
    let mut flat = vec![0.0_f64; (n_steps + 1) * n];
    flat[..n].copy_from_slice(theta0);
    let mut current = theta0.to_vec();
    let mut k1 = vec![0.0_f64; n];
    let mut k2 = vec![0.0_f64; n];
    let mut k3 = vec![0.0_f64; n];
    let mut k4 = vec![0.0_f64; n];
    let mut stage = vec![0.0_f64; n];
    let half = 0.5 * dt;
    for step in 0..n_steps {
        networked_force_into(&current, &coupling_flat, n, &mut k1);
        for j in 0..n {
            k1[j] += omega[j];
            stage[j] = current[j] + half * k1[j];
        }
        networked_force_into(&stage, &coupling_flat, n, &mut k2);
        for j in 0..n {
            k2[j] += omega[j];
            stage[j] = current[j] + half * k2[j];
        }
        networked_force_into(&stage, &coupling_flat, n, &mut k3);
        for j in 0..n {
            k3[j] += omega[j];
            stage[j] = current[j] + dt * k3[j];
        }
        networked_force_into(&stage, &coupling_flat, n, &mut k4);
        let base = (step + 1) * n;
        for j in 0..n {
            k4[j] += omega[j];
            current[j] += (dt / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }
        flat[base..(n + base)].copy_from_slice(&current[..n]);
    }
    Array2::from_shape_vec((n_steps + 1, n), flat)
        .expect("trajectory buffer matches (n_steps+1, n)")
}

/// Compute the reverse-mode adjoint of the networked-Kuramoto RK4 integrator.
///
/// Backpropagates the terminal cotangent through each step's four RK4 stages (recomputed from
/// the stored trajectory) and returns (∂L/∂θ₀, ∂L/∂ω, ∂L/∂K).
#[pyfunction]
pub fn kuramoto_rk4_vjp<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray2<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    dt: f64,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> KuramotoGradientResult<'py> {
    let path = trajectory.as_array();
    if path.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "trajectory must be two-dimensional, got shape {:?}",
            path.shape()
        )));
    }
    let n = path.shape()[1];
    let frequencies = validate_phase_vector(&omega, "omega")?;
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
    let seed = validate_phase_vector(&cotangent, "cotangent")?;
    if seed.len() != n {
        return Err(PyValueError::new_err(format!(
            "cotangent must have shape ({n},), got ({},)",
            seed.len()
        )));
    }
    let (grad_theta0, grad_omega, grad_coupling) =
        kuramoto_rk4_vjp_inner(&path, frequencies, &matrix, dt, seed);
    Ok((
        PyArray1::from_owned_array(py, grad_theta0),
        PyArray1::from_owned_array(py, grad_omega),
        PyArray2::from_owned_array(py, grad_coupling),
    ))
}

/// Pure Rust reverse-mode RK4 adjoint (no PyO3). Returns (∂L/∂θ₀, ∂L/∂ω, ∂L/∂K).
pub fn kuramoto_rk4_vjp_inner(
    trajectory: &ArrayView2<'_, f64>,
    omega: &[f64],
    coupling: &ArrayView2<'_, f64>,
    dt: f64,
    cotangent: &[f64],
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let n = trajectory.shape()[1];
    let n_steps = trajectory.shape()[0] - 1;
    let traj_flat = trajectory
        .as_slice()
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| trajectory.iter().copied().collect());
    let coupling_flat = coupling
        .as_slice()
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| coupling.iter().copied().collect());
    let half = 0.5 * dt;
    let mut adjoint: Vec<f64> = cotangent.to_vec();
    let mut grad_omega = vec![0.0_f64; n];
    let mut grad_coupling = vec![0.0_f64; n * n];
    let (mut k1, mut k2, mut k3) = (vec![0.0_f64; n], vec![0.0_f64; n], vec![0.0_f64; n]);
    let (mut stage2, mut stage3, mut stage4) =
        (vec![0.0_f64; n], vec![0.0_f64; n], vec![0.0_f64; n]);
    for step in (0..n_steps).rev() {
        let phases = &traj_flat[step * n..step * n + n];
        // Recompute the forward stages from θ_n.
        networked_force_into(phases, &coupling_flat, n, &mut k1);
        for j in 0..n {
            k1[j] += omega[j];
            stage2[j] = phases[j] + half * k1[j];
        }
        networked_force_into(&stage2, &coupling_flat, n, &mut k2);
        for j in 0..n {
            k2[j] += omega[j];
            stage3[j] = phases[j] + half * k2[j];
        }
        networked_force_into(&stage3, &coupling_flat, n, &mut k3);
        for j in 0..n {
            k3[j] += omega[j];
            stage4[j] = phases[j] + dt * k3[j];
        }
        // Backpropagate through θ_{n+1} = θ_n + (dt/6)(k1 + 2k2 + 2k3 + k4) and the stage chain.
        let mut gc1 = vec![0.0_f64; n];
        let mut gc2 = vec![0.0_f64; n];
        let mut gc3 = vec![0.0_f64; n];
        let mut gc4 = vec![0.0_f64; n];
        for j in 0..n {
            gc1[j] = (dt / 6.0) * adjoint[j];
            gc2[j] = (dt / 3.0) * adjoint[j];
            gc3[j] = (dt / 3.0) * adjoint[j];
            gc4[j] = (dt / 6.0) * adjoint[j];
        }
        let mut next = adjoint.clone();
        let mut backward = vec![0.0_f64; n];
        // Stage 4: s4 = θ_n + dt·k3.
        backward.iter_mut().for_each(|value| *value = 0.0);
        add_dt_jacobian_transpose_product(&stage4, &coupling_flat, n, 1.0, &gc4, &mut backward);
        for j in 0..n {
            next[j] += backward[j];
            gc3[j] += dt * backward[j];
        }
        // Stage 3: s3 = θ_n + ½dt·k2.
        backward.iter_mut().for_each(|value| *value = 0.0);
        add_dt_jacobian_transpose_product(&stage3, &coupling_flat, n, 1.0, &gc3, &mut backward);
        for j in 0..n {
            next[j] += backward[j];
            gc2[j] += half * backward[j];
        }
        // Stage 2: s2 = θ_n + ½dt·k1.
        backward.iter_mut().for_each(|value| *value = 0.0);
        add_dt_jacobian_transpose_product(&stage2, &coupling_flat, n, 1.0, &gc2, &mut backward);
        for j in 0..n {
            next[j] += backward[j];
            gc1[j] += half * backward[j];
        }
        // Stage 1: k1 = f(θ_n).
        backward.iter_mut().for_each(|value| *value = 0.0);
        add_dt_jacobian_transpose_product(phases, &coupling_flat, n, 1.0, &gc1, &mut backward);
        for j in 0..n {
            next[j] += backward[j];
        }
        // ∂L/∂ω and ∂L/∂K from each stage's cotangent.
        for j in 0..n {
            grad_omega[j] += gc1[j] + gc2[j] + gc3[j] + gc4[j];
        }
        accumulate_coupling_gradient(phases, &gc1, n, &mut grad_coupling);
        accumulate_coupling_gradient(&stage2, &gc2, n, &mut grad_coupling);
        accumulate_coupling_gradient(&stage3, &gc3, n, &mut grad_coupling);
        accumulate_coupling_gradient(&stage4, &gc4, n, &mut grad_coupling);
        adjoint = next;
    }
    (
        Array1::from_vec(adjoint),
        Array1::from_vec(grad_omega),
        Array2::from_shape_vec((n, n), grad_coupling).expect("grad_coupling buffer matches (n, n)"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kuramoto_coupling::networked_kuramoto_force_inner;
    use ndarray::Array2;

    fn _symmetric_coupling(n: usize, seed: f64) -> Array2<f64> {
        let mut matrix = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let value = 0.5 * (((i + 1) as f64 * (j + 2) as f64 * seed).sin()).abs();
                matrix[[i, j]] = value;
                matrix[[j, i]] = value;
            }
        }
        matrix
    }

    #[test]
    fn test_forward_matches_manual_euler() {
        let theta0 = vec![0.3, -1.1, 2.0, 0.7];
        let omega = vec![0.2, -0.5, 0.1, 0.4];
        let coupling = _symmetric_coupling(4, 0.37);
        let dt = 0.05;
        let n_steps = 10;
        let trajectory =
            kuramoto_euler_trajectory_inner(&theta0, &omega, &coupling.view(), dt, n_steps);
        assert_eq!(trajectory.shape(), &[n_steps + 1, 4]);
        // Row 0 is the initial state.
        for j in 0..4 {
            assert!((trajectory[[0, j]] - theta0[j]).abs() < 1e-15);
        }
        // Reproduce one step by hand.
        let force = networked_kuramoto_force_inner(&theta0, &coupling.view());
        for j in 0..4 {
            let expected = theta0[j] + dt * (omega[j] + force[j]);
            assert!((trajectory[[1, j]] - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn test_adjoint_matches_finite_difference() {
        let theta0 = vec![0.3, -1.1, 2.0, 0.7];
        let omega = vec![0.2, -0.5, 0.1, 0.4];
        let coupling = _symmetric_coupling(4, 0.41);
        let dt = 0.05;
        let n_steps = 8;
        let n = 4;
        // Objective L = Σ_j sin(θ_{N,j}); cotangent λ_N = cos(θ_N).
        let objective =
            |traj: &Array2<f64>| -> f64 { (0..n).map(|j| traj[[n_steps, j]].sin()).sum() };
        let trajectory =
            kuramoto_euler_trajectory_inner(&theta0, &omega, &coupling.view(), dt, n_steps);
        let cotangent: Vec<f64> = (0..n).map(|j| trajectory[[n_steps, j]].cos()).collect();
        let (grad_theta0, grad_omega, grad_coupling) =
            kuramoto_euler_vjp_inner(&trajectory.view(), &coupling.view(), dt, &cotangent);
        let h = 1e-6;
        // ∂L/∂θ₀
        for i in 0..n {
            let mut plus = theta0.clone();
            let mut minus = theta0.clone();
            plus[i] += h;
            minus[i] -= h;
            let lp = objective(&kuramoto_euler_trajectory_inner(
                &plus,
                &omega,
                &coupling.view(),
                dt,
                n_steps,
            ));
            let lm = objective(&kuramoto_euler_trajectory_inner(
                &minus,
                &omega,
                &coupling.view(),
                dt,
                n_steps,
            ));
            assert!(
                (grad_theta0[i] - (lp - lm) / (2.0 * h)).abs() < 1e-6,
                "grad_theta0[{i}]"
            );
        }
        // ∂L/∂ω
        for i in 0..n {
            let mut plus = omega.clone();
            let mut minus = omega.clone();
            plus[i] += h;
            minus[i] -= h;
            let lp = objective(&kuramoto_euler_trajectory_inner(
                &theta0,
                &plus,
                &coupling.view(),
                dt,
                n_steps,
            ));
            let lm = objective(&kuramoto_euler_trajectory_inner(
                &theta0,
                &minus,
                &coupling.view(),
                dt,
                n_steps,
            ));
            assert!(
                (grad_omega[i] - (lp - lm) / (2.0 * h)).abs() < 1e-6,
                "grad_omega[{i}]"
            );
        }
        // ∂L/∂K
        for p in 0..n {
            for q in 0..n {
                let mut plus = coupling.clone();
                let mut minus = coupling.clone();
                plus[[p, q]] += h;
                minus[[p, q]] -= h;
                let lp = objective(&kuramoto_euler_trajectory_inner(
                    &theta0,
                    &omega,
                    &plus.view(),
                    dt,
                    n_steps,
                ));
                let lm = objective(&kuramoto_euler_trajectory_inner(
                    &theta0,
                    &omega,
                    &minus.view(),
                    dt,
                    n_steps,
                ));
                assert!(
                    (grad_coupling[[p, q]] - (lp - lm) / (2.0 * h)).abs() < 1e-6,
                    "grad_coupling[{p},{q}]"
                );
            }
        }
    }

    #[test]
    fn test_zero_steps_is_identity() {
        let theta0 = vec![0.3, -1.1, 2.0];
        let omega = vec![0.0, 0.0, 0.0];
        let coupling = _symmetric_coupling(3, 0.5);
        let trajectory = kuramoto_euler_trajectory_inner(&theta0, &omega, &coupling.view(), 0.1, 0);
        assert_eq!(trajectory.shape(), &[1, 3]);
        let cotangent = vec![1.0, 2.0, 3.0];
        let (grad_theta0, grad_omega, grad_coupling) =
            kuramoto_euler_vjp_inner(&trajectory.view(), &coupling.view(), 0.1, &cotangent);
        // With no steps the objective depends only on θ₀, so ∂L/∂θ₀ = cotangent and the others
        // vanish.
        for j in 0..3 {
            assert!((grad_theta0[j] - cotangent[j]).abs() < 1e-15);
            assert!(grad_omega[j].abs() < 1e-15);
        }
        assert!(grad_coupling.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn test_rk4_fourth_order_convergence() {
        let theta0 = vec![0.3, -1.1, 2.0, 0.7];
        let omega = vec![0.2, -0.5, 0.1, 0.4];
        let coupling = _symmetric_coupling(4, 0.41);
        let total = 1.0_f64;
        let final_state = |steps: usize| -> Vec<f64> {
            let traj = kuramoto_rk4_trajectory_inner(
                &theta0,
                &omega,
                &coupling.view(),
                total / steps as f64,
                steps,
            );
            (0..4).map(|j| traj[[steps, j]]).collect()
        };
        let coarse = final_state(40);
        let mid = final_state(80);
        let fine = final_state(160);
        let err1: f64 = (0..4)
            .map(|j| (coarse[j] - mid[j]).abs())
            .fold(0.0, f64::max);
        let err2: f64 = (0..4).map(|j| (mid[j] - fine[j]).abs()).fold(0.0, f64::max);
        // Halving dt cuts a fourth-order error by ~16×.
        assert!(err1 / err2 > 12.0, "ratio {}", err1 / err2);
    }

    #[test]
    fn test_rk4_adjoint_matches_finite_difference() {
        let theta0 = vec![0.3, -1.1, 2.0, 0.7];
        let omega = vec![0.2, -0.5, 0.1, 0.4];
        let coupling = _symmetric_coupling(4, 0.41);
        let dt = 0.05;
        let n_steps = 8;
        let n = 4;
        let objective =
            |traj: &Array2<f64>| -> f64 { (0..n).map(|j| traj[[n_steps, j]].sin()).sum() };
        let trajectory =
            kuramoto_rk4_trajectory_inner(&theta0, &omega, &coupling.view(), dt, n_steps);
        let cotangent: Vec<f64> = (0..n).map(|j| trajectory[[n_steps, j]].cos()).collect();
        let (grad_theta0, grad_omega, grad_coupling) =
            kuramoto_rk4_vjp_inner(&trajectory.view(), &omega, &coupling.view(), dt, &cotangent);
        let h = 1e-6;
        for i in 0..n {
            let mut plus = theta0.clone();
            let mut minus = theta0.clone();
            plus[i] += h;
            minus[i] -= h;
            let lp = objective(&kuramoto_rk4_trajectory_inner(
                &plus,
                &omega,
                &coupling.view(),
                dt,
                n_steps,
            ));
            let lm = objective(&kuramoto_rk4_trajectory_inner(
                &minus,
                &omega,
                &coupling.view(),
                dt,
                n_steps,
            ));
            assert!(
                (grad_theta0[i] - (lp - lm) / (2.0 * h)).abs() < 1e-6,
                "grad_theta0[{i}]"
            );
        }
        for i in 0..n {
            let mut plus = omega.clone();
            let mut minus = omega.clone();
            plus[i] += h;
            minus[i] -= h;
            let lp = objective(&kuramoto_rk4_trajectory_inner(
                &theta0,
                &plus,
                &coupling.view(),
                dt,
                n_steps,
            ));
            let lm = objective(&kuramoto_rk4_trajectory_inner(
                &theta0,
                &minus,
                &coupling.view(),
                dt,
                n_steps,
            ));
            assert!(
                (grad_omega[i] - (lp - lm) / (2.0 * h)).abs() < 1e-6,
                "grad_omega[{i}]"
            );
        }
        for p in 0..n {
            for q in 0..n {
                let mut plus = coupling.clone();
                let mut minus = coupling.clone();
                plus[[p, q]] += h;
                minus[[p, q]] -= h;
                let lp = objective(&kuramoto_rk4_trajectory_inner(
                    &theta0,
                    &omega,
                    &plus.view(),
                    dt,
                    n_steps,
                ));
                let lm = objective(&kuramoto_rk4_trajectory_inner(
                    &theta0,
                    &omega,
                    &minus.view(),
                    dt,
                    n_steps,
                ));
                assert!(
                    (grad_coupling[[p, q]] - (lp - lm) / (2.0 * h)).abs() < 1e-6,
                    "grad_coupling[{p},{q}]"
                );
            }
        }
    }

    #[test]
    fn test_rk4_reduces_error_versus_euler() {
        // Over the same step budget RK4 tracks the reference (very fine RK4) far better than Euler.
        let theta0 = vec![0.3, -1.1, 2.0, 0.7];
        let omega = vec![0.2, -0.5, 0.1, 0.4];
        let coupling = _symmetric_coupling(4, 0.41);
        let total = 2.0_f64;
        let steps = 20;
        let dt = total / steps as f64;
        let reference =
            kuramoto_rk4_trajectory_inner(&theta0, &omega, &coupling.view(), total / 4000.0, 4000);
        let rk4 = kuramoto_rk4_trajectory_inner(&theta0, &omega, &coupling.view(), dt, steps);
        let euler = kuramoto_euler_trajectory_inner(&theta0, &omega, &coupling.view(), dt, steps);
        let rk4_err: f64 = (0..4)
            .map(|j| (rk4[[steps, j]] - reference[[4000, j]]).abs())
            .fold(0.0, f64::max);
        let euler_err: f64 = (0..4)
            .map(|j| (euler[[steps, j]] - reference[[4000, j]]).abs())
            .fold(0.0, f64::max);
        assert!(rk4_err < euler_err);
    }
}
