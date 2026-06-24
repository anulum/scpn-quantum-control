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
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
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
}
