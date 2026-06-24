// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Kuramoto coupling forces, Jacobians, energy and local order
//! Mean-field, networked and Sakaguchi coupling forces and stability Jacobians, the
//! interaction energy with its gradient and Hessian, and the network-local order parameter.

use ndarray::{Array1, Array2, ArrayView2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::kuramoto_common::validate_phase_vector;

/// Compute the Kuramoto mean-field coupling force F_j = K (S cos θ_j − C sin θ_j).
///
/// This is the phase-coupling term of the all-to-all Kuramoto dynamics, with
/// C = <cos θ> and S = <sin θ>.
#[pyfunction]
pub fn mean_field_force<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(PyArray1::from_owned_array(
        py,
        mean_field_force_inner(theta, coupling),
    ))
}

/// Pure Rust mean-field force (no PyO3).
pub fn mean_field_force_inner(theta: &[f64], coupling: f64) -> Array1<f64> {
    let n = theta.len();
    if n == 0 {
        return Array1::zeros(0);
    }
    let (mut c, mut s) = (0.0_f64, 0.0_f64);
    for &t in theta {
        c += t.cos();
        s += t.sin();
    }
    let count = n as f64;
    let cos_mean = c / count;
    let sin_mean = s / count;
    Array1::from_iter(
        theta
            .iter()
            .map(|&t| coupling * (sin_mean * t.cos() - cos_mean * t.sin())),
    )
}

/// Compute the Kuramoto synchronisation stability Jacobian J_jk = ∂F_j/∂θ_k.
///
/// J_jk = (K/N) cos(θ_j − θ_k) − K δ_jk (C cos θ_j + S sin θ_j). The matrix is symmetric
/// and every row sums to zero (the global-phase Goldstone mode).
#[pyfunction]
pub fn mean_field_jacobian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(PyArray2::from_owned_array(
        py,
        mean_field_jacobian_inner(theta, coupling),
    ))
}

/// Pure Rust mean-field stability Jacobian (no PyO3), returned row-major.
pub fn mean_field_jacobian_inner(theta: &[f64], coupling: f64) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    if n == 0 {
        return out;
    }
    let (mut c, mut s) = (0.0_f64, 0.0_f64);
    for &t in theta {
        c += t.cos();
        s += t.sin();
    }
    let count = n as f64;
    let cos_mean = c / count;
    let sin_mean = s / count;
    let scale = coupling / count;
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = scale * (theta[i] - theta[j]).cos();
        }
        out[[i, i]] -= coupling * (cos_mean * theta[i].cos() + sin_mean * theta[i].sin());
    }
    out
}

/// Compute the networked Kuramoto coupling force F_j = Σ_k K_jk sin(θ_k − θ_j).
///
/// ``coupling`` is the N×N coupling matrix K. The k = j term is sin(0) = 0, so the force
/// is independent of the diagonal of K.
#[pyfunction]
pub fn networked_kuramoto_force<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let matrix = coupling.as_array();
    let n = theta.len();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "coupling must be a square matrix of order {n}, got shape {:?}",
            matrix.shape()
        )));
    }
    Ok(PyArray1::from_owned_array(
        py,
        networked_kuramoto_force_inner(theta, &matrix),
    ))
}

/// Pure Rust networked Kuramoto force (no PyO3).
pub fn networked_kuramoto_force_inner(
    theta: &[f64],
    coupling: &ArrayView2<'_, f64>,
) -> Array1<f64> {
    let n = theta.len();
    let mut out = Array1::<f64>::zeros(n);
    for j in 0..n {
        let mut acc = 0.0_f64;
        for k in 0..n {
            acc += coupling[[j, k]] * (theta[k] - theta[j]).sin();
        }
        out[j] = acc;
    }
    out
}

/// Compute the networked Kuramoto stability Jacobian J_jl = ∂F_j/∂θ_l.
///
/// J_jl = K_jl cos(θ_l − θ_j) for l ≠ j, with J_jj = −Σ_{k≠j} K_jk cos(θ_k − θ_j). The
/// matrix is symmetric when K is, and every row sums to zero (the global-phase Goldstone
/// mode). It is independent of the diagonal of K.
#[pyfunction]
pub fn networked_kuramoto_jacobian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let matrix = coupling.as_array();
    let n = theta.len();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "coupling must be a square matrix of order {n}, got shape {:?}",
            matrix.shape()
        )));
    }
    Ok(PyArray2::from_owned_array(
        py,
        networked_kuramoto_jacobian_inner(theta, &matrix),
    ))
}

/// Pure Rust networked Kuramoto stability Jacobian (no PyO3), returned row-major.
pub fn networked_kuramoto_jacobian_inner(
    theta: &[f64],
    coupling: &ArrayView2<'_, f64>,
) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut diagonal = 0.0_f64;
        for l in 0..n {
            if l == j {
                continue;
            }
            let entry = coupling[[j, l]] * (theta[l] - theta[j]).cos();
            out[[j, l]] = entry;
            diagonal -= entry;
        }
        out[[j, j]] = diagonal;
    }
    out
}

/// Compute the Kuramoto interaction energy E = −½ Σ_jk K_jk cos(θ_j − θ_k).
///
/// ``coupling`` is the N×N coupling matrix K. For symmetric K this is the Lyapunov
/// function whose gradient flow is the dynamics.
#[pyfunction]
pub fn kuramoto_interaction_energy(
    theta: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let matrix = coupling.as_array();
    let n = theta.len();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "coupling must be a square matrix of order {n}, got shape {:?}",
            matrix.shape()
        )));
    }
    Ok(kuramoto_interaction_energy_inner(theta, &matrix))
}

/// Pure Rust Kuramoto interaction energy (no PyO3).
pub fn kuramoto_interaction_energy_inner(theta: &[f64], coupling: &ArrayView2<'_, f64>) -> f64 {
    let n = theta.len();
    let mut acc = 0.0_f64;
    for j in 0..n {
        for k in 0..n {
            acc += coupling[[j, k]] * (theta[j] - theta[k]).cos();
        }
    }
    -0.5 * acc
}

/// Compute the gradient ∂E/∂θ_j = ½ Σ_k (K_jk + K_kj) sin(θ_j − θ_k) of the interaction
/// energy.
///
/// The components sum to zero (E is invariant under a global phase shift). For symmetric K
/// this equals the negated networked-Kuramoto force.
#[pyfunction]
pub fn kuramoto_interaction_energy_gradient<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let matrix = coupling.as_array();
    let n = theta.len();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "coupling must be a square matrix of order {n}, got shape {:?}",
            matrix.shape()
        )));
    }
    Ok(PyArray1::from_owned_array(
        py,
        kuramoto_interaction_energy_gradient_inner(theta, &matrix),
    ))
}

/// Pure Rust Kuramoto interaction-energy gradient (no PyO3).
pub fn kuramoto_interaction_energy_gradient_inner(
    theta: &[f64],
    coupling: &ArrayView2<'_, f64>,
) -> Array1<f64> {
    let n = theta.len();
    let mut out = Array1::<f64>::zeros(n);
    for j in 0..n {
        let mut acc = 0.0_f64;
        for k in 0..n {
            acc += (coupling[[j, k]] + coupling[[k, j]]) * (theta[j] - theta[k]).sin();
        }
        out[j] = 0.5 * acc;
    }
    out
}

/// Compute the Hessian ∂²E/∂θ_i∂θ_l of the Kuramoto interaction energy.
///
/// H_il = −½(K_il + K_li) cos(θ_i − θ_l) for l ≠ i, with H_ii = −Σ_{l≠i} H_il. The matrix
/// is symmetric and every row sums to zero; for symmetric K it equals the negated networked
/// Jacobian.
#[pyfunction]
pub fn kuramoto_interaction_energy_hessian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let matrix = coupling.as_array();
    let n = theta.len();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "coupling must be a square matrix of order {n}, got shape {:?}",
            matrix.shape()
        )));
    }
    Ok(PyArray2::from_owned_array(
        py,
        kuramoto_interaction_energy_hessian_inner(theta, &matrix),
    ))
}

/// Pure Rust Kuramoto interaction-energy Hessian (no PyO3), returned row-major.
pub fn kuramoto_interaction_energy_hessian_inner(
    theta: &[f64],
    coupling: &ArrayView2<'_, f64>,
) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let mut diagonal = 0.0_f64;
        for l in 0..n {
            if l == i {
                continue;
            }
            let entry = -0.5 * (coupling[[i, l]] + coupling[[l, i]]) * (theta[i] - theta[l]).cos();
            out[[i, l]] = entry;
            diagonal -= entry;
        }
        out[[i, i]] = diagonal;
    }
    out
}

/// Compute the Kuramoto–Sakaguchi frustrated force F_j = Σ_{k≠j} K_jk sin(θ_k − θ_j − α).
///
/// ``coupling`` is the N×N matrix K and ``frustration`` is the angle α. The self-coupling
/// term is excluded. For α = 0 this is the networked-Kuramoto force.
#[pyfunction]
pub fn sakaguchi_force<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    frustration: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let matrix = coupling.as_array();
    let n = theta.len();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "coupling must be a square matrix of order {n}, got shape {:?}",
            matrix.shape()
        )));
    }
    Ok(PyArray1::from_owned_array(
        py,
        sakaguchi_force_inner(theta, &matrix, frustration),
    ))
}

/// Pure Rust Kuramoto–Sakaguchi force (no PyO3).
pub fn sakaguchi_force_inner(
    theta: &[f64],
    coupling: &ArrayView2<'_, f64>,
    frustration: f64,
) -> Array1<f64> {
    let n = theta.len();
    let mut out = Array1::<f64>::zeros(n);
    for j in 0..n {
        let mut acc = 0.0_f64;
        for k in 0..n {
            if k == j {
                continue;
            }
            acc += coupling[[j, k]] * (theta[k] - theta[j] - frustration).sin();
        }
        out[j] = acc;
    }
    out
}

/// Compute the Kuramoto–Sakaguchi stability Jacobian J_jl = ∂F_j/∂θ_l.
///
/// J_jl = K_jl cos(θ_l − θ_j − α) for l ≠ j, with J_jj = −Σ_{k≠j} K_jk cos(θ_k − θ_j − α).
/// Every row sums to zero; for α ≠ 0 the matrix is asymmetric even for symmetric K.
#[pyfunction]
pub fn sakaguchi_jacobian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: PyReadonlyArray2<'_, f64>,
    frustration: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let matrix = coupling.as_array();
    let n = theta.len();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "coupling must be a square matrix of order {n}, got shape {:?}",
            matrix.shape()
        )));
    }
    Ok(PyArray2::from_owned_array(
        py,
        sakaguchi_jacobian_inner(theta, &matrix, frustration),
    ))
}

/// Pure Rust Kuramoto–Sakaguchi stability Jacobian (no PyO3), returned row-major.
pub fn sakaguchi_jacobian_inner(
    theta: &[f64],
    coupling: &ArrayView2<'_, f64>,
    frustration: f64,
) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut diagonal = 0.0_f64;
        for l in 0..n {
            if l == j {
                continue;
            }
            let entry = coupling[[j, l]] * (theta[l] - theta[j] - frustration).cos();
            out[[j, l]] = entry;
            diagonal -= entry;
        }
        out[[j, j]] = diagonal;
    }
    out
}

/// Compute the network-local Kuramoto order parameter r_j = |Σ_k A_jk e^{iθ_k}| / Σ_k A_jk.
///
/// ``adjacency`` is the N×N non-negative adjacency matrix A. A zero-degree node has r_j = 0.
#[pyfunction]
pub fn local_order_parameter<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    adjacency: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let matrix = adjacency.as_array();
    let n = theta.len();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "adjacency must be a square matrix of order {n}, got shape {:?}",
            matrix.shape()
        )));
    }
    Ok(PyArray1::from_owned_array(
        py,
        local_order_parameter_inner(theta, &matrix),
    ))
}

/// Pure Rust network-local order parameter (no PyO3).
pub fn local_order_parameter_inner(theta: &[f64], adjacency: &ArrayView2<'_, f64>) -> Array1<f64> {
    let n = theta.len();
    let mut out = Array1::<f64>::zeros(n);
    let cos: Vec<f64> = theta.iter().map(|&t| t.cos()).collect();
    let sin: Vec<f64> = theta.iter().map(|&t| t.sin()).collect();
    for j in 0..n {
        let (mut c, mut s, mut d) = (0.0_f64, 0.0_f64, 0.0_f64);
        for k in 0..n {
            let a = adjacency[[j, k]];
            c += a * cos[k];
            s += a * sin[k];
            d += a;
        }
        if d != 0.0 {
            out[j] = (c * c + s * s).sqrt() / d;
        }
    }
    out
}

/// Compute the Jacobian ∂r_j/∂θ_l = (A_jl / d_j) sin(ψ_j − θ_l) of the local order parameter.
///
/// A zero-degree node or an incoherent neighbourhood (|Σ_k A_jk e^{iθ_k}| = 0) yields a zero
/// subgradient row.
#[pyfunction]
pub fn local_order_parameter_jacobian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    adjacency: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let matrix = adjacency.as_array();
    let n = theta.len();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "adjacency must be a square matrix of order {n}, got shape {:?}",
            matrix.shape()
        )));
    }
    Ok(PyArray2::from_owned_array(
        py,
        local_order_parameter_jacobian_inner(theta, &matrix),
    ))
}

/// Pure Rust local order parameter Jacobian (no PyO3), returned row-major.
pub fn local_order_parameter_jacobian_inner(
    theta: &[f64],
    adjacency: &ArrayView2<'_, f64>,
) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    let cos: Vec<f64> = theta.iter().map(|&t| t.cos()).collect();
    let sin: Vec<f64> = theta.iter().map(|&t| t.sin()).collect();
    for j in 0..n {
        let (mut c, mut s, mut d) = (0.0_f64, 0.0_f64, 0.0_f64);
        for k in 0..n {
            let a = adjacency[[j, k]];
            c += a * cos[k];
            s += a * sin[k];
            d += a;
        }
        let magnitude = (c * c + s * s).sqrt();
        let denominator = d * magnitude;
        if denominator == 0.0 {
            continue;
        }
        let inverse = 1.0 / denominator;
        for l in 0..n {
            out[[j, l]] = adjacency[[j, l]] * inverse * (s * cos[l] - c * sin[l]);
        }
    }
    out
}

/// Compute the network-local mean phase ψ_j = atan2(Σ_k A_jk sin θ_k, Σ_k A_jk cos θ_k).
///
/// A zero-degree node or an incoherent neighbourhood (|Σ_k A_jk e^{iθ_k}| = 0) yields ψ_j = 0.
#[pyfunction]
pub fn local_mean_phase<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    adjacency: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let matrix = adjacency.as_array();
    let n = theta.len();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "adjacency must be a square matrix of order {n}, got shape {:?}",
            matrix.shape()
        )));
    }
    Ok(PyArray1::from_owned_array(
        py,
        local_mean_phase_inner(theta, &matrix),
    ))
}

/// Pure Rust network-local mean phase (no PyO3).
pub fn local_mean_phase_inner(theta: &[f64], adjacency: &ArrayView2<'_, f64>) -> Array1<f64> {
    let n = theta.len();
    let mut out = Array1::<f64>::zeros(n);
    let cos: Vec<f64> = theta.iter().map(|&t| t.cos()).collect();
    let sin: Vec<f64> = theta.iter().map(|&t| t.sin()).collect();
    for j in 0..n {
        let (mut c, mut s) = (0.0_f64, 0.0_f64);
        for k in 0..n {
            let a = adjacency[[j, k]];
            c += a * cos[k];
            s += a * sin[k];
        }
        if c * c + s * s != 0.0 {
            out[j] = s.atan2(c);
        }
    }
    out
}

/// Compute the Jacobian ∂ψ_j/∂θ_l = A_jl cos(ψ_j − θ_l) / |Z_j| of the local mean phase.
///
/// A zero-degree node or an incoherent neighbourhood (|Z_j| = 0) yields a zero subgradient row.
#[pyfunction]
pub fn local_mean_phase_jacobian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    adjacency: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    let matrix = adjacency.as_array();
    let n = theta.len();
    if matrix.shape() != [n, n] {
        return Err(PyValueError::new_err(format!(
            "adjacency must be a square matrix of order {n}, got shape {:?}",
            matrix.shape()
        )));
    }
    Ok(PyArray2::from_owned_array(
        py,
        local_mean_phase_jacobian_inner(theta, &matrix),
    ))
}

/// Pure Rust local mean phase Jacobian (no PyO3), returned row-major.
pub fn local_mean_phase_jacobian_inner(
    theta: &[f64],
    adjacency: &ArrayView2<'_, f64>,
) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    let cos: Vec<f64> = theta.iter().map(|&t| t.cos()).collect();
    let sin: Vec<f64> = theta.iter().map(|&t| t.sin()).collect();
    for j in 0..n {
        let (mut c, mut s) = (0.0_f64, 0.0_f64);
        for k in 0..n {
            let a = adjacency[[j, k]];
            c += a * cos[k];
            s += a * sin[k];
        }
        let magnitude_squared = c * c + s * s;
        if magnitude_squared == 0.0 {
            continue;
        }
        let inverse = 1.0 / magnitude_squared;
        for l in 0..n {
            out[[j, l]] = adjacency[[j, l]] * inverse * (c * cos[l] + s * sin[l]);
        }
    }
    out
}

/// Compute the Daido m-th-harmonic mean-field force F_j = K (S_m cos m θ_j − C_m sin m θ_j).
///
/// With C_m = ⟨cos m θ⟩, S_m = ⟨sin m θ⟩. For m = 1 this is the mean-field force.
#[pyfunction]
pub fn daido_mean_field_force<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: f64,
    m: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if m < 1 {
        return Err(PyValueError::new_err(format!(
            "harmonic order m must be a positive integer, got {m}"
        )));
    }
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(PyArray1::from_owned_array(
        py,
        daido_mean_field_force_inner(theta, coupling, m as f64),
    ))
}

/// Pure Rust Daido m-th-harmonic mean-field force (no PyO3).
pub fn daido_mean_field_force_inner(theta: &[f64], coupling: f64, m: f64) -> Array1<f64> {
    let n = theta.len();
    if n == 0 {
        return Array1::zeros(0);
    }
    let (mut c, mut s) = (0.0_f64, 0.0_f64);
    for &t in theta {
        c += (m * t).cos();
        s += (m * t).sin();
    }
    let count = n as f64;
    let cos_mean = c / count;
    let sin_mean = s / count;
    Array1::from_iter(
        theta
            .iter()
            .map(|&t| coupling * (sin_mean * (m * t).cos() - cos_mean * (m * t).sin())),
    )
}

/// Compute the Daido m-th-harmonic mean-field stability Jacobian J_jl = ∂F_j/∂θ_l.
///
/// J_jl = K m [(1/N) cos(m(θ_j − θ_l)) − δ_jl (C_m cos m θ_j + S_m sin m θ_j)]. The matrix is
/// symmetric and every row sums to zero (the global-phase Goldstone mode).
#[pyfunction]
pub fn daido_mean_field_jacobian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: f64,
    m: i64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if m < 1 {
        return Err(PyValueError::new_err(format!(
            "harmonic order m must be a positive integer, got {m}"
        )));
    }
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(PyArray2::from_owned_array(
        py,
        daido_mean_field_jacobian_inner(theta, coupling, m as f64),
    ))
}

/// Pure Rust Daido m-th-harmonic mean-field stability Jacobian (no PyO3), returned row-major.
pub fn daido_mean_field_jacobian_inner(theta: &[f64], coupling: f64, m: f64) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    if n == 0 {
        return out;
    }
    let (mut c, mut s) = (0.0_f64, 0.0_f64);
    for &t in theta {
        c += (m * t).cos();
        s += (m * t).sin();
    }
    let count = n as f64;
    let cos_mean = c / count;
    let sin_mean = s / count;
    let scale = coupling * m / count;
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = scale * (m * (theta[i] - theta[j])).cos();
        }
        out[[i, i]] -=
            coupling * m * (cos_mean * (m * theta[i]).cos() + sin_mean * (m * theta[i]).sin());
    }
    out
}

/// Compute the Sakaguchi–Kuramoto mean-field force F_j = K r sin(ψ − θ_j − α).
///
/// Expanded as K [(S cos θ_j − C sin θ_j) cos α − (C cos θ_j + S sin θ_j) sin α] with
/// C = ⟨cos θ⟩, S = ⟨sin θ⟩. For α = 0 this is the mean-field force.
#[pyfunction]
pub fn sakaguchi_mean_field_force<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: f64,
    frustration: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(PyArray1::from_owned_array(
        py,
        sakaguchi_mean_field_force_inner(theta, coupling, frustration),
    ))
}

/// Pure Rust Sakaguchi–Kuramoto mean-field force (no PyO3).
pub fn sakaguchi_mean_field_force_inner(
    theta: &[f64],
    coupling: f64,
    frustration: f64,
) -> Array1<f64> {
    let n = theta.len();
    if n == 0 {
        return Array1::zeros(0);
    }
    let (mut c, mut s) = (0.0_f64, 0.0_f64);
    for &t in theta {
        c += t.cos();
        s += t.sin();
    }
    let count = n as f64;
    let cos_mean = c / count;
    let sin_mean = s / count;
    let cos_a = frustration.cos();
    let sin_a = frustration.sin();
    Array1::from_iter(theta.iter().map(|&t| {
        let in_phase = sin_mean * t.cos() - cos_mean * t.sin();
        let quadrature = cos_mean * t.cos() + sin_mean * t.sin();
        coupling * (in_phase * cos_a - quadrature * sin_a)
    }))
}

/// Compute the Sakaguchi–Kuramoto mean-field stability Jacobian J_jl = ∂F_j/∂θ_l.
///
/// J_jl = (K/N) cos(θ_j − θ_l + α) − δ_jl K (C cos(θ_j + α) + S sin(θ_j + α)). Non-symmetric for
/// α ≠ 0 (the dynamics are non-variational), yet every row sums to zero.
#[pyfunction]
pub fn sakaguchi_mean_field_jacobian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: f64,
    frustration: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(PyArray2::from_owned_array(
        py,
        sakaguchi_mean_field_jacobian_inner(theta, coupling, frustration),
    ))
}

/// Pure Rust Sakaguchi–Kuramoto mean-field stability Jacobian (no PyO3), returned row-major.
pub fn sakaguchi_mean_field_jacobian_inner(
    theta: &[f64],
    coupling: f64,
    frustration: f64,
) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    if n == 0 {
        return out;
    }
    let (mut c, mut s) = (0.0_f64, 0.0_f64);
    for &t in theta {
        c += t.cos();
        s += t.sin();
    }
    let count = n as f64;
    let cos_mean = c / count;
    let sin_mean = s / count;
    let scale = coupling / count;
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = scale * (theta[i] - theta[j] + frustration).cos();
        }
        out[[i, i]] -= coupling
            * (cos_mean * (theta[i] + frustration).cos()
                + sin_mean * (theta[i] + frustration).sin());
    }
    out
}

/// Compute the triadic (2-simplex) Kuramoto mean-field force F_j = K r² sin(2ψ − 2θ_j).
///
/// Expanded as K [2 C S cos 2θ_j − (C² − S²) sin 2θ_j] with C = ⟨cos θ⟩, S = ⟨sin θ⟩. The r²
/// scaling drives explosive (abrupt) synchronisation.
#[pyfunction]
pub fn triadic_mean_field_force<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(PyArray1::from_owned_array(
        py,
        triadic_mean_field_force_inner(theta, coupling),
    ))
}

/// Pure Rust triadic Kuramoto mean-field force (no PyO3).
pub fn triadic_mean_field_force_inner(theta: &[f64], coupling: f64) -> Array1<f64> {
    let n = theta.len();
    if n == 0 {
        return Array1::zeros(0);
    }
    let (mut c, mut s) = (0.0_f64, 0.0_f64);
    for &t in theta {
        c += t.cos();
        s += t.sin();
    }
    let count = n as f64;
    let cos_mean = c / count;
    let sin_mean = s / count;
    let in_phase = 2.0 * cos_mean * sin_mean;
    let quadrature = cos_mean * cos_mean - sin_mean * sin_mean;
    Array1::from_iter(
        theta
            .iter()
            .map(|&t| coupling * (in_phase * (2.0 * t).cos() - quadrature * (2.0 * t).sin())),
    )
}

/// Compute the triadic Kuramoto mean-field stability Jacobian J_jl = ∂F_j/∂θ_l.
///
/// J_jl = (2K/N) (C cos(2θ_j − θ_l) + S sin(2θ_j − θ_l)) on the off-diagonal, with the
/// −2K (2 C S sin 2θ_j + (C² − S²) cos 2θ_j) curvature added on the diagonal. Non-symmetric
/// (the higher-order mean field is non-variational), yet every row sums to zero.
#[pyfunction]
pub fn triadic_mean_field_jacobian<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'_, f64>,
    coupling: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let theta = validate_phase_vector(&theta, "theta")?;
    Ok(PyArray2::from_owned_array(
        py,
        triadic_mean_field_jacobian_inner(theta, coupling),
    ))
}

/// Pure Rust triadic Kuramoto mean-field stability Jacobian (no PyO3), returned row-major.
pub fn triadic_mean_field_jacobian_inner(theta: &[f64], coupling: f64) -> Array2<f64> {
    let n = theta.len();
    let mut out = Array2::<f64>::zeros((n, n));
    if n == 0 {
        return out;
    }
    let (mut c, mut s) = (0.0_f64, 0.0_f64);
    for &t in theta {
        c += t.cos();
        s += t.sin();
    }
    let count = n as f64;
    let cos_mean = c / count;
    let sin_mean = s / count;
    let scale = 2.0 * coupling / count;
    for i in 0..n {
        for j in 0..n {
            let offset = 2.0 * theta[i] - theta[j];
            out[[i, j]] = scale * (cos_mean * offset.cos() + sin_mean * offset.sin());
        }
        let in_phase = 2.0 * cos_mean * sin_mean;
        let quadrature = cos_mean * cos_mean - sin_mean * sin_mean;
        out[[i, i]] -= 2.0
            * coupling
            * (in_phase * (2.0 * theta[i]).sin() + quadrature * (2.0 * theta[i]).cos());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kuramoto_observables::order_parameter_inner;

    #[test]
    fn test_mean_field_force_matches_closed_form() {
        let theta = vec![0.3, -1.1, 2.0, 0.7];
        let k = 1.7;
        let force = mean_field_force_inner(&theta, k);
        let n = theta.len() as f64;
        let c: f64 = theta.iter().map(|t| t.cos()).sum::<f64>() / n;
        let s: f64 = theta.iter().map(|t| t.sin()).sum::<f64>() / n;
        for (j, &t) in theta.iter().enumerate() {
            let expected = k * (s * t.cos() - c * t.sin());
            assert!((force[j] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_mean_field_jacobian_matches_finite_difference_of_force() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let k = 2.5;
        let jacobian = mean_field_jacobian_inner(&theta, k);
        let h = 1e-6;
        for kk in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[kk] += h;
            minus[kk] -= h;
            let force_plus = mean_field_force_inner(&plus, k);
            let force_minus = mean_field_force_inner(&minus, k);
            for j in 0..theta.len() {
                let fd = (force_plus[j] - force_minus[j]) / (2.0 * h);
                assert!((jacobian[[j, kk]] - fd).abs() < 1e-6, "J[{j},{kk}]");
            }
        }
    }

    #[test]
    fn test_mean_field_jacobian_is_symmetric_and_rows_sum_to_zero() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5];
        let jacobian = mean_field_jacobian_inner(&theta, 1.3);
        let n = theta.len();
        for i in 0..n {
            for j in 0..n {
                assert!((jacobian[[i, j]] - jacobian[[j, i]]).abs() < 1e-15);
            }
            let row_sum: f64 = (0..n).map(|j| jacobian[[i, j]]).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_mean_field_empty() {
        assert!(mean_field_force_inner(&[], 1.0).is_empty());
        assert_eq!(mean_field_jacobian_inner(&[], 1.0).shape(), &[0, 0]);
    }

    #[test]
    fn test_daido_mean_field_m1_matches_mean_field() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let k = 1.7;
        let daido_force = daido_mean_field_force_inner(&theta, k, 1.0);
        let mean_force = mean_field_force_inner(&theta, k);
        let daido_jac = daido_mean_field_jacobian_inner(&theta, k, 1.0);
        let mean_jac = mean_field_jacobian_inner(&theta, k);
        for j in 0..theta.len() {
            assert!((daido_force[j] - mean_force[j]).abs() < 1e-12);
            for l in 0..theta.len() {
                assert!((daido_jac[[j, l]] - mean_jac[[j, l]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_daido_mean_field_jacobian_matches_finite_difference() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let (k, m) = (2.5, 2.0);
        let jacobian = daido_mean_field_jacobian_inner(&theta, k, m);
        let h = 1e-6;
        for l in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[l] += h;
            minus[l] -= h;
            let force_plus = daido_mean_field_force_inner(&plus, k, m);
            let force_minus = daido_mean_field_force_inner(&minus, k, m);
            for j in 0..theta.len() {
                let fd = (force_plus[j] - force_minus[j]) / (2.0 * h);
                assert!((jacobian[[j, l]] - fd).abs() < 1e-6, "J[{j},{l}]");
            }
        }
    }

    #[test]
    fn test_daido_mean_field_jacobian_symmetric_and_rows_sum_to_zero() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5];
        let jacobian = daido_mean_field_jacobian_inner(&theta, 1.3, 3.0);
        let n = theta.len();
        for i in 0..n {
            for j in 0..n {
                assert!((jacobian[[i, j]] - jacobian[[j, i]]).abs() < 1e-15);
            }
            let row_sum: f64 = (0..n).map(|j| jacobian[[i, j]]).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_daido_mean_field_empty() {
        assert!(daido_mean_field_force_inner(&[], 1.0, 2.0).is_empty());
        assert_eq!(
            daido_mean_field_jacobian_inner(&[], 1.0, 2.0).shape(),
            &[0, 0]
        );
    }

    #[test]
    fn test_sakaguchi_mean_field_zero_frustration_matches_mean_field() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let k = 1.7;
        let sak_force = sakaguchi_mean_field_force_inner(&theta, k, 0.0);
        let mean_force = mean_field_force_inner(&theta, k);
        let sak_jac = sakaguchi_mean_field_jacobian_inner(&theta, k, 0.0);
        let mean_jac = mean_field_jacobian_inner(&theta, k);
        for j in 0..theta.len() {
            assert!((sak_force[j] - mean_force[j]).abs() < 1e-12);
            for l in 0..theta.len() {
                assert!((sak_jac[[j, l]] - mean_jac[[j, l]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_sakaguchi_mean_field_jacobian_matches_finite_difference() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let (k, alpha) = (2.5, 0.8);
        let jacobian = sakaguchi_mean_field_jacobian_inner(&theta, k, alpha);
        let h = 1e-6;
        for l in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[l] += h;
            minus[l] -= h;
            let force_plus = sakaguchi_mean_field_force_inner(&plus, k, alpha);
            let force_minus = sakaguchi_mean_field_force_inner(&minus, k, alpha);
            for j in 0..theta.len() {
                let fd = (force_plus[j] - force_minus[j]) / (2.0 * h);
                assert!((jacobian[[j, l]] - fd).abs() < 1e-6, "J[{j},{l}]");
            }
        }
    }

    #[test]
    fn test_sakaguchi_mean_field_jacobian_rows_sum_to_zero_and_break_symmetry() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5];
        let jacobian = sakaguchi_mean_field_jacobian_inner(&theta, 1.3, 0.7);
        let n = theta.len();
        let mut asymmetry = 0.0_f64;
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| jacobian[[i, j]]).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
            for j in 0..n {
                asymmetry += (jacobian[[i, j]] - jacobian[[j, i]]).abs();
            }
        }
        // Frustration breaks the variational symmetry.
        assert!(asymmetry > 1e-3);
    }

    #[test]
    fn test_sakaguchi_mean_field_empty() {
        assert!(sakaguchi_mean_field_force_inner(&[], 1.0, 0.5).is_empty());
        assert_eq!(
            sakaguchi_mean_field_jacobian_inner(&[], 1.0, 0.5).shape(),
            &[0, 0]
        );
    }

    #[test]
    fn test_triadic_mean_field_force_matches_squared_first_moment() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let k = 1.7;
        let force = triadic_mean_field_force_inner(&theta, k);
        let n = theta.len() as f64;
        let c: f64 = theta.iter().map(|&t| t.cos()).sum::<f64>() / n;
        let s: f64 = theta.iter().map(|&t| t.sin()).sum::<f64>() / n;
        // F_j = K r² sin(2ψ − 2θ_j), r² e^{2iψ} = (C + iS)².
        let r2 = c * c + s * s;
        let psi = s.atan2(c);
        for (j, &t) in theta.iter().enumerate() {
            let expected = k * r2 * (2.0 * psi - 2.0 * t).sin();
            assert!((force[j] - expected).abs() < 1e-12, "F[{j}]");
        }
    }

    #[test]
    fn test_triadic_mean_field_jacobian_matches_finite_difference() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let k = 2.5;
        let jacobian = triadic_mean_field_jacobian_inner(&theta, k);
        let h = 1e-6;
        for l in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[l] += h;
            minus[l] -= h;
            let force_plus = triadic_mean_field_force_inner(&plus, k);
            let force_minus = triadic_mean_field_force_inner(&minus, k);
            for j in 0..theta.len() {
                let fd = (force_plus[j] - force_minus[j]) / (2.0 * h);
                assert!((jacobian[[j, l]] - fd).abs() < 1e-6, "J[{j},{l}]");
            }
        }
    }

    #[test]
    fn test_triadic_mean_field_jacobian_rows_sum_to_zero_and_break_symmetry() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5];
        let jacobian = triadic_mean_field_jacobian_inner(&theta, 1.3);
        let n = theta.len();
        let mut asymmetry = 0.0_f64;
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| jacobian[[i, j]]).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
            for j in 0..n {
                asymmetry += (jacobian[[i, j]] - jacobian[[j, i]]).abs();
            }
        }
        // The higher-order mean field is non-variational.
        assert!(asymmetry > 1e-3);
    }

    #[test]
    fn test_triadic_mean_field_empty() {
        assert!(triadic_mean_field_force_inner(&[], 1.0).is_empty());
        assert_eq!(triadic_mean_field_jacobian_inner(&[], 1.0).shape(), &[0, 0]);
    }

    fn _symmetric_coupling(n: usize, seed: f64) -> Array2<f64> {
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                k[[i, j]] = ((i as f64 + 1.0) * (j as f64 + 1.0) * seed).sin().abs();
            }
        }
        (&k + &k.t()) * 0.5
    }

    #[test]
    fn test_networked_force_matches_closed_form() {
        let theta = vec![0.3, -1.1, 2.0, 0.7];
        let k = _symmetric_coupling(4, 0.37);
        let force = networked_kuramoto_force_inner(&theta, &k.view());
        for (j, &tj) in theta.iter().enumerate() {
            let expected: f64 = (0..4).map(|kk| k[[j, kk]] * (theta[kk] - tj).sin()).sum();
            assert!((force[j] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_networked_jacobian_matches_finite_difference_of_force() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let k = _symmetric_coupling(5, 0.21);
        let jacobian = networked_kuramoto_jacobian_inner(&theta, &k.view());
        let h = 1e-6;
        for l in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[l] += h;
            minus[l] -= h;
            let force_plus = networked_kuramoto_force_inner(&plus, &k.view());
            let force_minus = networked_kuramoto_force_inner(&minus, &k.view());
            for j in 0..theta.len() {
                let fd = (force_plus[j] - force_minus[j]) / (2.0 * h);
                assert!((jacobian[[j, l]] - fd).abs() < 1e-6, "J[{j},{l}]");
            }
        }
    }

    #[test]
    fn test_networked_jacobian_symmetric_for_symmetric_coupling_with_zero_rows() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5];
        let k = _symmetric_coupling(5, 0.5);
        let jacobian = networked_kuramoto_jacobian_inner(&theta, &k.view());
        let n = theta.len();
        for i in 0..n {
            for j in 0..n {
                assert!((jacobian[[i, j]] - jacobian[[j, i]]).abs() < 1e-15);
            }
            let row_sum: f64 = (0..n).map(|j| jacobian[[i, j]]).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_networked_independent_of_coupling_diagonal() {
        let theta = vec![0.4, 1.2, 2.9, 0.1];
        let mut k = _symmetric_coupling(4, 0.8);
        let force_a = networked_kuramoto_force_inner(&theta, &k.view());
        let jacobian_a = networked_kuramoto_jacobian_inner(&theta, &k.view());
        for i in 0..4 {
            k[[i, i]] = 7.0;
        }
        let force_b = networked_kuramoto_force_inner(&theta, &k.view());
        let jacobian_b = networked_kuramoto_jacobian_inner(&theta, &k.view());
        for j in 0..4 {
            assert!((force_a[j] - force_b[j]).abs() < 1e-15);
            for l in 0..4 {
                assert!((jacobian_a[[j, l]] - jacobian_b[[j, l]]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_networked_empty() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(networked_kuramoto_force_inner(&[], &empty.view()).is_empty());
        assert_eq!(
            networked_kuramoto_jacobian_inner(&[], &empty.view()).shape(),
            &[0, 0]
        );
    }

    #[test]
    fn test_interaction_energy_matches_closed_form() {
        let theta = vec![0.3, -1.1, 2.0, 0.7];
        let k = _symmetric_coupling(4, 0.37);
        let energy = kuramoto_interaction_energy_inner(&theta, &k.view());
        let mut expected = 0.0;
        for j in 0..4 {
            for kk in 0..4 {
                expected += k[[j, kk]] * (theta[j] - theta[kk]).cos();
            }
        }
        assert!((energy - (-0.5 * expected)).abs() < 1e-12);
    }

    #[test]
    fn test_interaction_energy_gradient_matches_finite_difference() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        // asymmetric coupling to exercise the (K_jk + K_kj) symmetrisation
        let mut k = _symmetric_coupling(5, 0.21);
        k[[0, 1]] += 0.9;
        let gradient = kuramoto_interaction_energy_gradient_inner(&theta, &k.view());
        let h = 1e-6;
        for j in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[j] += h;
            minus[j] -= h;
            let energy_plus = kuramoto_interaction_energy_inner(&plus, &k.view());
            let energy_minus = kuramoto_interaction_energy_inner(&minus, &k.view());
            let fd = (energy_plus - energy_minus) / (2.0 * h);
            assert!((gradient[j] - fd).abs() < 1e-6, "grad[{j}]");
        }
    }

    #[test]
    fn test_interaction_energy_gradient_sums_to_zero_and_equals_negated_force() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5];
        let k = _symmetric_coupling(5, 0.5);
        let gradient = kuramoto_interaction_energy_gradient_inner(&theta, &k.view());
        let force = networked_kuramoto_force_inner(&theta, &k.view());
        let sum: f64 = gradient.iter().sum();
        assert!(sum.abs() < 1e-12);
        for j in 0..theta.len() {
            assert!(
                (gradient[j] + force[j]).abs() < 1e-12,
                "grad != -force at {j}"
            );
        }
    }

    #[test]
    fn test_interaction_energy_empty() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert_eq!(kuramoto_interaction_energy_inner(&[], &empty.view()), 0.0);
        assert!(kuramoto_interaction_energy_gradient_inner(&[], &empty.view()).is_empty());
    }

    #[test]
    fn test_interaction_energy_hessian_matches_finite_difference_of_gradient() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let mut k = _symmetric_coupling(5, 0.21);
        k[[0, 1]] += 0.9; // asymmetric to exercise the (K_il + K_li) symmetrisation
        let hessian = kuramoto_interaction_energy_hessian_inner(&theta, &k.view());
        let h = 1e-6;
        for l in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[l] += h;
            minus[l] -= h;
            let grad_plus = kuramoto_interaction_energy_gradient_inner(&plus, &k.view());
            let grad_minus = kuramoto_interaction_energy_gradient_inner(&minus, &k.view());
            for i in 0..theta.len() {
                let fd = (grad_plus[i] - grad_minus[i]) / (2.0 * h);
                assert!((hessian[[i, l]] - fd).abs() < 1e-6, "H[{i},{l}]");
            }
        }
    }

    #[test]
    fn test_interaction_energy_hessian_symmetric_rows_zero_equals_negated_jacobian() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5];
        let k = _symmetric_coupling(5, 0.5);
        let hessian = kuramoto_interaction_energy_hessian_inner(&theta, &k.view());
        let jacobian = networked_kuramoto_jacobian_inner(&theta, &k.view());
        let n = theta.len();
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| hessian[[i, j]]).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
            for j in 0..n {
                assert!((hessian[[i, j]] - hessian[[j, i]]).abs() < 1e-15);
                // symmetric K => Hessian = -networked Jacobian
                assert!((hessian[[i, j]] + jacobian[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_interaction_energy_hessian_empty() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert_eq!(
            kuramoto_interaction_energy_hessian_inner(&[], &empty.view()).shape(),
            &[0, 0]
        );
    }

    #[test]
    fn test_sakaguchi_force_matches_closed_form() {
        let theta = vec![0.3, -1.1, 2.0, 0.7];
        let k = _symmetric_coupling(4, 0.37);
        let alpha = 0.4;
        let force = sakaguchi_force_inner(&theta, &k.view(), alpha);
        for (j, &tj) in theta.iter().enumerate() {
            let expected: f64 = (0..4)
                .filter(|&kk| kk != j)
                .map(|kk| k[[j, kk]] * (theta[kk] - tj - alpha).sin())
                .sum();
            assert!((force[j] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sakaguchi_jacobian_matches_finite_difference() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let k = _symmetric_coupling(5, 0.21);
        let alpha = 0.7;
        let jacobian = sakaguchi_jacobian_inner(&theta, &k.view(), alpha);
        let h = 1e-6;
        for l in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[l] += h;
            minus[l] -= h;
            let force_plus = sakaguchi_force_inner(&plus, &k.view(), alpha);
            let force_minus = sakaguchi_force_inner(&minus, &k.view(), alpha);
            for j in 0..theta.len() {
                let fd = (force_plus[j] - force_minus[j]) / (2.0 * h);
                assert!((jacobian[[j, l]] - fd).abs() < 1e-6, "J[{j},{l}]");
            }
        }
    }

    #[test]
    fn test_sakaguchi_jacobian_rows_sum_to_zero_and_break_symmetry() {
        let theta = vec![0.1, 0.9, 2.3, 3.1, 5.5];
        let k = _symmetric_coupling(5, 0.5);
        let jacobian = sakaguchi_jacobian_inner(&theta, &k.view(), 0.6);
        let n = theta.len();
        let mut max_asymmetry = 0.0_f64;
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| jacobian[[i, j]]).sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
            for j in 0..n {
                max_asymmetry = max_asymmetry.max((jacobian[[i, j]] - jacobian[[j, i]]).abs());
            }
        }
        // frustration breaks reciprocity even for symmetric coupling
        assert!(max_asymmetry > 1e-2);
    }

    #[test]
    fn test_sakaguchi_reduces_to_networked_at_zero_frustration() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let k = _symmetric_coupling(5, 0.31);
        let sakaguchi = sakaguchi_force_inner(&theta, &k.view(), 0.0);
        let networked = networked_kuramoto_force_inner(&theta, &k.view());
        for j in 0..theta.len() {
            assert!((sakaguchi[j] - networked[j]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sakaguchi_empty() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(sakaguchi_force_inner(&[], &empty.view(), 0.5).is_empty());
        assert_eq!(
            sakaguchi_jacobian_inner(&[], &empty.view(), 0.5).shape(),
            &[0, 0]
        );
    }

    #[test]
    fn test_local_order_parameter_matches_closed_form() {
        let theta = vec![0.3, -1.1, 2.0, 0.7];
        let a = _symmetric_coupling(4, 0.37);
        let local = local_order_parameter_inner(&theta, &a.view());
        for j in 0..4 {
            let mut c = 0.0;
            let mut s = 0.0;
            let mut d = 0.0;
            for k in 0..4 {
                c += a[[j, k]] * theta[k].cos();
                s += a[[j, k]] * theta[k].sin();
                d += a[[j, k]];
            }
            assert!((local[j] - (c * c + s * s).sqrt() / d).abs() < 1e-12);
        }
    }

    #[test]
    fn test_local_order_parameter_all_to_all_equals_global() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2, 1.3];
        let ones = Array2::<f64>::ones((6, 6));
        let local = local_order_parameter_inner(&theta, &ones.view());
        let global = order_parameter_inner(&theta);
        for value in local.iter() {
            assert!((value - global).abs() < 1e-12);
        }
    }

    #[test]
    fn test_local_order_parameter_jacobian_matches_finite_difference() {
        let theta = vec![0.3, -1.1, 2.0, 0.7, 4.2];
        let a = _symmetric_coupling(5, 0.21);
        let jacobian = local_order_parameter_jacobian_inner(&theta, &a.view());
        let h = 1e-6;
        for l in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[l] += h;
            minus[l] -= h;
            let value_plus = local_order_parameter_inner(&plus, &a.view());
            let value_minus = local_order_parameter_inner(&minus, &a.view());
            for j in 0..theta.len() {
                let fd = (value_plus[j] - value_minus[j]) / (2.0 * h);
                assert!((jacobian[[j, l]] - fd).abs() < 1e-6, "J[{j},{l}]");
            }
        }
    }

    #[test]
    fn test_local_order_parameter_zero_degree_node() {
        let theta = vec![0.3, 1.1, 2.0];
        let mut a = _symmetric_coupling(3, 0.4);
        // isolate node 1: zero its row and column
        for k in 0..3 {
            a[[1, k]] = 0.0;
            a[[k, 1]] = 0.0;
        }
        let local = local_order_parameter_inner(&theta, &a.view());
        let jacobian = local_order_parameter_jacobian_inner(&theta, &a.view());
        assert_eq!(local[1], 0.0);
        for l in 0..3 {
            assert_eq!(jacobian[[1, l]], 0.0);
        }
    }

    #[test]
    fn test_local_order_parameter_empty() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(local_order_parameter_inner(&[], &empty.view()).is_empty());
        assert_eq!(
            local_order_parameter_jacobian_inner(&[], &empty.view()).shape(),
            &[0, 0]
        );
    }

    #[test]
    fn test_local_mean_phase_all_to_all_equals_global() {
        let theta = vec![0.2, 1.1, -0.7, 2.4, 3.3];
        let n = theta.len();
        let mut adjacency = Array2::<f64>::ones((n, n));
        for i in 0..n {
            adjacency[[i, i]] = 0.0;
        }
        let local = local_mean_phase_inner(&theta, &adjacency.view());
        // With the all-to-all adjacency the local complex order excludes only the self term,
        // so the per-node phase tracks the global mean phase of the remaining oscillators.
        for j in 0..n {
            let (mut c, mut s) = (0.0_f64, 0.0_f64);
            for k in 0..n {
                if k != j {
                    c += theta[k].cos();
                    s += theta[k].sin();
                }
            }
            assert!((local[j] - s.atan2(c)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_local_mean_phase_jacobian_matches_finite_difference() {
        let theta = vec![0.2, 1.1, -0.7, 2.4, 3.3];
        let adjacency = _symmetric_coupling(theta.len(), 0.37);
        let jacobian = local_mean_phase_jacobian_inner(&theta, &adjacency.view());
        let h = 1e-6;
        for l in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[l] += h;
            minus[l] -= h;
            let psi_plus = local_mean_phase_inner(&plus, &adjacency.view());
            let psi_minus = local_mean_phase_inner(&minus, &adjacency.view());
            for j in 0..theta.len() {
                let mut delta = psi_plus[j] - psi_minus[j];
                delta = (delta + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
                    - std::f64::consts::PI;
                let fd = delta / (2.0 * h);
                assert!((jacobian[[j, l]] - fd).abs() < 1e-6, "J[{j},{l}]");
            }
        }
    }

    #[test]
    fn test_local_mean_phase_zero_degree_node() {
        let theta = vec![0.3, 1.2, 2.1];
        let mut adjacency = Array2::<f64>::ones((3, 3));
        for k in 0..3 {
            adjacency[[1, k]] = 0.0;
        }
        let phase = local_mean_phase_inner(&theta, &adjacency.view());
        let jacobian = local_mean_phase_jacobian_inner(&theta, &adjacency.view());
        assert_eq!(phase[1], 0.0);
        for l in 0..3 {
            assert_eq!(jacobian[[1, l]], 0.0);
        }
    }

    #[test]
    fn test_local_mean_phase_empty() {
        let empty = Array2::<f64>::zeros((0, 0));
        assert!(local_mean_phase_inner(&[], &empty.view()).is_empty());
        assert_eq!(
            local_mean_phase_jacobian_inner(&[], &empty.view()).shape(),
            &[0, 0]
        );
    }
}
