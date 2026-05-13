// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — U(1) Lattice Gauge Computation

//! Rust-accelerated hot paths for the Ψ-field lattice gauge simulator.
//!
//! Implements the compute-intensive parts of HMC: plaquette action,
//! force computation, gauge-covariant kinetic energy, and topological
//! charge. These are the inner loops that dominate wall-clock time
//! for large lattices.
//!
//! Ref: Creutz, "Quarks, Gluons and Lattices" (1983)

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::validation::{
    validate_contiguous_slice, validate_finite, validate_n, validate_positive,
};

fn validate_gauge_scalar(value: f64, name: &str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite, got {value}"
        )));
    }
    Ok(())
}

fn validate_triangle_inputs<'a>(
    links: &'a PyReadonlyArray1<'_, f64>,
    triangles: &'a PyReadonlyArray1<'_, i64>,
    triangle_signs: &'a PyReadonlyArray1<'_, f64>,
    n_triangles: usize,
    force_edges: Option<usize>,
) -> PyResult<(&'a [f64], &'a [i64], &'a [f64])> {
    let a = validate_contiguous_slice(links, "links")?;
    let tri = validate_contiguous_slice(triangles, "triangles")?;
    let signs = validate_contiguous_slice(triangle_signs, "triangle_signs")?;
    validate_finite(a, "links")?;
    validate_finite(signs, "triangle_signs")?;
    let expected = n_triangles
        .checked_mul(3)
        .ok_or_else(|| PyValueError::new_err("n_triangles*3 overflows usize"))?;
    if tri.len() < expected {
        return Err(PyValueError::new_err(format!(
            "triangles length {} < 3*n_triangles = {expected}",
            tri.len()
        )));
    }
    if signs.len() < expected {
        return Err(PyValueError::new_err(format!(
            "triangle_signs length {} < 3*n_triangles = {expected}",
            signs.len()
        )));
    }
    for (idx, &edge) in tri.iter().take(expected).enumerate() {
        if edge < 0 {
            return Err(PyValueError::new_err(format!(
                "triangles[{idx}]={edge} must be non-negative"
            )));
        }
        let edge_idx = edge as usize;
        if edge_idx >= a.len() {
            return Err(PyValueError::new_err(format!(
                "triangles[{idx}]={edge_idx} exceeds links length {}",
                a.len()
            )));
        }
        if let Some(n_edges) = force_edges {
            if edge_idx >= n_edges {
                return Err(PyValueError::new_err(format!(
                    "triangles[{idx}]={edge_idx} exceeds n_edges {n_edges}"
                )));
            }
        }
    }
    Ok((a, tri, signs))
}

/// Compute total plaquette action and mean plaquette for triangle plaquettes.
///
/// Each plaquette (i,j,k) contributes Re(U_ij U_jk U_ki†) = cos(A_ij + A_jk − A_ik).
/// Returns (mean_plaquette, total_action).
///
/// edges: flat array [i0,j0, i1,j1, ...] — edge list
/// links: A values for each edge (same order as edges)
/// triangles: flat array [e0,e1,e2, ...] — indices into edges, 3 per triangle
///            with sign convention: e0=+, e1=+, e2=− (last edge reversed)
/// n_triangles: number of triangles
/// beta: coupling constant
#[pyfunction]
pub fn plaquette_action_batch(
    links: PyReadonlyArray1<'_, f64>,
    triangles: PyReadonlyArray1<'_, i64>,
    triangle_signs: PyReadonlyArray1<'_, f64>,
    n_triangles: usize,
    beta: f64,
) -> PyResult<(f64, f64)> {
    validate_positive(beta, "beta")?;
    let (a, tri, signs) =
        validate_triangle_inputs(&links, &triangles, &triangle_signs, n_triangles, None)?;

    let mut total = 0.0f64;

    for t in 0..n_triangles {
        let base = t * 3;
        let mut phase = 0.0f64;
        for k in 0..3 {
            let edge_idx = tri[base + k] as usize;
            let sign = signs[base + k];
            phase += sign * a[edge_idx];
        }
        total += phase.cos();
    }

    let mean = if n_triangles > 0 {
        total / n_triangles as f64
    } else {
        0.0
    };
    let action = -beta * total;
    Ok((mean, action))
}

/// Compute force dS/dA for each link from triangle plaquettes.
///
/// For each triangle containing link e with sign s:
///   dS/dA_e += β × s × sin(phase_triangle)
#[pyfunction]
pub fn gauge_force_batch<'py>(
    py: Python<'py>,
    links: PyReadonlyArray1<'_, f64>,
    triangles: PyReadonlyArray1<'_, i64>,
    triangle_signs: PyReadonlyArray1<'_, f64>,
    n_triangles: usize,
    n_edges: usize,
    beta: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_positive(beta, "beta")?;
    validate_n(n_edges, "n_edges")?;
    let (a, tri, signs) = validate_triangle_inputs(
        &links,
        &triangles,
        &triangle_signs,
        n_triangles,
        Some(n_edges),
    )?;

    let mut force = vec![0.0f64; n_edges];

    for t in 0..n_triangles {
        let base = t * 3;
        let mut phase = 0.0f64;
        for k in 0..3 {
            let edge_idx = tri[base + k] as usize;
            let sign = signs[base + k];
            phase += sign * a[edge_idx];
        }
        let sin_phase = phase.sin();

        for k in 0..3 {
            let edge_idx = tri[base + k] as usize;
            let sign = signs[base + k];
            force[edge_idx] += beta * sign * sin_phase;
        }
    }

    Ok(PyArray1::from_vec(py, force))
}

/// Gauge-covariant kinetic energy (hopping term).
///
/// T = Σ_edges (|φ_i|² + |φ_j|² − 2 Re(φ_i* × exp(igA_ij) × φ_j))
///
/// Takes real/imag parts of φ separately.
#[pyfunction]
pub fn gauge_covariant_kinetic_rust(
    phi_re: PyReadonlyArray1<'_, f64>,
    phi_im: PyReadonlyArray1<'_, f64>,
    links: PyReadonlyArray1<'_, f64>,
    edges: PyReadonlyArray2<'_, i64>,
    g_coupling: f64,
) -> PyResult<f64> {
    validate_gauge_scalar(g_coupling, "g_coupling")?;
    let pr = validate_contiguous_slice(&phi_re, "phi_re")?;
    let pi = validate_contiguous_slice(&phi_im, "phi_im")?;
    let a = validate_contiguous_slice(&links, "links")?;
    validate_finite(pr, "phi_re")?;
    validate_finite(pi, "phi_im")?;
    validate_finite(a, "links")?;
    if pi.len() != pr.len() {
        return Err(PyValueError::new_err(format!(
            "phi_im length {} != phi_re length {}",
            pi.len(),
            pr.len()
        )));
    }
    let e = edges.as_array();
    if e.ndim() != 2 || e.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "edges must have shape (n_edges, 2), got {:?}",
            e.shape()
        )));
    }
    let n_edges = e.nrows();
    if a.len() != n_edges {
        return Err(PyValueError::new_err(format!(
            "links length {} != edges rows {n_edges}",
            a.len()
        )));
    }
    for idx in 0..n_edges {
        for col in 0..2 {
            let vertex = e[[idx, col]];
            if vertex < 0 {
                return Err(PyValueError::new_err(format!(
                    "edges[{idx},{col}]={vertex} must be non-negative"
                )));
            }
            if vertex as usize >= pr.len() {
                return Err(PyValueError::new_err(format!(
                    "edges[{idx},{col}]={} exceeds phi length {}",
                    vertex,
                    pr.len()
                )));
            }
        }
    }

    let mut total = 0.0f64;

    for idx in 0..n_edges {
        let i = e[[idx, 0]] as usize;
        let j = e[[idx, 1]] as usize;
        let a_ij = a[idx];

        let rho_i = pr[i] * pr[i] + pi[i] * pi[i];
        let rho_j = pr[j] * pr[j] + pi[j] * pi[j];

        // U_ij = exp(igA_ij) = cos(gA) + i sin(gA)
        let cos_ga = (g_coupling * a_ij).cos();
        let sin_ga = (g_coupling * a_ij).sin();

        // φ_i* × U_ij × φ_j = (pr_i − i×pi_i)(cos+i×sin)(pr_j + i×pi_j)
        // Re part = pr_i(cos×pr_j − sin×pi_j) + pi_i(sin×pr_j + cos×pi_j)
        let hopping_re =
            pr[i] * (cos_ga * pr[j] - sin_ga * pi[j]) + pi[i] * (sin_ga * pr[j] + cos_ga * pi[j]);

        total += rho_i + rho_j - 2.0 * hopping_re;
    }

    Ok(total)
}

/// Topological charge Q = (1/2π) Σ_plaq wrap(phase_plaq).
#[pyfunction]
pub fn topological_charge_rust(
    links: PyReadonlyArray1<'_, f64>,
    triangles: PyReadonlyArray1<'_, i64>,
    triangle_signs: PyReadonlyArray1<'_, f64>,
    n_triangles: usize,
) -> PyResult<f64> {
    let (a, tri, signs) =
        validate_triangle_inputs(&links, &triangles, &triangle_signs, n_triangles, None)?;
    let pi = std::f64::consts::PI;
    let two_pi = 2.0 * pi;

    let mut q = 0.0f64;

    for t in 0..n_triangles {
        let base = t * 3;
        let mut phase = 0.0f64;
        for k in 0..3 {
            let edge_idx = tri[base + k] as usize;
            let sign = signs[base + k];
            phase += sign * a[edge_idx];
        }
        // Wrap to [−π, π)
        let wrapped = ((phase + pi) % two_pi + two_pi) % two_pi - pi;
        q += wrapped;
    }

    Ok(q / two_pi)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_plaquette_zero_links() {
        // All links = 0 → cos(0) = 1 for every plaquette
        let links = vec![0.0; 3];
        let tri = vec![0i64, 1, 2];
        let signs = vec![1.0, 1.0, -1.0];
        let (mean, action) = plaquette_action_batch_inner(&links, &tri, &signs, 1, 2.0);
        assert!((mean - 1.0).abs() < 1e-12, "zero links → cos(0) = 1");
        assert!((action - (-2.0)).abs() < 1e-12, "action = −β × 1");
    }

    #[test]
    fn test_topological_charge_zero() {
        // Smooth field (all zero) → Q = 0
        let links = vec![0.0; 3];
        let tri = vec![0i64, 1, 2];
        let signs = vec![1.0, 1.0, -1.0];
        let q = topological_charge_inner(&links, &tri, &signs, 1);
        assert!(q.abs() < 1e-12, "smooth field → Q = 0");
    }

    #[test]
    fn test_force_zero_at_minimum() {
        // All links = 0 → phase = 0 → sin(0) = 0 → force = 0
        let links = vec![0.0; 3];
        let tri = vec![0i64, 1, 2];
        let signs = vec![1.0, 1.0, -1.0];
        let force = gauge_force_inner(&links, &tri, &signs, 1, 3, 2.0);
        for f in &force {
            assert!(f.abs() < 1e-12, "minimum → zero force");
        }
    }

    #[test]
    fn test_kinetic_zero_field() {
        // φ = 0 everywhere → T = 0
        let pr = vec![0.0; 3];
        let pi_arr = vec![0.0; 3];
        let links = vec![0.1, 0.2, 0.3];
        let edges: Vec<[usize; 2]> = vec![[0, 1], [1, 2], [0, 2]];
        let t = kinetic_inner(&pr, &pi_arr, &links, &edges, 1.0);
        assert!(t.abs() < 1e-12);
    }

    // Pure Rust helpers for testing
    fn plaquette_action_batch_inner(
        links: &[f64],
        tri: &[i64],
        signs: &[f64],
        n_tri: usize,
        beta: f64,
    ) -> (f64, f64) {
        let mut total = 0.0f64;
        for t in 0..n_tri {
            let base = t * 3;
            let mut phase = 0.0;
            for k in 0..3 {
                phase += signs[base + k] * links[tri[base + k] as usize];
            }
            total += phase.cos();
        }
        let mean = total / n_tri.max(1) as f64;
        (mean, -beta * total)
    }

    fn topological_charge_inner(links: &[f64], tri: &[i64], signs: &[f64], n_tri: usize) -> f64 {
        let pi = std::f64::consts::PI;
        let two_pi = 2.0 * pi;
        let mut q = 0.0;
        for t in 0..n_tri {
            let base = t * 3;
            let mut phase = 0.0;
            for k in 0..3 {
                phase += signs[base + k] * links[tri[base + k] as usize];
            }
            let wrapped = ((phase + pi) % two_pi + two_pi) % two_pi - pi;
            q += wrapped;
        }
        q / two_pi
    }

    fn gauge_force_inner(
        links: &[f64],
        tri: &[i64],
        signs: &[f64],
        n_tri: usize,
        n_edges: usize,
        beta: f64,
    ) -> Vec<f64> {
        let mut force = vec![0.0f64; n_edges];
        for t in 0..n_tri {
            let base = t * 3;
            let mut phase = 0.0;
            for k in 0..3 {
                phase += signs[base + k] * links[tri[base + k] as usize];
            }
            let s = phase.sin();
            for k in 0..3 {
                force[tri[base + k] as usize] += beta * signs[base + k] * s;
            }
        }
        force
    }

    fn kinetic_inner(pr: &[f64], pi: &[f64], links: &[f64], edges: &[[usize; 2]], g: f64) -> f64 {
        let mut total = 0.0;
        for (idx, &[i, j]) in edges.iter().enumerate() {
            let rho_i = pr[i] * pr[i] + pi[i] * pi[i];
            let rho_j = pr[j] * pr[j] + pi[j] * pi[j];
            let cos_ga = (g * links[idx]).cos();
            let sin_ga = (g * links[idx]).sin();
            let hop = pr[i] * (cos_ga * pr[j] - sin_ga * pi[j])
                + pi[i] * (sin_ga * pr[j] + cos_ga * pi[j]);
            total += rho_i + rho_j - 2.0 * hop;
        }
        total
    }
}
