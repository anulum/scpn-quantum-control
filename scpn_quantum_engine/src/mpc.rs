// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Brute-Force Model Predictive Control

//! Brute-force optimal binary MPC for quantum control.
//!
//! Enumerates all 2^horizon action sequences and evaluates cost in parallel
//! via rayon. Returns the optimal action sequence, cost, and full cost landscape.
//! Used by the QAOA-MPC module for benchmarking against quantum optimisers.
//!
//! The tracking cost is `C(u) = sum_t ||u_t * v - r||^2` with `v = B * 1` the
//! row sums of the actuation matrix, `u_t` a binary on/off decision and `r` the
//! target. The residual stays a vector: collapsing it to `||B||` and `||r||`
//! would discard the target's sign and its direction relative to `B`. This must
//! stay numerically identical to the Python fallback in
//! `scpn_quantum_control.hardware.classical.classical_brute_mpc`.

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::validation::{
    validate_contiguous_slice, validate_finite, validate_flat_square, validate_n,
};

type BruteMpcResult<'py> = PyResult<(
    Bound<'py, PyArray1<i64>>,
    f64,
    Bound<'py, PyArray1<f64>>,
    usize,
)>;

/// Brute-force optimal binary MPC: enumerate all 2^horizon action sequences.
/// Cost enumeration uses rayon for every admitted horizon.
///
/// Evaluates `C(u) = sum_t ||u_t * v - r||^2` where `v = B * 1` is the row-sum
/// actuation vector of the `dim x dim` matrix `b_flat` (row-major) and `r` is
/// `target`. Bit `t` of an enumeration index carries `u_t`.
///
/// Returns (optimal_actions, optimal_cost, all_costs, n_evaluated).
#[pyfunction]
pub fn brute_mpc<'py>(
    py: Python<'py>,
    b_flat: PyReadonlyArray1<'_, f64>,
    target: PyReadonlyArray1<'_, f64>,
    dim: usize,
    horizon: usize,
) -> BruteMpcResult<'py> {
    validate_n(dim, "dim")?;
    validate_n(horizon, "horizon")?;
    if horizon > 25 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "horizon={horizon} too large (max 25, would allocate 2^{horizon} entries)"
        )));
    }
    let b_data = validate_contiguous_slice(&b_flat, "b_flat")?;
    let t_data = validate_contiguous_slice(&target, "target")?;
    validate_n(b_data.len(), "b_flat")?;
    validate_n(t_data.len(), "target")?;
    validate_flat_square(b_data, dim, "b_flat")?;
    if t_data.len() != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "target length {} != dim {dim}",
            t_data.len()
        )));
    }
    validate_finite(b_data, "b_flat")?;
    validate_finite(t_data, "target")?;
    let (best_actions, best_cost, costs, n_actions) = solve_mpc(b_data, t_data, dim, horizon);
    let actions_arr = Array1::from_vec(best_actions);
    let costs_arr = Array1::from_vec(costs);

    Ok((
        PyArray1::from_owned_array(py, actions_arr),
        best_cost,
        PyArray1::from_owned_array(py, costs_arr),
        n_actions,
    ))
}

/// Evaluate the validated binary tracking problem used by the Python boundary.
///
/// Inputs must satisfy the shape, finite-value and horizon guards in `brute_mpc`.
/// Returns little-endian actions, minimum cost, full landscape and its size.
fn solve_mpc(
    b_data: &[f64],
    t_data: &[f64],
    dim: usize,
    horizon: usize,
) -> (Vec<i64>, f64, Vec<f64>, usize) {
    let n_actions = 1usize << horizon;

    let actuation: Vec<f64> = (0..dim)
        .map(|row| b_data[row * dim..(row + 1) * dim].iter().sum())
        .collect();

    let costs: Vec<f64> = (0..n_actions)
        .into_par_iter()
        .map(|idx| {
            let mut cost = 0.0;
            for t in 0..horizon {
                let action = ((idx >> t) & 1) as f64;
                for (component, target_component) in actuation.iter().zip(t_data.iter()) {
                    let residual = action * component - target_component;
                    cost += residual * residual;
                }
            }
            cost
        })
        .collect();

    let mut best_idx = 0usize;
    let mut best_cost = costs[0];
    for (idx, &cost) in costs.iter().enumerate() {
        if cost < best_cost {
            best_cost = cost;
            best_idx = idx;
        }
    }

    let best_actions: Vec<i64> = (0..horizon)
        .map(|bit| ((best_idx >> bit) & 1) as i64)
        .collect();

    (best_actions, best_cost, costs, n_actions)
}

#[cfg(test)]
mod tests {
    use super::solve_mpc;

    /// Equal-cost sequences select the first little-endian enumeration index.
    #[test]
    fn tied_landscape_keeps_first_sequence() {
        let (actions, best, costs, evaluated) = solve_mpc(&[0.0], &[2.0], 1, 3);
        assert_eq!(actions, vec![0, 0, 0]);
        assert_eq!(best, 12.0);
        assert_eq!(costs, vec![12.0; 8]);
        assert_eq!(evaluated, 8);
    }

    /// Independent oracle: the cost straight from its definition, with no
    /// algebraic rearrangement shared with the kernel under test.
    fn oracle(b_flat: &[f64], target: &[f64], dim: usize, horizon: usize) -> Vec<f64> {
        let actuation: Vec<f64> = (0..dim)
            .map(|row| b_flat[row * dim..(row + 1) * dim].iter().sum())
            .collect();
        (0..1usize << horizon)
            .map(|idx| {
                let mut cost = 0.0;
                for t in 0..horizon {
                    let action = ((idx >> t) & 1) as f64;
                    for i in 0..dim {
                        let residual = action * actuation[i] - target[i];
                        cost += residual * residual;
                    }
                }
                cost
            })
            .collect()
    }

    /// Reproduce the recorded defect: a negative target must not be treated
    /// like its positive twin. The norm-only surrogate returned `[1, 0]` and
    /// selected `u = 1`; the documented cost is `(u + 1)^2 = [1, 4]`.
    #[test]
    fn signed_target_is_not_collapsed_to_a_norm() {
        let (actions, best, costs, evaluated) = solve_mpc(&[1.0], &[-1.0], 1, 1);
        assert_eq!(actions, vec![0]);
        assert_eq!(evaluated, 2);
        assert!((best - 1.0).abs() < 1e-12);
        assert!((costs[0] - 1.0).abs() < 1e-12, "u=0 costs (0+1)^2 = 1");
        assert!((costs[1] - 4.0).abs() < 1e-12, "u=1 costs (1+1)^2 = 4");
        assert!(costs[0] < costs[1], "the optimum is u=0, not u=1");

        let flipped = solve_mpc(&[1.0], &[1.0], 1, 1).2;
        assert!(
            (flipped[1] - 0.0).abs() < 1e-12,
            "flipping the target's sign must change the landscape"
        );
        assert!(costs[1] != flipped[1], "a norm-only cost would tie these");
    }

    /// A rotated target changes the answer even at fixed norms, because the
    /// `v . r` cross-term survives.
    #[test]
    fn rotated_target_changes_the_optimum() {
        let b = vec![0.6, -0.8, 0.8, 0.6];
        let aligned = solve_mpc(&b, &[-0.2, 1.4], 2, 2).2;
        let rotated = solve_mpc(&b, &[1.4, -0.2], 2, 2).2;
        assert_eq!(aligned, oracle(&b, &[-0.2, 1.4], 2, 2));
        assert_eq!(rotated, oracle(&b, &[1.4, -0.2], 2, 2));
        assert!(
            aligned
                .iter()
                .zip(rotated.iter())
                .any(|(a, r)| (a - r).abs() > 1e-9),
            "equal-norm targets with different directions must differ"
        );
    }

    /// Sparse and multi-step landscapes stay consistent with the definition.
    #[test]
    fn multi_step_landscape_matches_the_definition() {
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let target = vec![0.8, 0.6];
        let (actions, best, costs, evaluated) = solve_mpc(&b, &target, 2, 3);
        assert_eq!(costs, oracle(&b, &target, 2, 3));
        assert_eq!(actions, vec![1, 1, 1]);
        assert_eq!(evaluated, 8);
        assert!((best - costs[7]).abs() < 1e-12);
        assert_eq!(costs.len(), 8);
        let per_step_off = 0.8f64.powi(2) + 0.6f64.powi(2);
        let per_step_on = (1.0f64 - 0.8).powi(2) + (1.0f64 - 0.6).powi(2);
        assert!((costs[0] - 3.0 * per_step_off).abs() < 1e-12);
        assert!((costs[7] - 3.0 * per_step_on).abs() < 1e-12);
        assert!((costs[1] - (per_step_on + 2.0 * per_step_off)).abs() < 1e-12);
    }
}
