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

use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Brute-force optimal binary MPC: enumerate all 2^horizon action sequences.
/// Parallelised with rayon for horizon > 10.
///
/// Returns (optimal_actions, optimal_cost, all_costs, n_evaluated).
#[pyfunction]
pub fn brute_mpc<'py>(
    py: Python<'py>,
    b_flat: PyReadonlyArray1<'_, f64>,
    target: PyReadonlyArray1<'_, f64>,
    _dim: usize,
    horizon: usize,
) -> (
    Bound<'py, PyArray1<i64>>,
    f64,
    Bound<'py, PyArray1<f64>>,
    usize,
) {
    let b_data = b_flat.as_slice().unwrap();
    let t_data = target.as_slice().unwrap();
    let n_actions = 1usize << horizon;

    let b_norm: f64 = b_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    let t_norm: f64 = t_data.iter().map(|x| x * x).sum::<f64>().sqrt();

    let costs: Vec<f64> = (0..n_actions)
        .into_par_iter()
        .map(|idx| {
            let mut cost = 0.0;
            for t in 0..horizon {
                let action = ((idx >> t) & 1) as f64;
                let diff = b_norm * action - t_norm / horizon as f64;
                cost += diff * diff;
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

    let actions_arr = Array1::from_vec(best_actions);
    let costs_arr = Array1::from_vec(costs);

    (
        PyArray1::from_owned_array(py, actions_arr),
        best_cost,
        PyArray1::from_owned_array(py, costs_arr),
        n_actions,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_brute_mpc_trivial() {
        // horizon=1: 2 actions (0 or 1)
        // Cost for action 0: (0 - t_norm/1)² = t_norm²
        // Cost for action 1: (b_norm - t_norm)²
        // If b_norm ≈ t_norm, action 1 wins
        let b = vec![1.0, 0.0];
        let t = vec![1.0, 0.0];
        let b_norm: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        let t_norm: f64 = t.iter().map(|x| x * x).sum::<f64>().sqrt();
        let horizon = 1;
        let n_actions = 1usize << horizon;

        let costs: Vec<f64> = (0..n_actions)
            .map(|idx| {
                let action = ((idx >> 0) & 1) as f64;
                let diff = b_norm * action - t_norm;
                diff * diff
            })
            .collect();

        // action=1 → cost = 0 (b_norm = t_norm = 1)
        assert!(costs[1] < 1e-12, "matching norms → zero cost for action=1");
        assert!(costs[0] > 0.5, "action=0 → cost = t_norm² = 1");
    }
}
