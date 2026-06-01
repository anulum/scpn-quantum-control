// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Quantum Petri Superposition Kernels

//! Rust kernels for Quantum Petri superposition diagnostics.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::validation::{validate_contiguous_slice, validate_finite, validate_range};

#[pyfunction]
pub fn qpetri_transition_activity<'py>(
    py: Python<'py>,
    w_in_flat: PyReadonlyArray1<'_, f64>,
    marking: PyReadonlyArray1<'_, f64>,
    thresholds: PyReadonlyArray1<'_, f64>,
    n_transitions: usize,
    n_places: usize,
    sparsity_eps: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if n_transitions == 0 || n_places == 0 {
        return Err(PyValueError::new_err(format!(
            "n_transitions ({n_transitions}) and n_places ({n_places}) must be positive"
        )));
    }
    if !sparsity_eps.is_finite() || sparsity_eps < 0.0 {
        return Err(PyValueError::new_err(format!(
            "sparsity_eps must be finite and non-negative, got {sparsity_eps}"
        )));
    }

    let w_in = validate_contiguous_slice(&w_in_flat, "w_in_flat")?;
    let mark = validate_contiguous_slice(&marking, "marking")?;
    let thr = validate_contiguous_slice(&thresholds, "thresholds")?;

    validate_finite(w_in, "w_in_flat")?;
    validate_finite(mark, "marking")?;
    validate_finite(thr, "thresholds")?;

    if w_in.len() != n_transitions * n_places {
        return Err(PyValueError::new_err(format!(
            "w_in_flat length {} != n_transitions*n_places {}",
            w_in.len(),
            n_transitions * n_places
        )));
    }
    if mark.len() != n_places {
        return Err(PyValueError::new_err(format!(
            "marking length {} != n_places {n_places}",
            mark.len()
        )));
    }
    if thr.len() != n_transitions {
        return Err(PyValueError::new_err(format!(
            "thresholds length {} != n_transitions {n_transitions}",
            thr.len()
        )));
    }

    let clipped_marking: Vec<f64> = mark.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    let mut output = vec![0.0f64; n_transitions];

    for t in 0..n_transitions {
        validate_range(thr[t], 0.0, 1.0, "thresholds[t]")?;
        let row_start = t * n_places;
        let row_end = row_start + n_places;
        let row = &w_in[row_start..row_end];
        let mut incoming_sum = 0.0f64;
        let mut weighted_token = 0.0f64;
        for p in 0..n_places {
            let incoming = row[p].abs().clamp(0.0, 1.0);
            incoming_sum += incoming;
            weighted_token += incoming * clipped_marking[p];
        }
        if incoming_sum <= sparsity_eps {
            output[t] = 0.0;
            continue;
        }
        let activity = (weighted_token / incoming_sum) * thr[t];
        output[t] = activity.clamp(0.0, 1.0);
    }

    Ok(PyArray1::from_vec(py, output))
}

#[pyfunction]
pub fn qpetri_state_metrics(probabilities: PyReadonlyArray1<'_, f64>) -> PyResult<(f64, f64)> {
    let probs = validate_contiguous_slice(&probabilities, "probabilities")?;
    validate_finite(probs, "probabilities")?;

    let mut entropy = 0.0f64;
    let mut purity = 0.0f64;
    for &p in probs {
        validate_range(p, 0.0, 1.0, "probabilities[p]")?;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
        purity += p * p;
    }
    Ok((entropy, purity))
}

#[pyfunction]
pub fn qpetri_sample_marking<'py>(
    py: Python<'py>,
    probabilities: PyReadonlyArray1<'_, f64>,
    shots: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if shots == 0 {
        return Err(PyValueError::new_err("shots must be positive"));
    }
    let probs = validate_contiguous_slice(&probabilities, "probabilities")?;
    validate_finite(probs, "probabilities")?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut sampled = Vec::with_capacity(probs.len());

    for &p in probs {
        validate_range(p, 0.0, 1.0, "probabilities[p]")?;
        let mut success = 0usize;
        for _ in 0..shots {
            if rng.random::<f64>() < p {
                success += 1;
            }
        }
        sampled.push((success as f64) / (shots as f64));
    }

    Ok(PyArray1::from_vec(py, sampled))
}

#[pyfunction]
pub fn qpetri_campaign_aggregate<'py>(
    py: Python<'py>,
    output_markings_flat: PyReadonlyArray1<'_, f64>,
    transition_activity_flat: PyReadonlyArray1<'_, f64>,
    entropies: PyReadonlyArray1<'_, f64>,
    purities: PyReadonlyArray1<'_, f64>,
    n_steps: usize,
    n_places: usize,
    n_transitions: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    f64,
)> {
    if n_steps == 0 {
        return Err(PyValueError::new_err("n_steps must be positive"));
    }
    if n_places == 0 {
        return Err(PyValueError::new_err("n_places must be positive"));
    }
    if n_transitions == 0 {
        return Err(PyValueError::new_err("n_transitions must be positive"));
    }

    let outputs = validate_contiguous_slice(&output_markings_flat, "output_markings_flat")?;
    let activity =
        validate_contiguous_slice(&transition_activity_flat, "transition_activity_flat")?;
    let entropy = validate_contiguous_slice(&entropies, "entropies")?;
    let purity = validate_contiguous_slice(&purities, "purities")?;

    validate_finite(outputs, "output_markings_flat")?;
    validate_finite(activity, "transition_activity_flat")?;
    validate_finite(entropy, "entropies")?;
    validate_finite(purity, "purities")?;

    if outputs.len() != n_steps * n_places {
        return Err(PyValueError::new_err(format!(
            "output_markings_flat length {} != n_steps*n_places {}",
            outputs.len(),
            n_steps * n_places
        )));
    }
    if activity.len() != n_steps * n_transitions {
        return Err(PyValueError::new_err(format!(
            "transition_activity_flat length {} != n_steps*n_transitions {}",
            activity.len(),
            n_steps * n_transitions
        )));
    }
    if entropy.len() != n_steps {
        return Err(PyValueError::new_err(format!(
            "entropies length {} != n_steps {n_steps}",
            entropy.len()
        )));
    }
    if purity.len() != n_steps {
        return Err(PyValueError::new_err(format!(
            "purities length {} != n_steps {n_steps}",
            purity.len()
        )));
    }

    let mut mean_output = vec![0.0f64; n_places];
    for s in 0..n_steps {
        let row = &outputs[s * n_places..(s + 1) * n_places];
        for p in 0..n_places {
            mean_output[p] += row[p];
        }
    }
    for value in &mut mean_output {
        *value /= n_steps as f64;
    }

    let mut mean_activity = vec![0.0f64; n_transitions];
    for s in 0..n_steps {
        let row = &activity[s * n_transitions..(s + 1) * n_transitions];
        for t in 0..n_transitions {
            mean_activity[t] += row[t];
        }
    }
    for value in &mut mean_activity {
        *value /= n_steps as f64;
    }

    let mean_entropy = entropy.iter().sum::<f64>() / (n_steps as f64);
    let mean_purity = purity.iter().sum::<f64>() / (n_steps as f64);

    Ok((
        PyArray1::from_vec(py, mean_output),
        PyArray1::from_vec(py, mean_activity),
        mean_entropy,
        mean_purity,
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn state_metrics_pure_state() {
        let probs: [f64; 4] = [1.0, 0.0, 0.0, 0.0];
        let mut entropy = 0.0f64;
        let mut purity = 0.0f64;
        for &p in &probs {
            if p > 0.0 {
                entropy -= p * p.log2();
            }
            purity += p * p;
        }
        assert!(entropy.abs() < 1e-12);
        assert!((purity - 1.0).abs() < 1e-12);
    }
}
