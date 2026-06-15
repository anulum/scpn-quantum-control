// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Sub-microsecond realtime telemetry kernel

//! Sub-microsecond outer-loop jitter and deadline telemetry.
//!
//! Computes inter-cycle jitter percentiles and deadline-miss counts from
//! integer-nanosecond cycle timestamps. Percentiles use the same
//! linear-interpolation rule (`method="linear"`, R-7) as
//! `numpy.quantile`, replicating NumPy's branchful `_lerp` for bit-true
//! parity with the Python reference in
//! `scpn_quantum_control.control.realtime_runtime`.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::validation::{validate_contiguous_slice, validate_finite};

/// NumPy-compatible linear-interpolation percentile over an ascending slice.
///
/// Replicates `numpy._lerp`: interpolate from the low value below the midpoint
/// and from the high value at or above it, so the floating-point result matches
/// `numpy.quantile(..., method="linear")` bit for bit.
fn linear_percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let virtual_index = q * (n as f64 - 1.0);
    let lo = virtual_index.floor();
    let lo_idx = lo as usize;
    if lo_idx + 1 >= n {
        return sorted[n - 1];
    }
    let gamma = virtual_index - lo;
    let a = sorted[lo_idx];
    let b = sorted[lo_idx + 1];
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

fn ascending_copy(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).expect("jitter values are finite"));
    v
}

fn percentile_quad(sorted: &[f64]) -> (f64, f64, f64, f64) {
    let pmax = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() - 1]
    };
    (
        linear_percentile(sorted, 0.50),
        linear_percentile(sorted, 0.95),
        linear_percentile(sorted, 0.99),
        pmax,
    )
}

/// Return `(p50, p95, p99, max)` of a jitter array in nanoseconds.
#[pyfunction]
pub fn sub_us_jitter_percentiles(
    jitters: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, f64, f64)> {
    let j = validate_contiguous_slice(&jitters, "jitters")?;
    validate_finite(j, "jitters")?;
    if j.is_empty() {
        return Ok((0.0, 0.0, 0.0, 0.0));
    }
    let sorted = ascending_copy(j);
    Ok(percentile_quad(&sorted))
}

/// Return `(p50, p95, p99, max, deadline_misses, count)` from cycle timestamps.
///
/// Jitter of cycle `i > 0` is `|(start[i] - start[i-1]) - target_period_ns|`;
/// cycle `0` has zero jitter. A cycle misses its deadline when
/// `end_ns > deadline_ns`.
#[pyfunction]
pub fn sub_us_tracker_summary(
    start_ns: PyReadonlyArray1<'_, i64>,
    end_ns: PyReadonlyArray1<'_, i64>,
    deadline_ns: PyReadonlyArray1<'_, i64>,
    target_period_ns: f64,
) -> PyResult<(f64, f64, f64, f64, i64, i64)> {
    if !target_period_ns.is_finite() || target_period_ns <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "target_period_ns must be finite and positive, got {target_period_ns}"
        )));
    }
    let start = validate_contiguous_slice(&start_ns, "start_ns")?;
    let end = validate_contiguous_slice(&end_ns, "end_ns")?;
    let deadline = validate_contiguous_slice(&deadline_ns, "deadline_ns")?;
    let n = start.len();
    if n == 0 {
        return Err(PyValueError::new_err("no cycles to summarise"));
    }
    if end.len() != n || deadline.len() != n {
        return Err(PyValueError::new_err(
            "start_ns, end_ns, deadline_ns must have equal length",
        ));
    }

    let mut jitters = Vec::with_capacity(n);
    jitters.push(0.0_f64);
    for i in 1..n {
        let interval = (start[i] - start[i - 1]) as f64;
        jitters.push((interval - target_period_ns).abs());
    }

    let mut misses: i64 = 0;
    for i in 0..n {
        if end[i] < start[i] {
            return Err(PyValueError::new_err(
                "every end_ns must be >= its start_ns",
            ));
        }
        if deadline[i] < start[i] {
            return Err(PyValueError::new_err(
                "every deadline_ns must be >= its start_ns",
            ));
        }
        if end[i] > deadline[i] {
            misses += 1;
        }
    }

    let sorted = ascending_copy(&jitters);
    let (p50, p95, p99, pmax) = percentile_quad(&sorted);
    Ok((p50, p95, p99, pmax, misses, n as i64))
}
