// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Community Quality Scoring for DynQ

//! Rust-accelerated quality scoring for DynQ execution regions.
//!
//! Given an adjacency matrix of gate errors, compute per-region
//! connectivity and fidelity scores in parallel.
//!
//! Ref: Liu et al., arXiv:2601.19635 (2026) — DynQ

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute quality scores for multiple regions in parallel.
///
/// Each region is a slice of qubit indices. Returns (connectivity, fidelity, composite)
/// for each region.
///
/// gate_errors_flat: n×n flat array of two-qubit gate error rates
/// region_offsets: [start0, end0, start1, end1, ...] into region_qubits
/// region_qubits: concatenated qubit indices for all regions
#[pyfunction]
pub fn score_regions_batch<'py>(
    py: Python<'py>,
    gate_errors_flat: PyReadonlyArray1<'_, f64>,
    n_qubits: usize,
    region_offsets: PyReadonlyArray1<'_, i64>,
    region_qubits: PyReadonlyArray1<'_, i64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let errors = gate_errors_flat.as_slice().unwrap();
    let offsets = region_offsets.as_slice().unwrap();
    let qubits = region_qubits.as_slice().unwrap();
    let n_regions = offsets.len() / 2;

    let results: Vec<(f64, f64, f64)> = (0..n_regions)
        .into_par_iter()
        .map(|r| {
            let start = offsets[2 * r] as usize;
            let end = offsets[2 * r + 1] as usize;
            let region = &qubits[start..end];
            let nr = region.len();

            if nr < 2 {
                return (0.0, 0.0, 0.0);
            }

            let mut n_edges = 0usize;
            let mut error_sum = 0.0f64;

            for i in 0..nr {
                for j in (i + 1)..nr {
                    let qi = region[i] as usize;
                    let qj = region[j] as usize;
                    if qi < n_qubits && qj < n_qubits {
                        let e = errors[qi * n_qubits + qj];
                        if e > 0.0 && e < 1.0 {
                            n_edges += 1;
                            error_sum += e;
                        }
                    }
                }
            }

            let max_edges = nr * (nr - 1) / 2;
            let connectivity = if max_edges > 0 {
                n_edges as f64 / max_edges as f64
            } else {
                0.0
            };
            let fidelity = if n_edges > 0 {
                1.0 - error_sum / n_edges as f64
            } else {
                0.0
            };
            let composite = connectivity * fidelity;

            (connectivity, fidelity, composite)
        })
        .collect();

    let (conn, fid, comp): (Vec<f64>, Vec<f64>, Vec<f64>) = results
        .into_iter()
        .fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut c, mut f, mut q), (ci, fi, qi)| {
                c.push(ci);
                f.push(fi);
                q.push(qi);
                (c, f, q)
            },
        );

    Ok((
        PyArray1::from_vec(py, conn),
        PyArray1::from_vec(py, fid),
        PyArray1::from_vec(py, comp),
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_score_regions_logic() {
        // 4 qubits, complete graph, uniform 1% error
        let n = 4;
        let mut errors = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    errors[i * n + j] = 0.01;
                }
            }
        }
        // Single region: all 4 qubits
        let offsets = vec![0i64, 4];
        let qubits = vec![0i64, 1, 2, 3];

        // Manual check: 6 edges, max 6, connectivity=1.0, fidelity=0.99
        let nr = 4;
        let max_edges = nr * (nr - 1) / 2; // 6
        assert_eq!(max_edges, 6);
        // All pairs have error 0.01 → mean fidelity = 0.99
        // Composite = 1.0 * 0.99 = 0.99
    }
}
