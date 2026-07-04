// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD SVD linalg replay helpers

//! Bounded singular-value replay helpers for Program AD effect IR.
//!
//! Python Program AD emits one `linalg:svdvals:2x2:<index>` SSA node per scalar
//! output when `np.linalg.svd(..., compute_uv=False)` is traced on a 2x2 matrix.
//! This module owns the Rust-side contract for that narrow singular-value
//! surface so the main Program AD evaluator remains a dispatcher. It deliberately
//! fails closed for non-2x2 matrices, malformed metadata, non-finite inputs,
//! rank-deficient matrices, and repeated singular values because general SVD,
//! singular-vector outputs, pseudoinverses, and dynamic linalg metadata need a
//! broader policy before promotion.

const DISTINCT_SINGULAR_VALUE_TOLERANCE: f64 = 1.0e-10;
const POSITIVE_SINGULAR_VALUE_TOLERANCE: f64 = 1.0e-12;
const EIGENVECTOR_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
struct Svdvals2x2 {
    output_index: usize,
    values: [f64; 4],
    singular_values: [f64; 2],
    right_vectors: [[f64; 2]; 2],
}

/// Return whether an operation label belongs to bounded singular-value replay.
pub(crate) fn is_svdvals_operation(operation: &str) -> bool {
    operation.starts_with("linalg:svdvals:")
}

/// Evaluate one descending singular value from a row-major 2x2 Program AD node.
pub(crate) fn svdvals_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let metadata = parse_svdvals_2x2(effect_index, operation, input_values)?;
    Ok(metadata.singular_values[metadata.output_index])
}

/// Return local reverse contributions for one scalar 2x2 singular-value node.
pub(crate) fn svdvals_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} svdvals cotangent must be finite"
        ));
    }
    let metadata = parse_svdvals_2x2(effect_index, operation, input_values)?;
    let [a, b, c, d] = metadata.values;
    let sigma = metadata.singular_values[metadata.output_index];
    let vector = metadata.right_vectors[metadata.output_index];
    let av = [a * vector[0] + b * vector[1], c * vector[0] + d * vector[1]];
    let left_vector = [av[0] / sigma, av[1] / sigma];
    let contributions = vec![
        output_cotangent * left_vector[0] * vector[0],
        output_cotangent * left_vector[0] * vector[1],
        output_cotangent * left_vector[1] * vector[0],
        output_cotangent * left_vector[1] * vector[1],
    ];
    if contributions.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} svdvals cotangent contribution must be finite"
        ));
    }
    Ok(contributions)
}

fn parse_svdvals_2x2(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<Svdvals2x2, String> {
    if input_values.len() != 4 {
        return Err(format!(
            "effect {effect_index} svdvals Rust replay supports only 2x2 matrices"
        ));
    }
    if input_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} svdvals inputs must be finite"
        ));
    }
    let output_index = parse_svdvals_index(effect_index, operation)?;
    if output_index >= 2 {
        return Err(format!(
            "effect {effect_index} svdvals 2x2 output index must be 0 or 1"
        ));
    }
    let [a, b, c, d] = input_values else {
        return Err(format!(
            "effect {effect_index} svdvals Rust replay supports only 2x2 matrices"
        ));
    };
    let values = [*a, *b, *c, *d];
    let gram00 = a * a + c * c;
    let gram01 = a * b + c * d;
    let gram11 = b * b + d * d;
    if [gram00, gram01, gram11]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(format!(
            "effect {effect_index} svdvals Gram matrix must be finite"
        ));
    }
    let diagonal_delta = gram00 - gram11;
    let gap = (diagonal_delta * diagonal_delta + 4.0 * gram01 * gram01).sqrt();
    let center = 0.5 * (gram00 + gram11);
    let lower_eigenvalue = center - 0.5 * gap;
    let upper_eigenvalue = center + 0.5 * gap;
    if [lower_eigenvalue, upper_eigenvalue, gap]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(format!(
            "effect {effect_index} svdvals Gram eigensystem must be finite"
        ));
    }
    let spectral_scale = 1.0_f64
        .max(lower_eigenvalue.abs())
        .max(upper_eigenvalue.abs());
    if lower_eigenvalue <= POSITIVE_SINGULAR_VALUE_TOLERANCE * spectral_scale {
        return Err(format!(
            "effect {effect_index} svdvals gradient requires positive singular values"
        ));
    }
    if gap <= DISTINCT_SINGULAR_VALUE_TOLERANCE * spectral_scale {
        return Err(format!(
            "effect {effect_index} svdvals gradient requires distinct singular values"
        ));
    }
    let singular_values = [upper_eigenvalue.sqrt(), lower_eigenvalue.sqrt()];
    if singular_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} svdvals output must be finite"
        ));
    }
    let right_vectors = [
        symmetric_2x2_unit_eigenvector(effect_index, gram00, gram01, gram11, upper_eigenvalue)?,
        symmetric_2x2_unit_eigenvector(effect_index, gram00, gram01, gram11, lower_eigenvalue)?,
    ];
    Ok(Svdvals2x2 {
        output_index,
        values,
        singular_values,
        right_vectors,
    })
}

fn parse_svdvals_index(effect_index: usize, operation: &str) -> Result<usize, String> {
    let parts = operation.split(':').collect::<Vec<&str>>();
    if parts.len() != 4 || parts[0] != "linalg" || parts[1] != "svdvals" || parts[2] != "2x2" {
        return Err(format!(
            "effect {effect_index} svdvals operation metadata is malformed"
        ));
    }
    parts[3]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} svdvals output index metadata is malformed"))
}

fn symmetric_2x2_unit_eigenvector(
    effect_index: usize,
    a: f64,
    b: f64,
    d: f64,
    eigenvalue: f64,
) -> Result<[f64; 2], String> {
    let scale = 1.0_f64.max(a.abs()).max(b.abs()).max(d.abs());
    let raw = if b.abs() <= EIGENVECTOR_TOLERANCE * scale {
        if a >= d {
            if eigenvalue >= 0.5 * (a + d) {
                [1.0, 0.0]
            } else {
                [0.0, 1.0]
            }
        } else if eigenvalue >= 0.5 * (a + d) {
            [0.0, 1.0]
        } else {
            [1.0, 0.0]
        }
    } else {
        let first = [b, eigenvalue - a];
        let second = [eigenvalue - d, b];
        if squared_norm(first) >= squared_norm(second) {
            first
        } else {
            second
        }
    };
    let norm = squared_norm(raw).sqrt();
    if !norm.is_finite() || norm <= EIGENVECTOR_TOLERANCE * scale {
        return Err(format!(
            "effect {effect_index} svdvals right singular vector is ill-conditioned"
        ));
    }
    Ok([raw[0] / norm, raw[1] / norm])
}

fn squared_norm(vector: [f64; 2]) -> f64 {
    vector[0] * vector[0] + vector[1] * vector[1]
}
