// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD SVD linalg replay helpers

//! Bounded singular-value replay helpers for Program AD effect IR.
//!
//! Python Program AD emits one `linalg:svdvals:<rows>x<cols>:<index>` SSA node
//! per scalar output when `np.linalg.svd(..., compute_uv=False)` is traced on a
//! static rank-2 matrix. This module owns the Rust-side singular-value contract
//! so the main Program AD evaluator remains a dispatcher. It deliberately fails
//! closed for malformed metadata, non-finite inputs, rank-deficient matrices,
//! repeated singular values, singular-vector outputs, pseudoinverses, and
//! dynamic linalg metadata because those surfaces need separate promotion.

use nalgebra::DMatrix;

const DISTINCT_SINGULAR_VALUE_TOLERANCE: f64 = 1.0e-10;
const POSITIVE_SINGULAR_VALUE_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
struct SvdvalsMetadata {
    rows: usize,
    cols: usize,
    output_index: usize,
    singular_values: Vec<f64>,
    left_vectors: DMatrix<f64>,
    right_vectors_t: DMatrix<f64>,
}

/// Return whether an operation label belongs to bounded singular-value replay.
pub(crate) fn is_svdvals_operation(operation: &str) -> bool {
    operation.starts_with("linalg:svdvals:")
}

/// Evaluate one descending singular value from a row-major Program AD node.
pub(crate) fn svdvals_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let metadata = parse_svdvals(effect_index, operation, input_values)?;
    Ok(metadata.singular_values[metadata.output_index])
}

/// Return local reverse contributions for one scalar singular-value node.
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
    let metadata = parse_svdvals(effect_index, operation, input_values)?;
    let mut contributions = Vec::with_capacity(metadata.rows * metadata.cols);
    for row in 0..metadata.rows {
        for col in 0..metadata.cols {
            contributions.push(
                output_cotangent
                    * metadata.left_vectors[(row, metadata.output_index)]
                    * metadata.right_vectors_t[(metadata.output_index, col)],
            );
        }
    }
    if contributions.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} svdvals cotangent contribution must be finite"
        ));
    }
    Ok(contributions)
}

fn parse_svdvals(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<SvdvalsMetadata, String> {
    let (rows, cols, output_index) = parse_svdvals_metadata(effect_index, operation)?;
    let output_size = rows.min(cols);
    if output_index >= output_size {
        return Err(format!(
            "effect {effect_index} svdvals output index is outside the singular-value spectrum"
        ));
    }
    let expected_inputs = rows * cols;
    if input_values.len() != expected_inputs {
        return Err(format!(
            "effect {effect_index} svdvals requires {expected_inputs} flattened matrix operands"
        ));
    }
    if input_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} svdvals inputs must be finite"
        ));
    }

    let matrix = DMatrix::from_row_slice(rows, cols, input_values);
    let decomposition = matrix.svd(true, true);
    let singular_values = decomposition.singular_values.as_slice().to_vec();
    if singular_values.len() != output_size {
        return Err(format!(
            "effect {effect_index} svdvals decomposition returned an unexpected spectrum size"
        ));
    }
    if singular_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} svdvals output must be finite"
        ));
    }
    validate_singular_values(effect_index, &singular_values)?;
    let left_vectors = decomposition.u.ok_or_else(|| {
        format!("effect {effect_index} svdvals decomposition did not return left vectors")
    })?;
    let right_vectors_t = decomposition.v_t.ok_or_else(|| {
        format!("effect {effect_index} svdvals decomposition did not return right vectors")
    })?;
    if left_vectors.nrows() != rows
        || left_vectors.ncols() != output_size
        || right_vectors_t.nrows() != output_size
        || right_vectors_t.ncols() != cols
    {
        return Err(format!(
            "effect {effect_index} svdvals decomposition returned incompatible vector shapes"
        ));
    }
    if left_vectors.iter().any(|value| !value.is_finite())
        || right_vectors_t.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "effect {effect_index} svdvals singular vectors must be finite"
        ));
    }
    Ok(SvdvalsMetadata {
        rows,
        cols,
        output_index,
        singular_values,
        left_vectors,
        right_vectors_t,
    })
}

fn parse_svdvals_metadata(
    effect_index: usize,
    operation: &str,
) -> Result<(usize, usize, usize), String> {
    let parts = operation.split(':').collect::<Vec<&str>>();
    if parts.len() != 4 || parts[0] != "linalg" || parts[1] != "svdvals" {
        return Err(format!(
            "effect {effect_index} svdvals operation metadata is malformed"
        ));
    }
    let shape = parts[2].split('x').collect::<Vec<&str>>();
    if shape.len() != 2 {
        return Err(format!(
            "effect {effect_index} svdvals shape metadata is malformed"
        ));
    }
    let rows = shape[0]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} svdvals row metadata is malformed"))?;
    let cols = shape[1]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} svdvals column metadata is malformed"))?;
    if rows == 0 || cols == 0 {
        return Err(format!(
            "effect {effect_index} svdvals shape metadata must be positive"
        ));
    }
    let output_index = parts[3]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} svdvals output index metadata is malformed"))?;
    Ok((rows, cols, output_index))
}

fn validate_singular_values(effect_index: usize, singular_values: &[f64]) -> Result<(), String> {
    let scale = singular_values
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    if singular_values
        .iter()
        .any(|value| *value <= POSITIVE_SINGULAR_VALUE_TOLERANCE * scale)
    {
        return Err(format!(
            "effect {effect_index} svdvals gradient requires positive singular values"
        ));
    }
    for left in 0..singular_values.len() {
        for right in (left + 1)..singular_values.len() {
            if (singular_values[left] - singular_values[right]).abs()
                <= DISTINCT_SINGULAR_VALUE_TOLERANCE * scale
            {
                return Err(format!(
                    "effect {effect_index} svdvals gradient requires distinct singular values"
                ));
            }
        }
    }
    Ok(())
}
