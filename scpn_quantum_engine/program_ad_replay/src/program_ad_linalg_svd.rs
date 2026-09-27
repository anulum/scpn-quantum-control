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

use crate::program_ad_ir::reserve_replay_buffer;
use crate::program_ad_lifecycle::replay_checkpoint;
use nalgebra::{DMatrix, Dyn, VecStorage};

include!("program_ad_linalg_svd/workspace.rs");

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
    let mut contributions = reserve_replay_buffer(matrix_entry_count(
        effect_index,
        metadata.rows,
        metadata.cols,
    )?)?;
    for row in 0..metadata.rows {
        for col in 0..metadata.cols {
            if col % 256 == 0 {
                replay_checkpoint()?;
            }
            contributions.push(
                output_cotangent
                    * metadata.left_vectors[(row, metadata.output_index)]
                    * metadata.right_vectors_t[(metadata.output_index, col)],
            );
        }
    }
    validate_finite_buffer(effect_index, "cotangent contribution", &contributions)?;
    replay_checkpoint()?;
    Ok(contributions)
}

fn parse_svdvals(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<SvdvalsMetadata, String> {
    let (rows, cols, output_index) =
        validate_svdvals_layout(effect_index, operation, input_values.len())?;
    let output_size = rows.min(cols);
    let expected_inputs = matrix_entry_count(effect_index, rows, cols)?;
    validate_finite_buffer(effect_index, "inputs", input_values)?;
    let mut column_major = reserve_replay_buffer(expected_inputs)?;
    for col in 0..cols {
        for row in 0..rows {
            if row % 256 == 0 {
                replay_checkpoint()?;
            }
            column_major.push(input_values[row * cols + col]);
        }
    }
    replay_checkpoint()?;
    let matrix = DMatrix::from_data(VecStorage::new(Dyn(rows), Dyn(cols), column_major));
    let decomposition = matrix
        .try_svd_unordered(
            true,
            true,
            f64::EPSILON * 5.0,
            crate::program_ad_lifecycle::replay_solver_iterations(),
        )
        .ok_or_else(|| format!("effect {effect_index} SVD solver iteration limit exceeded"))?;
    replay_checkpoint()?;
    let mut singular_values: Vec<f64> = decomposition.singular_values.data.into();
    if singular_values.len() != output_size {
        return Err(format!(
            "effect {effect_index} svdvals decomposition returned an unexpected spectrum size"
        ));
    }
    validate_finite_buffer(effect_index, "output", &singular_values)?;
    validate_singular_values(effect_index, &singular_values)?;
    let mut left_vectors = decomposition.u.ok_or_else(|| {
        format!("effect {effect_index} svdvals decomposition did not return left vectors")
    })?;
    let mut right_vectors_t = decomposition.v_t.ok_or_else(|| {
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
    validate_finite_buffer(effect_index, "singular vectors", left_vectors.as_slice())?;
    validate_finite_buffer(effect_index, "singular vectors", right_vectors_t.as_slice())?;
    for index in 0..output_size {
        replay_checkpoint()?;
        let mut largest = index;
        for candidate in (index + 1)..output_size {
            if candidate % 256 == 0 {
                replay_checkpoint()?;
            }
            if singular_values[candidate] > singular_values[largest] {
                largest = candidate;
            }
        }
        if largest != index {
            singular_values.swap(index, largest);
            for row in 0..rows {
                if row % 256 == 0 {
                    replay_checkpoint()?;
                }
                left_vectors.swap((row, index), (row, largest));
            }
            for col in 0..cols {
                if col % 256 == 0 {
                    replay_checkpoint()?;
                }
                right_vectors_t.swap((index, col), (largest, col));
            }
        }
    }
    replay_checkpoint()?;
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
    replay_checkpoint()?;
    for _ in operation.as_bytes().chunks(256) {
        replay_checkpoint()?;
    }
    let mut fields = operation.split(':');
    let mut parts = [""; 4];
    for part in &mut parts {
        *part = fields.next().ok_or_else(|| {
            format!("effect {effect_index} svdvals operation metadata is malformed")
        })?;
    }
    if fields.next().is_some() || parts[0] != "linalg" || parts[1] != "svdvals" {
        return Err(format!(
            "effect {effect_index} svdvals operation metadata is malformed"
        ));
    }
    let mut dimensions = parts[2].split('x');
    let shape = [
        dimensions.next().unwrap_or(""),
        dimensions.next().unwrap_or(""),
    ];
    if dimensions.next().is_some() || shape.iter().any(|label| label.is_empty()) {
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
    let mut scale = 1.0_f64;
    for (index, value) in singular_values.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        scale = scale.max(value.abs());
    }
    for (index, value) in singular_values.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        if *value <= POSITIVE_SINGULAR_VALUE_TOLERANCE * scale {
            return Err(format!(
                "effect {effect_index} svdvals gradient requires positive singular values"
            ));
        }
    }
    for left in 0..singular_values.len() {
        replay_checkpoint()?;
        for right in (left + 1)..singular_values.len() {
            if right % 256 == 0 {
                replay_checkpoint()?;
            }
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

fn matrix_entry_count(effect_index: usize, rows: usize, cols: usize) -> Result<usize, String> {
    rows.checked_mul(cols)
        .filter(|entries| *entries <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| {
            format!("effect {effect_index} svdvals matrix bytes exceed native addressability")
        })
}

fn validate_finite_buffer(effect_index: usize, role: &str, values: &[f64]) -> Result<(), String> {
    for (index, value) in values.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        if !value.is_finite() {
            return Err(format!(
                "effect {effect_index} svdvals {role} must be finite"
            ));
        }
    }
    replay_checkpoint()?;
    Ok(())
}
