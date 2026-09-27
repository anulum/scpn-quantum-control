// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD pseudoinverse linalg replay helpers

//! Bounded pseudoinverse replay helpers for Program AD effect IR.
//!
//! Python Program AD emits one `linalg:pinv:<rows>x<cols>:<rcond>:<row>:<col>`
//! SSA node per scalar pseudoinverse output. This module owns the Rust-side
//! constant-rank replay contract for small matrices so the main Program AD
//! evaluator remains a dispatcher. It deliberately fails closed for matrices
//! outside the rank-1/`N x 2`/`2 x N` boundary, rank-threshold crossings, malformed
//! metadata, non-finite inputs, and Hermitian or dynamic cutoff policies because
//! those need a broader linalg policy before promotion.

use crate::program_ad_ir::{filled_replay_buffer, reserve_replay_buffer};
use crate::program_ad_lifecycle::replay_checkpoint;

#[derive(Debug, Clone, PartialEq)]
struct PinvMetadata {
    rows: usize,
    cols: usize,
    output_row: usize,
    output_col: usize,
    values: Vec<f64>,
    pinv: Vec<f64>,
}

/// Return whether an operation label belongs to bounded `np.linalg.pinv` replay.
pub(crate) fn is_pinv_operation(operation: &str) -> bool {
    operation.starts_with("linalg:pinv:")
}

/// Evaluate one scalar pseudoinverse output from a bounded Program AD node.
pub(crate) fn pinv_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let metadata = parse_pinv(effect_index, operation, input_values, false)?;
    Ok(metadata.pinv[metadata.output_row * metadata.rows + metadata.output_col])
}

/// Return local reverse contributions for one scalar pseudoinverse output node.
pub(crate) fn pinv_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} pinv cotangent must be finite"
        ));
    }
    let metadata = parse_pinv(effect_index, operation, input_values, true)?;
    let mut cotangent =
        filled_replay_buffer(matrix_entry_count(metadata.cols, metadata.rows)?, 0.0_f64)?;
    cotangent[metadata.output_row * metadata.rows + metadata.output_col] = output_cotangent;
    let adjoint = pinv_vjp(
        effect_index,
        metadata.rows,
        metadata.cols,
        &metadata.values,
        &metadata.pinv,
        &cotangent,
    )?;
    validate_finite_values(effect_index, "cotangent contribution", &adjoint)?;
    Ok(adjoint)
}

fn parse_pinv(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    requires_adjoint: bool,
) -> Result<PinvMetadata, String> {
    let (rows, cols, rcond, output_row, output_col) = validate_pinv_layout(
        effect_index,
        operation,
        input_values.len(),
        requires_adjoint,
    )?;
    validate_finite_values(effect_index, "inputs", input_values)?;
    let mut values = reserve_replay_buffer(input_values.len())?;
    for chunk in input_values.chunks(256) {
        replay_checkpoint()?;
        values.extend_from_slice(chunk);
    }
    replay_checkpoint()?;
    let pinv = pinv_bounded(effect_index, rows, cols, &values, rcond)?;
    validate_finite_values(effect_index, "output", &pinv)?;
    Ok(PinvMetadata {
        rows,
        cols,
        output_row,
        output_col,
        values,
        pinv,
    })
}

include!("program_ad_linalg_pinv/metadata.rs");
include!("program_ad_linalg_pinv/workspace.rs");

fn is_bounded_pinv_shape(rows: usize, cols: usize) -> bool {
    rows == 1 || cols == 1 || rows == 2 || cols == 2
}

fn pinv_bounded(
    effect_index: usize,
    rows: usize,
    cols: usize,
    matrix: &[f64],
    rcond: f64,
) -> Result<Vec<f64>, String> {
    if rows == 1 || cols == 1 {
        return pinv_rank1(effect_index, matrix, rcond);
    }
    pinv_rank2(effect_index, rows, cols, matrix, rcond)
}

fn pinv_rank1(effect_index: usize, matrix: &[f64], rcond: f64) -> Result<Vec<f64>, String> {
    let mut norm_squared = -0.0_f64;
    for (index, value) in matrix.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        norm_squared += value * value;
    }
    ensure_constant_rank1(effect_index, norm_squared, rcond)?;
    let mut pinv = reserve_replay_buffer(matrix.len())?;
    for (index, value) in matrix.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        pinv.push(value / norm_squared);
    }
    validate_finite_values(effect_index, "output", &pinv)?;
    Ok(pinv)
}

fn ensure_constant_rank1(effect_index: usize, norm_squared: f64, rcond: f64) -> Result<(), String> {
    if !norm_squared.is_finite() {
        return Err(format!(
            "effect {effect_index} pinv singular value must be finite"
        ));
    }
    if norm_squared <= 0.0 {
        return Err(format!(
            "effect {effect_index} pinv requires a constant full-rank matrix above cutoff"
        ));
    }
    let singular_value = norm_squared.sqrt();
    let scale = 1.0_f64.max(singular_value);
    if singular_value <= rcond * scale {
        return Err(format!(
            "effect {effect_index} pinv requires a constant full-rank matrix above cutoff"
        ));
    }
    Ok(())
}

fn pinv_rank2(
    effect_index: usize,
    rows: usize,
    cols: usize,
    matrix: &[f64],
    rcond: f64,
) -> Result<Vec<f64>, String> {
    if rows >= cols {
        let gram = gram_columns(rows, matrix)?;
        ensure_constant_rank2(effect_index, "column", gram, rcond)?;
        let inverse = invert_2x2(effect_index, gram, "column Gram matrix")?;
        let matrix_t = transpose(matrix, rows, cols)?;
        matmul(&inverse, 2, 2, &matrix_t, rows)
    } else {
        let gram = gram_rows(rows, cols, matrix)?;
        ensure_constant_rank2(effect_index, "row", gram, rcond)?;
        let inverse = invert_2x2(effect_index, gram, "row Gram matrix")?;
        let matrix_t = transpose(matrix, rows, cols)?;
        matmul(&matrix_t, cols, 2, &inverse, 2)
    }
}

fn ensure_constant_rank2(
    effect_index: usize,
    orientation: &str,
    gram: [f64; 4],
    rcond: f64,
) -> Result<(), String> {
    let [a, b, c, d] = gram;
    if (b - c).abs() > 1.0e-10 * 1.0_f64.max(a.abs()).max(b.abs()).max(c.abs()).max(d.abs()) {
        return Err(format!(
            "effect {effect_index} pinv {orientation} Gram matrix is not symmetric"
        ));
    }
    let off_diagonal = 0.5 * (b + c);
    let diagonal_delta = a - d;
    let gap = (diagonal_delta * diagonal_delta + 4.0 * off_diagonal * off_diagonal).sqrt();
    let center = 0.5 * (a + d);
    let eigenvalues = [center + 0.5 * gap, center - 0.5 * gap];
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} pinv Gram eigenvalues must be finite"
        ));
    }
    if eigenvalues[1] <= 0.0 {
        return Err(format!(
            "effect {effect_index} pinv requires a constant full-rank matrix above cutoff"
        ));
    }
    let singular_values = [eigenvalues[0].sqrt(), eigenvalues[1].sqrt()];
    let scale = 1.0_f64.max(singular_values[0]).max(singular_values[1]);
    if singular_values[1] <= rcond * scale {
        return Err(format!(
            "effect {effect_index} pinv requires a constant full-rank matrix above cutoff"
        ));
    }
    Ok(())
}

fn pinv_vjp(
    effect_index: usize,
    rows: usize,
    cols: usize,
    matrix: &[f64],
    pinv: &[f64],
    cotangent: &[f64],
) -> Result<Vec<f64>, String> {
    replay_checkpoint()?;
    matrix_entry_count(rows, cols)?;
    matrix_entry_count(rows, rows)?;
    matrix_entry_count(cols, cols)?;
    let left_projector = subtract(&identity(cols)?, &matmul(pinv, cols, rows, matrix, cols)?)?;
    let right_projector = subtract(&identity(rows)?, &matmul(matrix, rows, cols, pinv, rows)?)?;
    let pinv_t = transpose(pinv, cols, rows)?;
    let cotangent_t = transpose(cotangent, cols, rows)?;
    let left_projector_t = transpose(&left_projector, cols, cols)?;
    let right_projector_t = transpose(&right_projector, rows, rows)?;

    let term1_left = matmul(&pinv_t, rows, cols, cotangent, rows)?;
    let term1 = matmul(&term1_left, rows, rows, &pinv_t, cols)?;

    let term2_a = matmul(&right_projector_t, rows, rows, &cotangent_t, cols)?;
    let term2_b = matmul(&term2_a, rows, cols, pinv, rows)?;
    let term2 = matmul(&term2_b, rows, rows, &pinv_t, cols)?;

    let term3_a = matmul(&pinv_t, rows, cols, pinv, rows)?;
    let term3_b = matmul(&term3_a, rows, rows, &cotangent_t, cols)?;
    let term3 = matmul(&term3_b, rows, cols, &left_projector_t, cols)?;

    let mut adjoint = reserve_replay_buffer(matrix_entry_count(rows, cols)?)?;
    for index in 0..matrix.len() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        adjoint.push(-term1[index] + term2[index] + term3[index]);
    }
    validate_finite_values(effect_index, "adjoint matrix", &adjoint)?;
    Ok(adjoint)
}

fn gram_columns(rows: usize, matrix: &[f64]) -> Result<[f64; 4], String> {
    let mut gram = [0.0_f64; 4];
    for row in 0..rows {
        if row % 256 == 0 {
            replay_checkpoint()?;
        }
        let x = matrix[row * 2];
        let y = matrix[row * 2 + 1];
        gram[0] += x * x;
        gram[1] += x * y;
        gram[2] += y * x;
        gram[3] += y * y;
    }
    replay_checkpoint()?;
    Ok(gram)
}

fn gram_rows(rows: usize, cols: usize, matrix: &[f64]) -> Result<[f64; 4], String> {
    if rows != 2 {
        return Err("pinv row Gram matrix requires two rows".to_owned());
    }
    let mut gram = [0.0_f64; 4];
    for col in 0..cols {
        if col % 256 == 0 {
            replay_checkpoint()?;
        }
        let x = matrix[col];
        let y = matrix[cols + col];
        gram[0] += x * x;
        gram[1] += x * y;
        gram[2] += y * x;
        gram[3] += y * y;
    }
    replay_checkpoint()?;
    Ok(gram)
}

fn invert_2x2(effect_index: usize, matrix: [f64; 4], label: &str) -> Result<[f64; 4], String> {
    let [a, b, c, d] = matrix;
    let determinant = a * d - b * c;
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(format!("effect {effect_index} pinv {label} is singular"));
    }
    Ok([
        d / determinant,
        -b / determinant,
        -c / determinant,
        a / determinant,
    ])
}

include!("program_ad_linalg_pinv/matrix_ops.rs");
