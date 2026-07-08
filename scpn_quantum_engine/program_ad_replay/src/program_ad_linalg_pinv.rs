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
    let metadata = parse_pinv(effect_index, operation, input_values)?;
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
    let metadata = parse_pinv(effect_index, operation, input_values)?;
    let mut cotangent = vec![0.0_f64; metadata.cols * metadata.rows];
    cotangent[metadata.output_row * metadata.rows + metadata.output_col] = output_cotangent;
    let adjoint = pinv_vjp(
        effect_index,
        metadata.rows,
        metadata.cols,
        &metadata.values,
        &metadata.pinv,
        &cotangent,
    )?;
    if adjoint.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} pinv cotangent contribution must be finite"
        ));
    }
    Ok(adjoint)
}

fn parse_pinv(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<PinvMetadata, String> {
    let (rows, cols, rcond, output_row, output_col) = parse_pinv_metadata(effect_index, operation)?;
    if !is_bounded_pinv_shape(rows, cols) {
        return Err(format!(
            "effect {effect_index} pinv Rust replay supports only rank-1, Nx2, and 2xN matrices"
        ));
    }
    if input_values.len() != rows * cols {
        return Err(format!(
            "effect {effect_index} pinv requires {} flattened matrix operands",
            rows * cols
        ));
    }
    if input_values.iter().any(|value| !value.is_finite()) {
        return Err(format!("effect {effect_index} pinv inputs must be finite"));
    }
    if output_row >= cols || output_col >= rows {
        return Err(format!(
            "effect {effect_index} pinv output index is outside pseudoinverse shape"
        ));
    }
    let values = input_values.to_vec();
    let pinv = pinv_bounded(effect_index, rows, cols, &values, rcond)?;
    Ok(PinvMetadata {
        rows,
        cols,
        output_row,
        output_col,
        values,
        pinv,
    })
}

fn parse_pinv_metadata(
    effect_index: usize,
    operation: &str,
) -> Result<(usize, usize, f64, usize, usize), String> {
    let parts = operation.split(':').collect::<Vec<&str>>();
    if parts.len() != 6 || parts[0] != "linalg" || parts[1] != "pinv" {
        return Err(format!(
            "effect {effect_index} pinv operation metadata is malformed"
        ));
    }
    let shape = parts[2].split('x').collect::<Vec<&str>>();
    if shape.len() != 2 {
        return Err(format!(
            "effect {effect_index} pinv shape metadata is malformed"
        ));
    }
    let rows = shape[0]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} pinv row metadata is malformed"))?;
    let cols = shape[1]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} pinv column metadata is malformed"))?;
    if rows == 0 || cols == 0 {
        return Err(format!(
            "effect {effect_index} pinv shape metadata must be positive"
        ));
    }
    let rcond = parts[3]
        .parse::<f64>()
        .map_err(|_| format!("effect {effect_index} pinv cutoff metadata is malformed"))?;
    if !rcond.is_finite() || rcond < 0.0 {
        return Err(format!(
            "effect {effect_index} pinv cutoff metadata must be finite and non-negative"
        ));
    }
    let output_row = parts[4]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} pinv output-row metadata is malformed"))?;
    let output_col = parts[5]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} pinv output-column metadata is malformed"))?;
    Ok((rows, cols, rcond, output_row, output_col))
}

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
    let norm_squared = matrix.iter().map(|value| value * value).sum::<f64>();
    ensure_constant_rank1(effect_index, norm_squared, rcond)?;
    Ok(matrix.iter().map(|value| value / norm_squared).collect())
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
        let gram = gram_columns(rows, matrix);
        ensure_constant_rank2(effect_index, "column", gram, rcond)?;
        let inverse = invert_2x2(effect_index, gram, "column Gram matrix")?;
        let matrix_t = transpose(matrix, rows, cols);
        Ok(matmul(&inverse, 2, 2, &matrix_t, rows))
    } else {
        let gram = gram_rows(rows, cols, matrix);
        ensure_constant_rank2(effect_index, "row", gram, rcond)?;
        let inverse = invert_2x2(effect_index, gram, "row Gram matrix")?;
        let matrix_t = transpose(matrix, rows, cols);
        Ok(matmul(&matrix_t, cols, 2, &inverse, 2))
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
    let left_projector = subtract(&identity(cols), &matmul(pinv, cols, rows, matrix, cols));
    let right_projector = subtract(&identity(rows), &matmul(matrix, rows, cols, pinv, rows));
    let pinv_t = transpose(pinv, cols, rows);
    let cotangent_t = transpose(cotangent, cols, rows);
    let left_projector_t = transpose(&left_projector, cols, cols);
    let right_projector_t = transpose(&right_projector, rows, rows);

    let term1_left = matmul(&pinv_t, rows, cols, cotangent, rows);
    let term1 = matmul(&term1_left, rows, rows, &pinv_t, cols);

    let term2_a = matmul(&right_projector_t, rows, rows, &cotangent_t, cols);
    let term2_b = matmul(&term2_a, rows, cols, pinv, rows);
    let term2 = matmul(&term2_b, rows, rows, &pinv_t, cols);

    let term3_a = matmul(&pinv_t, rows, cols, pinv, rows);
    let term3_b = matmul(&term3_a, rows, rows, &cotangent_t, cols);
    let term3 = matmul(&term3_b, rows, cols, &left_projector_t, cols);

    let adjoint = (0..rows * cols)
        .map(|index| -term1[index] + term2[index] + term3[index])
        .collect::<Vec<f64>>();
    if adjoint.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} pinv adjoint matrix must be finite"
        ));
    }
    Ok(adjoint)
}

fn gram_columns(rows: usize, matrix: &[f64]) -> [f64; 4] {
    let mut gram = [0.0_f64; 4];
    for row in 0..rows {
        let x = matrix[row * 2];
        let y = matrix[row * 2 + 1];
        gram[0] += x * x;
        gram[1] += x * y;
        gram[2] += y * x;
        gram[3] += y * y;
    }
    gram
}

fn gram_rows(rows: usize, cols: usize, matrix: &[f64]) -> [f64; 4] {
    debug_assert_eq!(rows, 2);
    let mut gram = [0.0_f64; 4];
    for col in 0..cols {
        let x = matrix[col];
        let y = matrix[cols + col];
        gram[0] += x * x;
        gram[1] += x * y;
        gram[2] += y * x;
        gram[3] += y * y;
    }
    gram
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

fn matmul(
    left: &[f64],
    left_rows: usize,
    left_cols: usize,
    right: &[f64],
    right_cols: usize,
) -> Vec<f64> {
    let mut result = vec![0.0_f64; left_rows * right_cols];
    for row in 0..left_rows {
        for col in 0..right_cols {
            result[row * right_cols + col] = (0..left_cols)
                .map(|inner| left[row * left_cols + inner] * right[inner * right_cols + col])
                .sum();
        }
    }
    result
}

fn transpose(matrix: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            result[col * rows + row] = matrix[row * cols + col];
        }
    }
    result
}

fn identity(size: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; size * size];
    for index in 0..size {
        result[index * size + index] = 1.0;
    }
    result
}

fn subtract(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| left_value - right_value)
        .collect()
}
