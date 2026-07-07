// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD matrix-power linalg replay helpers

//! Bounded matrix-power replay helpers for Program AD effect IR.
//!
//! Python Program AD emits one
//! `linalg:matrix_power:<rows>x<cols>:power:<n>:<row>:<col>` SSA node per
//! scalar output of a statically exponentiated square matrix. This module owns
//! Rust-side contract parsing, value replay, and exact local VJP replay for
//! positive, zero, and negative integer exponents. Negative powers require a
//! nonsingular matrix and fail closed instead of silently widening the replay
//! claim.

#[derive(Debug, Clone, PartialEq)]
struct MatrixPowerMetadata {
    size: usize,
    exponent: i64,
    output_row: usize,
    output_col: usize,
}

/// Return whether an operation label belongs to bounded `np.linalg.matrix_power` replay.
pub(crate) fn is_matrix_power_operation(operation: &str) -> bool {
    operation.starts_with("linalg:matrix_power:")
}

/// Evaluate one scalar output from a compact static `matrix_power` Program AD node.
pub(crate) fn matrix_power_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let metadata = parse_matrix_power(effect_index, operation, input_values)?;
    let output = matrix_power(effect_index, input_values, metadata.size, metadata.exponent)?;
    Ok(output[metadata.output_row * metadata.size + metadata.output_col])
}

/// Return local reverse contributions for one scalar `matrix_power` output node.
pub(crate) fn matrix_power_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} matrix_power cotangent must be finite"
        ));
    }
    let metadata = parse_matrix_power(effect_index, operation, input_values)?;
    let mut cotangent = vec![0.0_f64; metadata.size * metadata.size];
    cotangent[metadata.output_row * metadata.size + metadata.output_col] = output_cotangent;
    let adjoint = matrix_power_vjp(
        effect_index,
        input_values,
        metadata.size,
        metadata.exponent,
        &cotangent,
    )?;
    if adjoint.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} matrix_power cotangent contribution must be finite"
        ));
    }
    Ok(adjoint)
}

fn parse_matrix_power(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<MatrixPowerMetadata, String> {
    let parts = operation.split(':').collect::<Vec<&str>>();
    if parts.len() != 7 || parts[0] != "linalg" || parts[1] != "matrix_power" || parts[3] != "power"
    {
        return Err(format!(
            "effect {effect_index} matrix_power operation metadata is malformed"
        ));
    }
    let size = parse_square_shape(effect_index, parts[2])?;
    let exponent = parts[4].parse::<i64>().map_err(|_| {
        format!("effect {effect_index} matrix_power exponent metadata is malformed")
    })?;
    let output_row = parts[5].parse::<usize>().map_err(|_| {
        format!("effect {effect_index} matrix_power output-row metadata is malformed")
    })?;
    let output_col = parts[6].parse::<usize>().map_err(|_| {
        format!("effect {effect_index} matrix_power output-column metadata is malformed")
    })?;
    if output_row >= size || output_col >= size {
        return Err(format!(
            "effect {effect_index} matrix_power output index is outside matrix shape"
        ));
    }
    let expected = size
        .checked_mul(size)
        .ok_or_else(|| format!("effect {effect_index} matrix_power shape size overflows"))?;
    if input_values.len() != expected {
        return Err(format!(
            "effect {effect_index} matrix_power requires {expected} flattened matrix operands"
        ));
    }
    if input_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} matrix_power inputs must be finite"
        ));
    }
    Ok(MatrixPowerMetadata {
        size,
        exponent,
        output_row,
        output_col,
    })
}

fn parse_square_shape(effect_index: usize, label: &str) -> Result<usize, String> {
    let parts = label.split('x').collect::<Vec<&str>>();
    if parts.len() != 2 {
        return Err(format!(
            "effect {effect_index} matrix_power shape metadata is malformed"
        ));
    }
    let rows = parts[0]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} matrix_power row metadata is malformed"))?;
    let cols = parts[1]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} matrix_power column metadata is malformed"))?;
    if rows == 0 || rows != cols {
        return Err(format!(
            "effect {effect_index} matrix_power requires non-empty square matrix metadata"
        ));
    }
    Ok(rows)
}

fn matrix_power(
    effect_index: usize,
    matrix: &[f64],
    size: usize,
    exponent: i64,
) -> Result<Vec<f64>, String> {
    if exponent >= 0 {
        return matrix_power_nonnegative(matrix, size, exponent_count(effect_index, exponent)?);
    }
    let count = exponent_magnitude(effect_index, exponent)?;
    let inverse = invert_square(effect_index, matrix, size)?;
    matrix_power_nonnegative(&inverse, size, count)
}

fn matrix_power_vjp(
    effect_index: usize,
    matrix: &[f64],
    size: usize,
    exponent: i64,
    cotangent: &[f64],
) -> Result<Vec<f64>, String> {
    if exponent == 0 {
        return Ok(vec![0.0; size * size]);
    }
    if exponent > 0 {
        let count = exponent_count(effect_index, exponent)?;
        let powers = prefix_powers(matrix, size, count)?;
        let mut total = vec![0.0_f64; size * size];
        for index in 0..count {
            let left = transpose(&powers[index], size);
            let right = transpose(&powers[count - 1 - index], size);
            total = add_matrices(
                &total,
                &matmul(&matmul(&left, cotangent, size)?, &right, size)?,
            )?;
        }
        return Ok(total);
    }
    let count = exponent_magnitude(effect_index, exponent)?;
    let inverse = invert_square(effect_index, matrix, size)?;
    let powers = prefix_powers(&inverse, size, count)?;
    let mut inverse_adjoint = vec![0.0_f64; size * size];
    for index in 0..count {
        let left = transpose(&powers[index], size);
        let right = transpose(&powers[count - 1 - index], size);
        inverse_adjoint = add_matrices(
            &inverse_adjoint,
            &matmul(&matmul(&left, cotangent, size)?, &right, size)?,
        )?;
    }
    let inverse_t = transpose(&inverse, size);
    let adjoint = matmul(
        &matmul(&inverse_t, &inverse_adjoint, size)?,
        &inverse_t,
        size,
    )?;
    Ok(adjoint.into_iter().map(|value| -value).collect())
}

fn exponent_count(effect_index: usize, exponent: i64) -> Result<usize, String> {
    usize::try_from(exponent).map_err(|_| {
        format!("effect {effect_index} matrix_power exponent magnitude is outside replay range")
    })
}

fn exponent_magnitude(effect_index: usize, exponent: i64) -> Result<usize, String> {
    exponent
        .checked_abs()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            format!("effect {effect_index} matrix_power exponent magnitude is outside replay range")
        })
}

fn matrix_power_nonnegative(matrix: &[f64], size: usize, count: usize) -> Result<Vec<f64>, String> {
    let mut total = identity(size);
    for _ in 0..count {
        total = matmul(&total, matrix, size)?;
    }
    Ok(total)
}

fn prefix_powers(matrix: &[f64], size: usize, count: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut powers = Vec::with_capacity(count);
    let mut current = identity(size);
    for _ in 0..count {
        powers.push(current.clone());
        current = matmul(&current, matrix, size)?;
    }
    Ok(powers)
}

fn identity(size: usize) -> Vec<f64> {
    let mut matrix = vec![0.0_f64; size * size];
    for index in 0..size {
        matrix[index * size + index] = 1.0;
    }
    matrix
}

fn transpose(matrix: &[f64], size: usize) -> Vec<f64> {
    let mut transposed = vec![0.0_f64; size * size];
    for row in 0..size {
        for col in 0..size {
            transposed[col * size + row] = matrix[row * size + col];
        }
    }
    transposed
}

fn add_matrices(left: &[f64], right: &[f64]) -> Result<Vec<f64>, String> {
    if left.len() != right.len() {
        return Err("matrix_power matrix sums require equal shapes".to_owned());
    }
    let result = left
        .iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<f64>>();
    if result.iter().any(|value| !value.is_finite()) {
        return Err("matrix_power matrix sum entries must be finite".to_owned());
    }
    Ok(result)
}

fn matmul(left: &[f64], right: &[f64], size: usize) -> Result<Vec<f64>, String> {
    if left.len() != size * size || right.len() != size * size {
        return Err("matrix_power matrix product operands must match shape".to_owned());
    }
    let mut output = vec![0.0_f64; size * size];
    for row in 0..size {
        for col in 0..size {
            let mut value = 0.0_f64;
            for inner in 0..size {
                value += left[row * size + inner] * right[inner * size + col];
            }
            if !value.is_finite() {
                return Err("matrix_power matrix product entry must be finite".to_owned());
            }
            output[row * size + col] = value;
        }
    }
    Ok(output)
}

fn invert_square(effect_index: usize, matrix: &[f64], size: usize) -> Result<Vec<f64>, String> {
    let width = size
        .checked_mul(2)
        .ok_or_else(|| format!("effect {effect_index} matrix_power inverse width overflows"))?;
    let mut augmented = vec![0.0_f64; size * width];
    for row in 0..size {
        for column in 0..size {
            augmented[row * width + column] = matrix[row * size + column];
        }
        augmented[row * width + size + row] = 1.0;
    }
    for column in 0..size {
        let mut pivot = column;
        let mut best = augmented[column * width + column].abs();
        for row in (column + 1)..size {
            let candidate = augmented[row * width + column].abs();
            if candidate > best {
                pivot = row;
                best = candidate;
            }
        }
        if best == 0.0 || !best.is_finite() {
            return Err(format!(
                "effect {effect_index} matrix_power requires a nonsingular matrix"
            ));
        }
        if pivot != column {
            for col in 0..width {
                augmented.swap(column * width + col, pivot * width + col);
            }
        }
        let pivot_value = augmented[column * width + column];
        for col in 0..width {
            augmented[column * width + col] /= pivot_value;
        }
        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = augmented[row * width + column];
            for col in 0..width {
                augmented[row * width + col] -= factor * augmented[column * width + col];
            }
        }
    }
    let mut inverse = vec![0.0_f64; size * size];
    for row in 0..size {
        for column in 0..size {
            let value = augmented[row * width + size + column];
            if !value.is_finite() {
                return Err(format!(
                    "effect {effect_index} matrix_power inverse entries must be finite"
                ));
            }
            inverse[row * size + column] = value;
        }
    }
    Ok(inverse)
}
