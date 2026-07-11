// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Native compiler AD numerical kernels

//! Dimension-generic kernels and shared validation for compiler-backed AD parity.

mod specialized_2x2;

pub use specialized_2x2::{
    matrix_2x2_determinant_gradient_inner, matrix_2x2_determinant_jvp_inner,
    matrix_2x2_determinant_value_inner, matrix_2x2_determinant_vjp_inner,
    matrix_2x2_eigensystem_jvp_inner, matrix_2x2_eigensystem_sum_gradient_inner,
    matrix_2x2_eigensystem_value_inner, matrix_2x2_eigensystem_vjp_inner,
    matrix_2x2_eigenvalues_jvp_inner, matrix_2x2_eigenvalues_sum_gradient_inner,
    matrix_2x2_eigenvalues_value_inner, matrix_2x2_eigenvalues_vjp_inner,
    matrix_2x2_inverse_jvp_inner, matrix_2x2_inverse_sum_gradient_inner,
    matrix_2x2_inverse_value_inner, matrix_2x2_inverse_vjp_inner, matrix_2x2_solve_jvp_inner,
    matrix_2x2_solve_sum_gradient_inner, matrix_2x2_solve_value_inner, matrix_2x2_solve_vjp_inner,
    symmetric_2x2_cholesky_jvp_inner, symmetric_2x2_cholesky_sum_gradient_inner,
    symmetric_2x2_cholesky_value_inner, symmetric_2x2_cholesky_vjp_inner,
    symmetric_2x2_eigenvalues_jvp_inner, symmetric_2x2_eigenvalues_sum_gradient_inner,
    symmetric_2x2_eigenvalues_value_inner, symmetric_2x2_eigenvalues_vjp_inner,
};

fn checked_vector<const N: usize>(
    vector: &[f64],
    label: &str,
    primitive: &str,
) -> Result<[f64; N], String> {
    if vector.len() != N {
        return Err(format!("{primitive} requires {N} {label} value(s)"));
    }
    let mut checked = [0.0; N];
    for (index, value) in vector.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label}[{index}] is not finite ({value})"));
        }
        checked[index] = *value;
    }
    Ok(checked)
}

fn checked_vector_dynamic(
    expected: usize,
    vector: &[f64],
    label: &str,
    primitive: &str,
) -> Result<Vec<f64>, String> {
    if vector.len() != expected {
        return Err(format!("{primitive} requires {expected} {label} value(s)"));
    }
    let mut checked = vec![0.0; expected];
    for (index, value) in vector.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label}[{index}] is not finite ({value})"));
        }
        checked[index] = *value;
    }
    Ok(checked)
}

fn checked_matrix_quadratic_form_values(
    dimension: usize,
    values: &[f64],
    primitive: &str,
) -> Result<(), String> {
    if dimension == 0 {
        return Err(format!("{primitive} dimension must be positive"));
    }
    let expected = dimension * dimension + dimension;
    if values.len() != expected {
        return Err(format!(
            "{primitive} requires dimension * dimension + dimension values"
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("values[{index}] is not finite ({value})"));
        }
    }
    Ok(())
}

fn checked_matrix_quadratic_form_vector<'a>(
    dimension: usize,
    values: &'a [f64],
    label: &str,
    primitive: &str,
) -> Result<&'a [f64], String> {
    checked_matrix_quadratic_form_values(dimension, values, primitive)?;
    if label != "values" {
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("{label}[{index}] is not finite ({value})"));
            }
        }
    }
    Ok(values)
}

fn matrix_quadratic_form_value_count(dimension: usize) -> usize {
    dimension * dimension + dimension
}

fn matrix_quadratic_form_matrix_index(dimension: usize, row: usize, column: usize) -> usize {
    row * dimension + column
}

fn matrix_quadratic_form_vector_index(dimension: usize, index: usize) -> usize {
    dimension * dimension + index
}

/// Evaluate x^T A x for row-major A followed by vector x.
pub fn matrix_quadratic_form_value_inner(
    dimension: usize,
    values: &[f64],
) -> Result<[f64; 1], String> {
    checked_matrix_quadratic_form_values(
        dimension,
        values,
        "native matrix quadratic form Rust value kernel",
    )?;
    let mut total = 0.0;
    for row in 0..dimension {
        let x_row = values[matrix_quadratic_form_vector_index(dimension, row)];
        for column in 0..dimension {
            let a = values[matrix_quadratic_form_matrix_index(dimension, row, column)];
            let x_column = values[matrix_quadratic_form_vector_index(dimension, column)];
            total += x_row * a * x_column;
        }
    }
    Ok([total])
}

/// Apply the exact JVP for x^T A x over row-major [A, x].
pub fn matrix_quadratic_form_jvp_inner(
    dimension: usize,
    values: &[f64],
    tangent: &[f64],
) -> Result<[f64; 1], String> {
    checked_matrix_quadratic_form_values(
        dimension,
        values,
        "native matrix quadratic form Rust JVP kernel",
    )?;
    checked_matrix_quadratic_form_vector(
        dimension,
        tangent,
        "tangent",
        "native matrix quadratic form Rust JVP kernel",
    )?;
    let gradient = matrix_quadratic_form_gradient_inner(dimension, values)?;
    let mut total = 0.0;
    for index in 0..matrix_quadratic_form_value_count(dimension) {
        total += gradient[index] * tangent[index];
    }
    Ok([total])
}

/// Apply the exact VJP for x^T A x over row-major [A, x].
pub fn matrix_quadratic_form_vjp_inner(
    dimension: usize,
    values: &[f64],
    cotangent: &[f64],
) -> Result<Vec<f64>, String> {
    checked_matrix_quadratic_form_values(
        dimension,
        values,
        "native matrix quadratic form Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<1>(
        cotangent,
        "cotangent",
        "native matrix quadratic form Rust VJP kernel",
    )?;
    let mut gradient = matrix_quadratic_form_gradient_inner(dimension, values)?;
    for value in &mut gradient {
        *value *= cotangent[0];
    }
    Ok(gradient)
}

/// Return the scalar-output gradient of x^T A x over row-major [A, x].
pub fn matrix_quadratic_form_gradient_inner(
    dimension: usize,
    values: &[f64],
) -> Result<Vec<f64>, String> {
    checked_matrix_quadratic_form_values(
        dimension,
        values,
        "native matrix quadratic form Rust gradient kernel",
    )?;
    let mut gradient = vec![0.0; matrix_quadratic_form_value_count(dimension)];
    for row in 0..dimension {
        let x_row = values[matrix_quadratic_form_vector_index(dimension, row)];
        for column in 0..dimension {
            let x_column = values[matrix_quadratic_form_vector_index(dimension, column)];
            let matrix_index = matrix_quadratic_form_matrix_index(dimension, row, column);
            gradient[matrix_index] = x_row * x_column;
        }
    }
    for row in 0..dimension {
        let mut vector_gradient = 0.0;
        for column in 0..dimension {
            let row_value = values[matrix_quadratic_form_matrix_index(dimension, row, column)];
            let column_value = values[matrix_quadratic_form_matrix_index(dimension, column, row)];
            let x_column = values[matrix_quadratic_form_vector_index(dimension, column)];
            vector_gradient += (row_value + column_value) * x_column;
        }
        gradient[matrix_quadratic_form_vector_index(dimension, row)] = vector_gradient;
    }
    Ok(gradient)
}

fn checked_matrix_square_values(
    dimension: usize,
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<(), String> {
    if dimension == 0 {
        return Err(format!("{primitive} dimension must be positive"));
    }
    let expected = dimension * dimension;
    if values.len() != expected {
        return Err(format!(
            "{primitive} requires dimension * dimension {label} value(s)"
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label}[{index}] is not finite ({value})"));
        }
    }
    Ok(())
}

fn matrix_square_index(dimension: usize, row: usize, column: usize) -> usize {
    row * dimension + column
}

fn matrix_vector_product_vector_index(dimension: usize, row: usize) -> usize {
    dimension * dimension + row
}

/// Evaluate trace(A) over row-major finite real square matrices.
pub fn matrix_trace_value_inner(dimension: usize, values: &[f64]) -> Result<[f64; 1], String> {
    checked_matrix_square_values(
        dimension,
        values,
        "values",
        "native matrix trace Rust value kernel",
    )?;
    let total = (0..dimension)
        .map(|index| values[matrix_square_index(dimension, index, index)])
        .sum();
    Ok([total])
}

/// Apply the exact JVP for trace(A).
pub fn matrix_trace_jvp_inner(
    dimension: usize,
    values: &[f64],
    tangent: &[f64],
) -> Result<[f64; 1], String> {
    checked_matrix_square_values(
        dimension,
        values,
        "values",
        "native matrix trace Rust JVP kernel",
    )?;
    checked_matrix_square_values(
        dimension,
        tangent,
        "tangent",
        "native matrix trace Rust JVP kernel",
    )?;
    let total = (0..dimension)
        .map(|index| tangent[matrix_square_index(dimension, index, index)])
        .sum();
    Ok([total])
}

/// Apply the exact VJP for trace(A).
pub fn matrix_trace_vjp_inner(
    dimension: usize,
    values: &[f64],
    cotangent: &[f64],
) -> Result<Vec<f64>, String> {
    checked_matrix_square_values(
        dimension,
        values,
        "values",
        "native matrix trace Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<1>(
        cotangent,
        "cotangent",
        "native matrix trace Rust VJP kernel",
    )?;
    let mut gradient = matrix_trace_gradient_inner(dimension, values)?;
    for value in &mut gradient {
        *value *= cotangent[0];
    }
    Ok(gradient)
}

/// Return the scalar-output identity-mask gradient for trace(A).
pub fn matrix_trace_gradient_inner(dimension: usize, values: &[f64]) -> Result<Vec<f64>, String> {
    checked_matrix_square_values(
        dimension,
        values,
        "values",
        "native matrix trace Rust gradient kernel",
    )?;
    let mut gradient = vec![0.0; dimension * dimension];
    for index in 0..dimension {
        gradient[matrix_square_index(dimension, index, index)] = 1.0;
    }
    Ok(gradient)
}

/// Evaluate sum_ij A_ij^2 over row-major finite real square matrices.
pub fn matrix_frobenius_norm_squared_value_inner(
    dimension: usize,
    values: &[f64],
) -> Result<[f64; 1], String> {
    checked_matrix_square_values(
        dimension,
        values,
        "values",
        "native matrix Frobenius-squared Rust value kernel",
    )?;
    Ok([values.iter().map(|value| value * value).sum()])
}

/// Apply the exact JVP for sum_ij A_ij^2.
pub fn matrix_frobenius_norm_squared_jvp_inner(
    dimension: usize,
    values: &[f64],
    tangent: &[f64],
) -> Result<[f64; 1], String> {
    checked_matrix_square_values(
        dimension,
        values,
        "values",
        "native matrix Frobenius-squared Rust JVP kernel",
    )?;
    checked_matrix_square_values(
        dimension,
        tangent,
        "tangent",
        "native matrix Frobenius-squared Rust JVP kernel",
    )?;
    let total = values
        .iter()
        .zip(tangent.iter())
        .map(|(value, tangent)| 2.0 * value * tangent)
        .sum();
    Ok([total])
}

/// Apply the exact VJP for sum_ij A_ij^2.
pub fn matrix_frobenius_norm_squared_vjp_inner(
    dimension: usize,
    values: &[f64],
    cotangent: &[f64],
) -> Result<Vec<f64>, String> {
    checked_matrix_square_values(
        dimension,
        values,
        "values",
        "native matrix Frobenius-squared Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<1>(
        cotangent,
        "cotangent",
        "native matrix Frobenius-squared Rust VJP kernel",
    )?;
    Ok(values
        .iter()
        .map(|value| 2.0 * value * cotangent[0])
        .collect())
}

/// Return the scalar-output gradient of sum_ij A_ij^2.
pub fn matrix_frobenius_norm_squared_gradient_inner(
    dimension: usize,
    values: &[f64],
) -> Result<Vec<f64>, String> {
    checked_matrix_square_values(
        dimension,
        values,
        "values",
        "native matrix Frobenius-squared Rust gradient kernel",
    )?;
    Ok(values.iter().map(|value| 2.0 * value).collect())
}

fn checked_matrix_vector_product_values(
    dimension: usize,
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<(), String> {
    if dimension == 0 {
        return Err(format!("{primitive} dimension must be positive"));
    }
    let expected = dimension * dimension + dimension;
    if values.len() != expected {
        return Err(format!(
            "{primitive} requires dimension * dimension + dimension {label} value(s)"
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label}[{index}] is not finite ({value})"));
        }
    }
    Ok(())
}

/// Evaluate A @ x over row-major finite real square matrices and finite vectors.
pub fn matrix_vector_product_value_inner(
    dimension: usize,
    values: &[f64],
) -> Result<Vec<f64>, String> {
    checked_matrix_vector_product_values(
        dimension,
        values,
        "values",
        "native matrix-vector product Rust value kernel",
    )?;
    let mut output = vec![0.0; dimension];
    for row in 0..dimension {
        let mut total = 0.0;
        for column in 0..dimension {
            total += values[matrix_square_index(dimension, row, column)]
                * values[matrix_vector_product_vector_index(dimension, column)];
        }
        output[row] = total;
    }
    Ok(output)
}

/// Apply the exact JVP for A @ x.
pub fn matrix_vector_product_jvp_inner(
    dimension: usize,
    values: &[f64],
    tangent: &[f64],
) -> Result<Vec<f64>, String> {
    checked_matrix_vector_product_values(
        dimension,
        values,
        "values",
        "native matrix-vector product Rust JVP kernel",
    )?;
    checked_matrix_vector_product_values(
        dimension,
        tangent,
        "tangent",
        "native matrix-vector product Rust JVP kernel",
    )?;
    let mut output = vec![0.0; dimension];
    for (row, output_row) in output.iter_mut().enumerate().take(dimension) {
        let mut total = 0.0;
        for column in 0..dimension {
            let matrix_index = matrix_square_index(dimension, row, column);
            let vector_index = matrix_vector_product_vector_index(dimension, column);
            total += tangent[matrix_index] * values[vector_index];
            total += values[matrix_index] * tangent[vector_index];
        }
        *output_row = total;
    }
    Ok(output)
}

/// Apply the exact VJP for A @ x.
pub fn matrix_vector_product_vjp_inner(
    dimension: usize,
    values: &[f64],
    cotangent: &[f64],
) -> Result<Vec<f64>, String> {
    checked_matrix_vector_product_values(
        dimension,
        values,
        "values",
        "native matrix-vector product Rust VJP kernel",
    )?;
    let cotangent = checked_vector_dynamic(
        dimension,
        cotangent,
        "cotangent",
        "native matrix-vector product Rust VJP kernel",
    )?;
    let mut gradient = vec![0.0; dimension * dimension + dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            gradient[matrix_square_index(dimension, row, column)] =
                cotangent[row] * values[matrix_vector_product_vector_index(dimension, column)];
        }
    }
    for column in 0..dimension {
        let mut vector_gradient = 0.0;
        for row in 0..dimension {
            vector_gradient += values[matrix_square_index(dimension, row, column)] * cotangent[row];
        }
        gradient[matrix_vector_product_vector_index(dimension, column)] = vector_gradient;
    }
    Ok(gradient)
}

/// Sum-output gradient provenance helper for the vector-output matrix-vector primitive.
pub fn matrix_vector_product_sum_gradient_inner(
    dimension: usize,
    values: &[f64],
) -> Result<Vec<f64>, String> {
    let cotangent = vec![1.0; dimension];
    matrix_vector_product_vjp_inner(dimension, values, &cotangent)
}

fn checked_matrix_matrix_product_values(
    dimension: usize,
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<(), String> {
    if dimension == 0 {
        return Err(format!("{primitive} dimension must be positive"));
    }
    let expected = 2 * dimension * dimension;
    if values.len() != expected {
        return Err(format!(
            "{primitive} requires 2 * dimension * dimension {label} value(s)"
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label}[{index}] is not finite ({value})"));
        }
    }
    Ok(())
}

fn matrix_matrix_product_right_index(dimension: usize, row: usize, column: usize) -> usize {
    dimension * dimension + matrix_square_index(dimension, row, column)
}

/// Evaluate A @ B over row-major finite real square matrix pairs.
pub fn matrix_matrix_product_value_inner(
    dimension: usize,
    values: &[f64],
) -> Result<Vec<f64>, String> {
    checked_matrix_matrix_product_values(
        dimension,
        values,
        "values",
        "native matrix-matrix product Rust value kernel",
    )?;
    let mut output = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            let mut total = 0.0;
            for inner in 0..dimension {
                total += values[matrix_square_index(dimension, row, inner)]
                    * values[matrix_matrix_product_right_index(dimension, inner, column)];
            }
            output[matrix_square_index(dimension, row, column)] = total;
        }
    }
    Ok(output)
}

/// Apply the exact JVP for A @ B.
pub fn matrix_matrix_product_jvp_inner(
    dimension: usize,
    values: &[f64],
    tangent: &[f64],
) -> Result<Vec<f64>, String> {
    checked_matrix_matrix_product_values(
        dimension,
        values,
        "values",
        "native matrix-matrix product Rust JVP kernel",
    )?;
    checked_matrix_matrix_product_values(
        dimension,
        tangent,
        "tangent",
        "native matrix-matrix product Rust JVP kernel",
    )?;
    let mut output = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            let mut total = 0.0;
            for inner in 0..dimension {
                let left_index = matrix_square_index(dimension, row, inner);
                let right_index = matrix_matrix_product_right_index(dimension, inner, column);
                total += tangent[left_index] * values[right_index];
                total += values[left_index] * tangent[right_index];
            }
            output[matrix_square_index(dimension, row, column)] = total;
        }
    }
    Ok(output)
}

/// Apply the exact VJP for A @ B.
pub fn matrix_matrix_product_vjp_inner(
    dimension: usize,
    values: &[f64],
    cotangent: &[f64],
) -> Result<Vec<f64>, String> {
    checked_matrix_matrix_product_values(
        dimension,
        values,
        "values",
        "native matrix-matrix product Rust VJP kernel",
    )?;
    checked_matrix_square_values(
        dimension,
        cotangent,
        "cotangent",
        "native matrix-matrix product Rust VJP kernel",
    )?;
    let mut gradient = vec![0.0; 2 * dimension * dimension];
    for row in 0..dimension {
        for inner in 0..dimension {
            let mut left_gradient = 0.0;
            for column in 0..dimension {
                left_gradient += cotangent[matrix_square_index(dimension, row, column)]
                    * values[matrix_matrix_product_right_index(dimension, inner, column)];
            }
            gradient[matrix_square_index(dimension, row, inner)] = left_gradient;
        }
    }
    for inner in 0..dimension {
        for column in 0..dimension {
            let mut right_gradient = 0.0;
            for row in 0..dimension {
                right_gradient += values[matrix_square_index(dimension, row, inner)]
                    * cotangent[matrix_square_index(dimension, row, column)];
            }
            gradient[matrix_matrix_product_right_index(dimension, inner, column)] = right_gradient;
        }
    }
    Ok(gradient)
}

/// Sum-output gradient provenance helper for the matrix-output matrix-matrix primitive.
pub fn matrix_matrix_product_sum_gradient_inner(
    dimension: usize,
    values: &[f64],
) -> Result<Vec<f64>, String> {
    let cotangent = vec![1.0; dimension * dimension];
    matrix_matrix_product_vjp_inner(dimension, values, &cotangent)
}

fn checked_vector_dot_values(
    dimension: usize,
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<(), String> {
    if dimension == 0 {
        return Err(format!("{primitive} dimension must be positive"));
    }
    let expected = 2 * dimension;
    if values.len() != expected {
        return Err(format!(
            "{primitive} requires 2 * dimension {label} value(s)"
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label}[{index}] is not finite ({value})"));
        }
    }
    Ok(())
}

/// Evaluate dot(x, y) over concatenated [x, y] finite real vectors.
pub fn vector_dot_value_inner(dimension: usize, values: &[f64]) -> Result<[f64; 1], String> {
    checked_vector_dot_values(
        dimension,
        values,
        "values",
        "native vector dot Rust value kernel",
    )?;
    let total = (0..dimension)
        .map(|index| values[index] * values[dimension + index])
        .sum();
    Ok([total])
}

/// Apply the exact JVP for dot(x, y) over concatenated [x, y].
pub fn vector_dot_jvp_inner(
    dimension: usize,
    values: &[f64],
    tangent: &[f64],
) -> Result<[f64; 1], String> {
    checked_vector_dot_values(
        dimension,
        values,
        "values",
        "native vector dot Rust JVP kernel",
    )?;
    checked_vector_dot_values(
        dimension,
        tangent,
        "tangent",
        "native vector dot Rust JVP kernel",
    )?;
    let total = (0..dimension)
        .map(|index| {
            values[dimension + index] * tangent[index] + values[index] * tangent[dimension + index]
        })
        .sum();
    Ok([total])
}

/// Apply the exact VJP for dot(x, y) over concatenated [x, y].
pub fn vector_dot_vjp_inner(
    dimension: usize,
    values: &[f64],
    cotangent: &[f64],
) -> Result<Vec<f64>, String> {
    checked_vector_dot_values(
        dimension,
        values,
        "values",
        "native vector dot Rust VJP kernel",
    )?;
    let cotangent =
        checked_vector::<1>(cotangent, "cotangent", "native vector dot Rust VJP kernel")?;
    let mut gradient = vector_dot_gradient_inner(dimension, values)?;
    for value in &mut gradient {
        *value *= cotangent[0];
    }
    Ok(gradient)
}

/// Return the scalar-output gradient [y, x] for dot(x, y).
pub fn vector_dot_gradient_inner(dimension: usize, values: &[f64]) -> Result<Vec<f64>, String> {
    checked_vector_dot_values(
        dimension,
        values,
        "values",
        "native vector dot Rust gradient kernel",
    )?;
    let mut gradient = vec![0.0; 2 * dimension];
    gradient[..dimension].copy_from_slice(&values[dimension..(dimension + dimension)]);
    gradient[dimension..(dimension + dimension)].copy_from_slice(&values[..dimension]);
    Ok(gradient)
}

fn checked_vector_squared_norm_values(
    dimension: usize,
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<(), String> {
    if dimension == 0 {
        return Err(format!("{primitive} dimension must be positive"));
    }
    if values.len() != dimension {
        return Err(format!("{primitive} requires dimension {label} value(s)"));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label}[{index}] is not finite ({value})"));
        }
    }
    Ok(())
}

/// Evaluate sum_i x_i^2 for finite real vectors.
pub fn vector_squared_norm_value_inner(
    dimension: usize,
    values: &[f64],
) -> Result<[f64; 1], String> {
    checked_vector_squared_norm_values(
        dimension,
        values,
        "values",
        "native vector squared norm Rust value kernel",
    )?;
    Ok([values.iter().map(|value| value * value).sum()])
}

/// Apply the exact JVP for sum_i x_i^2.
pub fn vector_squared_norm_jvp_inner(
    dimension: usize,
    values: &[f64],
    tangent: &[f64],
) -> Result<[f64; 1], String> {
    checked_vector_squared_norm_values(
        dimension,
        values,
        "values",
        "native vector squared norm Rust JVP kernel",
    )?;
    checked_vector_squared_norm_values(
        dimension,
        tangent,
        "tangent",
        "native vector squared norm Rust JVP kernel",
    )?;
    let total = values
        .iter()
        .zip(tangent.iter())
        .map(|(value, tangent)| 2.0 * value * tangent)
        .sum();
    Ok([total])
}

/// Apply the exact VJP for sum_i x_i^2.
pub fn vector_squared_norm_vjp_inner(
    dimension: usize,
    values: &[f64],
    cotangent: &[f64],
) -> Result<Vec<f64>, String> {
    checked_vector_squared_norm_values(
        dimension,
        values,
        "values",
        "native vector squared norm Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<1>(
        cotangent,
        "cotangent",
        "native vector squared norm Rust VJP kernel",
    )?;
    Ok(values
        .iter()
        .map(|value| 2.0 * value * cotangent[0])
        .collect())
}

/// Return the scalar-output gradient of sum_i x_i^2.
pub fn vector_squared_norm_gradient_inner(
    dimension: usize,
    values: &[f64],
) -> Result<Vec<f64>, String> {
    checked_vector_squared_norm_values(
        dimension,
        values,
        "values",
        "native vector squared norm Rust gradient kernel",
    )?;
    Ok(values.iter().map(|value| 2.0 * value).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (left, right)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*left - *right).abs() < 1e-12,
                "index {index}: left={left}, right={right}"
            );
        }
    }

    #[test]
    fn matrix_quadratic_form_value_jvp_vjp_and_gradient_match_closed_form() {
        let values = [2.0, -1.0, 0.5, 3.0, 1.5, -2.0];
        let tangent = [0.1, -0.2, 0.3, 0.4, -0.5, 0.25];
        let cotangent = [1.25];

        assert_close(
            &matrix_quadratic_form_value_inner(2, &values).unwrap(),
            &[18.0],
        );
        assert_close(
            &matrix_quadratic_form_jvp_inner(2, &values, &tangent).unwrap(),
            &[-5.1625],
        );
        assert_close(
            &matrix_quadratic_form_vjp_inner(2, &values, &cotangent).unwrap(),
            &[2.8125, -3.75, -3.75, 5.0, 8.75, -15.9375],
        );
        assert_close(
            &matrix_quadratic_form_gradient_inner(2, &values).unwrap(),
            &[2.25, -3.0, -3.0, 4.0, 7.0, -12.75],
        );
    }

    #[test]
    fn matrix_quadratic_form_boundaries_fail_closed() {
        let wrong_count = matrix_quadratic_form_value_inner(2, &[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("dimension * dimension + dimension"));
        let non_finite =
            matrix_quadratic_form_gradient_inner(2, &[2.0, -1.0, 0.5, 3.0, f64::NAN, -2.0])
                .unwrap_err();
        assert!(non_finite.contains("not finite"));
        let zero_dimension =
            matrix_quadratic_form_value_inner(0, &[2.0, -1.0, 0.5, 3.0, 1.5, -2.0]).unwrap_err();
        assert!(zero_dimension.contains("dimension must be positive"));
    }

    #[test]
    fn matrix_trace_value_jvp_vjp_and_gradient_match_closed_form() {
        let values = [2.0, -1.0, 0.5, 3.0];
        let tangent = [0.1, -0.2, 0.3, 0.4];
        let cotangent = [1.25];

        assert_close(&matrix_trace_value_inner(2, &values).unwrap(), &[5.0]);
        assert_close(
            &matrix_trace_jvp_inner(2, &values, &tangent).unwrap(),
            &[0.5],
        );
        assert_close(
            &matrix_trace_vjp_inner(2, &values, &cotangent).unwrap(),
            &[1.25, 0.0, 0.0, 1.25],
        );
        assert_close(
            &matrix_trace_gradient_inner(2, &values).unwrap(),
            &[1.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn matrix_trace_boundaries_fail_closed() {
        let wrong_count = matrix_trace_value_inner(2, &[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("dimension * dimension values"));
        let non_finite = matrix_trace_gradient_inner(2, &[2.0, f64::NAN, 0.5, 3.0]).unwrap_err();
        assert!(non_finite.contains("not finite"));
        let tangent_count = matrix_trace_jvp_inner(2, &[2.0, -1.0, 0.5, 3.0], &[1.0]).unwrap_err();
        assert!(tangent_count.contains("dimension * dimension tangent value"));
        let cotangent_count =
            matrix_trace_vjp_inner(2, &[2.0, -1.0, 0.5, 3.0], &[1.0, 2.0]).unwrap_err();
        assert!(cotangent_count.contains("requires 1 cotangent value"));
        let zero_dimension = matrix_trace_value_inner(0, &[1.0]).unwrap_err();
        assert!(zero_dimension.contains("dimension must be positive"));
    }

    #[test]
    fn matrix_frobenius_norm_squared_value_jvp_vjp_and_gradient_match_closed_form() {
        let values = [2.0, -1.0, 0.5, 3.0];
        let tangent = [0.1, -0.2, 0.3, 0.4];
        let cotangent = [1.25];

        assert_close(
            &matrix_frobenius_norm_squared_value_inner(2, &values).unwrap(),
            &[14.25],
        );
        assert_close(
            &matrix_frobenius_norm_squared_jvp_inner(2, &values, &tangent).unwrap(),
            &[3.5],
        );
        assert_close(
            &matrix_frobenius_norm_squared_vjp_inner(2, &values, &cotangent).unwrap(),
            &[5.0, -2.5, 1.25, 7.5],
        );
        assert_close(
            &matrix_frobenius_norm_squared_gradient_inner(2, &values).unwrap(),
            &[4.0, -2.0, 1.0, 6.0],
        );
    }

    #[test]
    fn matrix_frobenius_norm_squared_boundaries_fail_closed() {
        let wrong_count = matrix_frobenius_norm_squared_value_inner(2, &[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("dimension * dimension values"));
        let non_finite =
            matrix_frobenius_norm_squared_gradient_inner(2, &[2.0, f64::NAN, 0.5, 3.0])
                .unwrap_err();
        assert!(non_finite.contains("not finite"));
        let tangent_count =
            matrix_frobenius_norm_squared_jvp_inner(2, &[2.0, -1.0, 0.5, 3.0], &[1.0]).unwrap_err();
        assert!(tangent_count.contains("dimension * dimension tangent value"));
        let cotangent_count =
            matrix_frobenius_norm_squared_vjp_inner(2, &[2.0, -1.0, 0.5, 3.0], &[1.0, 2.0])
                .unwrap_err();
        assert!(cotangent_count.contains("requires 1 cotangent value"));
        let zero_dimension = matrix_frobenius_norm_squared_value_inner(0, &[1.0]).unwrap_err();
        assert!(zero_dimension.contains("dimension must be positive"));
    }

    #[test]
    fn matrix_vector_product_value_jvp_vjp_and_sum_gradient_match_closed_form() {
        let values = [2.0, -1.0, 0.5, 3.0, 1.5, -2.0];
        let tangent = [0.1, -0.2, 0.3, 0.4, -0.5, 0.25];
        let cotangent = [1.25, -0.5];

        assert_close(
            &matrix_vector_product_value_inner(2, &values).unwrap(),
            &[5.0, -5.25],
        );
        assert_close(
            &matrix_vector_product_jvp_inner(2, &values, &tangent).unwrap(),
            &[-0.7, 0.15],
        );
        assert_close(
            &matrix_vector_product_vjp_inner(2, &values, &cotangent).unwrap(),
            &[1.875, -2.5, -0.75, 1.0, 2.25, -2.75],
        );
        assert_close(
            &matrix_vector_product_sum_gradient_inner(2, &values).unwrap(),
            &[1.5, -2.0, 1.5, -2.0, 2.5, 2.0],
        );
    }

    #[test]
    fn matrix_vector_product_boundaries_fail_closed() {
        let wrong_count = matrix_vector_product_value_inner(2, &[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("dimension * dimension + dimension values"));
        let non_finite =
            matrix_vector_product_sum_gradient_inner(2, &[2.0, f64::NAN, 0.5, 3.0, 1.5, -2.0])
                .unwrap_err();
        assert!(non_finite.contains("not finite"));
        let tangent_count =
            matrix_vector_product_jvp_inner(2, &[2.0, -1.0, 0.5, 3.0, 1.5, -2.0], &[1.0])
                .unwrap_err();
        assert!(tangent_count.contains("dimension * dimension + dimension tangent value"));
        let cotangent_count =
            matrix_vector_product_vjp_inner(2, &[2.0, -1.0, 0.5, 3.0, 1.5, -2.0], &[1.0])
                .unwrap_err();
        assert!(cotangent_count.contains("requires 2 cotangent value"));
        let zero_dimension = matrix_vector_product_value_inner(0, &[1.0]).unwrap_err();
        assert!(zero_dimension.contains("dimension must be positive"));
    }

    #[test]
    fn matrix_matrix_product_value_jvp_vjp_and_sum_gradient_match_closed_form() {
        let values = [1.0, -2.0, 0.5, 3.0, 4.0, -1.0, 2.0, 0.25];
        let tangent = [0.2, -0.1, 0.3, 0.4, -0.5, 0.75, 0.25, -0.2];
        let cotangent = [1.25, -0.5, 0.75, 2.0];

        assert_close(
            &matrix_matrix_product_value_inner(2, &values).unwrap(),
            &[0.0, -1.5, 8.0, 0.25],
        );
        assert_close(
            &matrix_matrix_product_jvp_inner(2, &values, &tangent).unwrap(),
            &[-0.4, 0.925, 2.5, -0.425],
        );
        assert_close(
            &matrix_matrix_product_vjp_inner(2, &values, &cotangent).unwrap(),
            &[5.5, 2.375, 1.0, 2.0, 1.625, 0.5, -0.25, 7.0],
        );
        assert_close(
            &matrix_matrix_product_sum_gradient_inner(2, &values).unwrap(),
            &[3.0, 2.25, 3.0, 2.25, 1.5, 1.5, 1.0, 1.0],
        );
    }

    #[test]
    fn matrix_matrix_product_boundaries_fail_closed() {
        let wrong_count = matrix_matrix_product_value_inner(2, &[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("2 * dimension * dimension values"));
        let non_finite = matrix_matrix_product_sum_gradient_inner(
            2,
            &[1.0, f64::NAN, 0.5, 3.0, 4.0, -1.0, 2.0, 0.25],
        )
        .unwrap_err();
        assert!(non_finite.contains("not finite"));
        let tangent_count = matrix_matrix_product_jvp_inner(
            2,
            &[1.0, -2.0, 0.5, 3.0, 4.0, -1.0, 2.0, 0.25],
            &[1.0],
        )
        .unwrap_err();
        assert!(tangent_count.contains("2 * dimension * dimension tangent value"));
        let cotangent_count = matrix_matrix_product_vjp_inner(
            2,
            &[1.0, -2.0, 0.5, 3.0, 4.0, -1.0, 2.0, 0.25],
            &[1.0],
        )
        .unwrap_err();
        assert!(cotangent_count.contains("dimension * dimension cotangent value"));
        let zero_dimension = matrix_matrix_product_value_inner(0, &[1.0]).unwrap_err();
        assert!(zero_dimension.contains("dimension must be positive"));
    }

    #[test]
    fn vector_dot_value_jvp_vjp_and_gradient_match_closed_form() {
        let values = [1.0, 2.0, -3.0, 4.0];
        let tangent = [0.5, -1.0, 2.0, -0.25];
        let cotangent = [1.25];

        assert_close(&vector_dot_value_inner(2, &values).unwrap(), &[5.0]);
        assert_close(
            &vector_dot_jvp_inner(2, &values, &tangent).unwrap(),
            &[-4.0],
        );
        assert_close(
            &vector_dot_vjp_inner(2, &values, &cotangent).unwrap(),
            &[-3.75, 5.0, 1.25, 2.5],
        );
        assert_close(
            &vector_dot_gradient_inner(2, &values).unwrap(),
            &[-3.0, 4.0, 1.0, 2.0],
        );
    }

    #[test]
    fn vector_dot_boundaries_fail_closed() {
        let wrong_count = vector_dot_value_inner(2, &[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("2 * dimension values"));
        let non_finite = vector_dot_gradient_inner(2, &[1.0, f64::NAN, -3.0, 4.0]).unwrap_err();
        assert!(non_finite.contains("not finite"));
        let tangent_count = vector_dot_jvp_inner(2, &[1.0, 2.0, -3.0, 4.0], &[1.0]).unwrap_err();
        assert!(tangent_count.contains("2 * dimension tangent value"));
        let cotangent_count =
            vector_dot_vjp_inner(2, &[1.0, 2.0, -3.0, 4.0], &[1.0, 2.0]).unwrap_err();
        assert!(cotangent_count.contains("requires 1 cotangent value"));
        let zero_dimension = vector_dot_value_inner(0, &[1.0]).unwrap_err();
        assert!(zero_dimension.contains("dimension must be positive"));
    }

    #[test]
    fn vector_squared_norm_value_jvp_vjp_and_gradient_match_closed_form() {
        let values = [1.5, -2.0, 0.25];
        let tangent = [-0.5, 0.75, 2.0];
        let cotangent = [1.25];

        assert_close(
            &vector_squared_norm_value_inner(3, &values).unwrap(),
            &[6.3125],
        );
        assert_close(
            &vector_squared_norm_jvp_inner(3, &values, &tangent).unwrap(),
            &[-3.5],
        );
        assert_close(
            &vector_squared_norm_vjp_inner(3, &values, &cotangent).unwrap(),
            &[3.75, -5.0, 0.625],
        );
        assert_close(
            &vector_squared_norm_gradient_inner(3, &values).unwrap(),
            &[3.0, -4.0, 0.5],
        );
    }

    #[test]
    fn vector_squared_norm_boundaries_fail_closed() {
        let wrong_count = vector_squared_norm_value_inner(3, &[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("requires dimension values"));
        let non_finite = vector_squared_norm_gradient_inner(3, &[1.0, f64::NAN, 2.0]).unwrap_err();
        assert!(non_finite.contains("not finite"));
        let tangent_count = vector_squared_norm_jvp_inner(3, &[1.0, 2.0, 3.0], &[1.0]).unwrap_err();
        assert!(tangent_count.contains("requires dimension tangent value"));
        let cotangent_count =
            vector_squared_norm_vjp_inner(3, &[1.0, 2.0, 3.0], &[1.0, 2.0]).unwrap_err();
        assert!(cotangent_count.contains("requires 1 cotangent value"));
        let zero_dimension = vector_squared_norm_value_inner(0, &[1.0]).unwrap_err();
        assert!(zero_dimension.contains("dimension must be positive"));
    }
}
