// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Native compiler AD kernels

//! Native Rust kernels for bounded compiler-backed AD primitive parity.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::validation::{validate_contiguous_slice, validate_finite};

const DISCRIMINANT_EPS: f64 = 1.0e-24;
const UPPER_CHART_EPS: f64 = 1.0e-12;
const MATRIX_2X2_DETERMINANT_EPS: f64 = 1.0e-12;

fn py_value_error(error: String) -> PyErr {
    PyValueError::new_err(error)
}

fn checked_matrix_2x2_values(values: &[f64]) -> Result<[f64; 4], String> {
    if values.len() != 4 {
        return Err(
            "native matrix 2x2 eigensystem Rust kernel requires row-major matrix values".into(),
        );
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("values[{index}] is not finite ({value})"));
        }
    }
    let checked = [values[0], values[1], values[2], values[3]];
    let delta = checked[0] - checked[3];
    let discriminant = delta * delta + 4.0 * checked[1] * checked[2];
    if !discriminant.is_finite() || discriminant <= DISCRIMINANT_EPS {
        return Err(
            "native matrix 2x2 eigensystem Rust kernel requires real distinct eigenvalues".into(),
        );
    }
    if checked[1].abs() <= UPPER_CHART_EPS {
        return Err(
            "native matrix 2x2 eigensystem Rust kernel requires a non-zero upper off-diagonal eigenvector chart"
                .into(),
        );
    }
    Ok(checked)
}

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

fn eigensystem_geometry(values: &[f64; 4]) -> ([f64; 2], [[f64; 2]; 2], [f64; 2], f64) {
    let [a, b, c, d] = *values;
    let trace = a + d;
    let delta = a - d;
    let discriminant = delta * delta + 4.0 * b * c;
    let root = discriminant.sqrt();
    let lower = 0.5 * (trace - root);
    let upper = 0.5 * (trace + root);
    let q_lower = 0.5 * (-delta - root);
    let q_upper = 0.5 * (-delta + root);
    let lower_norm = (b * b + q_lower * q_lower).sqrt();
    let upper_norm = (b * b + q_upper * q_upper).sqrt();
    (
        [lower, upper],
        [
            [b / lower_norm, q_lower / lower_norm],
            [b / upper_norm, q_upper / upper_norm],
        ],
        [q_lower, q_upper],
        root,
    )
}

/// Evaluate the bounded real-simple nonsymmetric 2x2 eigensystem chart.
pub fn matrix_2x2_eigensystem_value_inner(values: &[f64]) -> Result<[f64; 6], String> {
    let checked = checked_matrix_2x2_values(values)?;
    let (eigenvalues, eigenvectors, _, _) = eigensystem_geometry(&checked);
    Ok([
        eigenvalues[0],
        eigenvalues[1],
        eigenvectors[0][0],
        eigenvectors[1][0],
        eigenvectors[0][1],
        eigenvectors[1][1],
    ])
}

/// Apply the exact JVP for the bounded real-simple nonsymmetric 2x2 eigensystem chart.
pub fn matrix_2x2_eigensystem_jvp_inner(
    values: &[f64],
    tangent: &[f64],
) -> Result<[f64; 6], String> {
    let checked = checked_matrix_2x2_values(values)?;
    let tangent = checked_vector::<4>(
        tangent,
        "tangent",
        "native matrix 2x2 eigensystem Rust JVP kernel",
    )?;
    let [a, b, c, d] = checked;
    let [ta, tb, tc, td] = tangent;
    let trace_tangent = ta + td;
    let delta = a - d;
    let delta_tangent = ta - td;
    let discriminant = delta * delta + 4.0 * b * c;
    let root = discriminant.sqrt();
    let discriminant_tangent = 2.0 * delta * delta_tangent + 4.0 * (tb * c + b * tc);
    let root_tangent = discriminant_tangent / (2.0 * root);
    let lower_tangent = 0.5 * (trace_tangent - root_tangent);
    let upper_tangent = 0.5 * (trace_tangent + root_tangent);
    let (_, eigenvectors, q_values, _) = eigensystem_geometry(&checked);
    let q_lower_tangent = lower_tangent - ta;
    let q_upper_tangent = upper_tangent - ta;

    fn vector_tangent(b: f64, q: f64, vector: [f64; 2], tb: f64, tq: f64) -> [f64; 2] {
        let norm = (b * b + q * q).sqrt();
        let dot = vector[0] * tb + vector[1] * tq;
        [(tb - vector[0] * dot) / norm, (tq - vector[1] * dot) / norm]
    }

    let lower_vector_tangent = vector_tangent(b, q_values[0], eigenvectors[0], tb, q_lower_tangent);
    let upper_vector_tangent = vector_tangent(b, q_values[1], eigenvectors[1], tb, q_upper_tangent);
    Ok([
        lower_tangent,
        upper_tangent,
        lower_vector_tangent[0],
        upper_vector_tangent[0],
        lower_vector_tangent[1],
        upper_vector_tangent[1],
    ])
}

/// Apply the exact VJP for the bounded real-simple nonsymmetric 2x2 eigensystem chart.
pub fn matrix_2x2_eigensystem_vjp_inner(
    values: &[f64],
    cotangent: &[f64],
) -> Result<[f64; 4], String> {
    let checked = checked_matrix_2x2_values(values)?;
    let cotangent = checked_vector::<6>(
        cotangent,
        "cotangent",
        "native matrix 2x2 eigensystem Rust VJP kernel",
    )?;
    let [a, b, c, d] = checked;
    let delta = a - d;
    let (_, eigenvectors, q_values, root) = eigensystem_geometry(&checked);

    fn raw_vector_adjoint(b: f64, q: f64, vector: [f64; 2], cotangent: [f64; 2]) -> [f64; 2] {
        let norm = (b * b + q * q).sqrt();
        let dot = vector[0] * cotangent[0] + vector[1] * cotangent[1];
        [
            (cotangent[0] - vector[0] * dot) / norm,
            (cotangent[1] - vector[1] * dot) / norm,
        ]
    }

    let lower_raw_adjoint = raw_vector_adjoint(
        b,
        q_values[0],
        eigenvectors[0],
        [cotangent[2], cotangent[4]],
    );
    let upper_raw_adjoint = raw_vector_adjoint(
        b,
        q_values[1],
        eigenvectors[1],
        [cotangent[3], cotangent[5]],
    );
    let lower_eigenvalue_adjoint = cotangent[0] + lower_raw_adjoint[1];
    let upper_eigenvalue_adjoint = cotangent[1] + upper_raw_adjoint[1];
    let alpha = 0.5 * (lower_eigenvalue_adjoint + upper_eigenvalue_adjoint);
    let beta = (upper_eigenvalue_adjoint - lower_eigenvalue_adjoint) / (4.0 * root);
    let adj_a_eigenvalues = alpha + 2.0 * delta * beta;
    let adj_d_eigenvalues = alpha - 2.0 * delta * beta;
    let adj_b_eigenvalues = 4.0 * c * beta;
    let adj_c_eigenvalues = 4.0 * b * beta;
    let q_adjoint_sum = lower_raw_adjoint[1] + upper_raw_adjoint[1];
    let b_chart_adjoint = lower_raw_adjoint[0] + upper_raw_adjoint[0];
    Ok([
        adj_a_eigenvalues - q_adjoint_sum,
        adj_b_eigenvalues + b_chart_adjoint,
        adj_c_eigenvalues,
        adj_d_eigenvalues,
    ])
}

/// Sum-output gradient provenance helper for the vector-output eigensystem primitive.
pub fn matrix_2x2_eigensystem_sum_gradient_inner(values: &[f64]) -> Result<[f64; 4], String> {
    matrix_2x2_eigensystem_vjp_inner(values, &[1.0; 6])
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

fn checked_matrix_2x2_determinant_values(
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<[f64; 4], String> {
    if values.len() != 4 {
        return Err(format!(
            "{primitive} requires row-major 2x2 matrix {label} values"
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label}[{index}] is not finite ({value})"));
        }
    }
    Ok([values[0], values[1], values[2], values[3]])
}

/// Evaluate det(A) for a row-major finite real 2x2 matrix.
pub fn matrix_2x2_determinant_value_inner(values: &[f64]) -> Result<[f64; 1], String> {
    let [a00, a01, a10, a11] = checked_matrix_2x2_determinant_values(
        values,
        "values",
        "native matrix 2x2 determinant Rust value kernel",
    )?;
    Ok([a00 * a11 - a01 * a10])
}

/// Apply the exact JVP for det(A) over row-major 2x2 matrices.
pub fn matrix_2x2_determinant_jvp_inner(
    values: &[f64],
    tangent: &[f64],
) -> Result<[f64; 1], String> {
    let [a00, a01, a10, a11] = checked_matrix_2x2_determinant_values(
        values,
        "values",
        "native matrix 2x2 determinant Rust JVP kernel",
    )?;
    let [t00, t01, t10, t11] = checked_matrix_2x2_determinant_values(
        tangent,
        "tangent",
        "native matrix 2x2 determinant Rust JVP kernel",
    )?;
    Ok([t00 * a11 + a00 * t11 - t01 * a10 - a01 * t10])
}

/// Apply the exact VJP for det(A) over row-major 2x2 matrices.
pub fn matrix_2x2_determinant_vjp_inner(
    values: &[f64],
    cotangent: &[f64],
) -> Result<[f64; 4], String> {
    let [a00, a01, a10, a11] = checked_matrix_2x2_determinant_values(
        values,
        "values",
        "native matrix 2x2 determinant Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<1>(
        cotangent,
        "cotangent",
        "native matrix 2x2 determinant Rust VJP kernel",
    )?;
    Ok([
        cotangent[0] * a11,
        -cotangent[0] * a10,
        -cotangent[0] * a01,
        cotangent[0] * a00,
    ])
}

/// Return the scalar-output adjugate gradient for det(A).
pub fn matrix_2x2_determinant_gradient_inner(values: &[f64]) -> Result<[f64; 4], String> {
    matrix_2x2_determinant_vjp_inner(values, &[1.0])
}

fn checked_matrix_2x2_inverse_values(
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<([f64; 4], f64), String> {
    let checked = checked_matrix_2x2_determinant_values(values, label, primitive)?;
    let determinant = checked[0] * checked[3] - checked[1] * checked[2];
    if !determinant.is_finite() || determinant.abs() <= MATRIX_2X2_DETERMINANT_EPS {
        return Err(format!(
            "{primitive} requires a nonsingular row-major 2x2 matrix"
        ));
    }
    Ok((checked, determinant))
}

fn inverse_2x2_from_checked(values: &[f64; 4], determinant: f64) -> [f64; 4] {
    [
        values[3] / determinant,
        -values[1] / determinant,
        -values[2] / determinant,
        values[0] / determinant,
    ]
}

fn matmul_2x2(left: &[f64; 4], right: &[f64; 4]) -> [f64; 4] {
    [
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
    ]
}

fn transpose_2x2(values: &[f64; 4]) -> [f64; 4] {
    [values[0], values[2], values[1], values[3]]
}

/// Evaluate inv(A) for a nonsingular row-major finite real 2x2 matrix.
pub fn matrix_2x2_inverse_value_inner(values: &[f64]) -> Result<[f64; 4], String> {
    let (checked, determinant) = checked_matrix_2x2_inverse_values(
        values,
        "values",
        "native matrix 2x2 inverse Rust value kernel",
    )?;
    Ok(inverse_2x2_from_checked(&checked, determinant))
}

/// Apply the exact JVP -A^-1 dA A^-1 for row-major 2x2 inverse.
pub fn matrix_2x2_inverse_jvp_inner(values: &[f64], tangent: &[f64]) -> Result<[f64; 4], String> {
    let (checked, determinant) = checked_matrix_2x2_inverse_values(
        values,
        "values",
        "native matrix 2x2 inverse Rust JVP kernel",
    )?;
    let tangent = checked_matrix_2x2_determinant_values(
        tangent,
        "tangent",
        "native matrix 2x2 inverse Rust JVP kernel",
    )?;
    let inverse = inverse_2x2_from_checked(&checked, determinant);
    let product = matmul_2x2(&matmul_2x2(&inverse, &tangent), &inverse);
    Ok([-product[0], -product[1], -product[2], -product[3]])
}

/// Apply the exact VJP -A^-T C A^-T for row-major 2x2 inverse.
pub fn matrix_2x2_inverse_vjp_inner(values: &[f64], cotangent: &[f64]) -> Result<[f64; 4], String> {
    let (checked, determinant) = checked_matrix_2x2_inverse_values(
        values,
        "values",
        "native matrix 2x2 inverse Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<4>(
        cotangent,
        "cotangent",
        "native matrix 2x2 inverse Rust VJP kernel",
    )?;
    let inverse = inverse_2x2_from_checked(&checked, determinant);
    let inverse_transpose = transpose_2x2(&inverse);
    let product = matmul_2x2(
        &matmul_2x2(&inverse_transpose, &cotangent),
        &inverse_transpose,
    );
    Ok([-product[0], -product[1], -product[2], -product[3]])
}

/// Return the sum-output gradient provenance for the vector-output 2x2 inverse.
pub fn matrix_2x2_inverse_sum_gradient_inner(values: &[f64]) -> Result<[f64; 4], String> {
    matrix_2x2_inverse_vjp_inner(values, &[1.0; 4])
}

fn checked_matrix_2x2_solve_values(
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<([f64; 4], [f64; 2], f64), String> {
    if values.len() != 6 {
        return Err(format!(
            "{primitive} requires row-major 2x2 matrix plus rhs {label} values"
        ));
    }
    let (matrix, determinant) = checked_matrix_2x2_inverse_values(&values[..4], label, primitive)?;
    for (offset, value) in values[4..6].iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{primitive} {label} value {} is not finite",
                offset + 4
            ));
        }
    }
    Ok((matrix, [values[4], values[5]], determinant))
}

fn checked_matrix_2x2_solve_tangent_values(
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<([f64; 4], [f64; 2]), String> {
    if values.len() != 6 {
        return Err(format!(
            "{primitive} requires row-major 2x2 matrix plus rhs {label} values"
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{primitive} {label} value {index} is not finite"));
        }
    }
    Ok((
        [values[0], values[1], values[2], values[3]],
        [values[4], values[5]],
    ))
}

fn solve_2x2_from_checked(matrix: &[f64; 4], determinant: f64, rhs: &[f64; 2]) -> [f64; 2] {
    [
        (matrix[3] * rhs[0] - matrix[1] * rhs[1]) / determinant,
        (-matrix[2] * rhs[0] + matrix[0] * rhs[1]) / determinant,
    ]
}

/// Evaluate A^-1 b for a nonsingular row-major finite real 2x2 system.
pub fn matrix_2x2_solve_value_inner(values: &[f64]) -> Result<[f64; 2], String> {
    let (matrix, rhs, determinant) = checked_matrix_2x2_solve_values(
        values,
        "values",
        "native matrix 2x2 solve Rust value kernel",
    )?;
    Ok(solve_2x2_from_checked(&matrix, determinant, &rhs))
}

/// Apply the exact JVP A^-1 (db - dA x) for row-major 2x2 linear solves.
pub fn matrix_2x2_solve_jvp_inner(values: &[f64], tangent: &[f64]) -> Result<[f64; 2], String> {
    let (matrix, rhs, determinant) = checked_matrix_2x2_solve_values(
        values,
        "values",
        "native matrix 2x2 solve Rust JVP kernel",
    )?;
    let (tangent_matrix, tangent_rhs) = checked_matrix_2x2_solve_tangent_values(
        tangent,
        "tangent",
        "native matrix 2x2 solve Rust JVP kernel",
    )?;
    let primal = solve_2x2_from_checked(&matrix, determinant, &rhs);
    let residual = [
        tangent_rhs[0] - tangent_matrix[0] * primal[0] - tangent_matrix[1] * primal[1],
        tangent_rhs[1] - tangent_matrix[2] * primal[0] - tangent_matrix[3] * primal[1],
    ];
    Ok(solve_2x2_from_checked(&matrix, determinant, &residual))
}

/// Apply the exact VJP for row-major 2x2 solves: dA=-lambda x^T, db=lambda.
pub fn matrix_2x2_solve_vjp_inner(values: &[f64], cotangent: &[f64]) -> Result<[f64; 6], String> {
    let (matrix, rhs, determinant) = checked_matrix_2x2_solve_values(
        values,
        "values",
        "native matrix 2x2 solve Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<2>(
        cotangent,
        "cotangent",
        "native matrix 2x2 solve Rust VJP kernel",
    )?;
    let primal = solve_2x2_from_checked(&matrix, determinant, &rhs);
    let inverse = inverse_2x2_from_checked(&matrix, determinant);
    let inverse_transpose = transpose_2x2(&inverse);
    let adjoint = [
        inverse_transpose[0] * cotangent[0] + inverse_transpose[1] * cotangent[1],
        inverse_transpose[2] * cotangent[0] + inverse_transpose[3] * cotangent[1],
    ];
    Ok([
        -adjoint[0] * primal[0],
        -adjoint[0] * primal[1],
        -adjoint[1] * primal[0],
        -adjoint[1] * primal[1],
        adjoint[0],
        adjoint[1],
    ])
}

/// Return the sum-output gradient provenance for the vector-output 2x2 solve.
pub fn matrix_2x2_solve_sum_gradient_inner(values: &[f64]) -> Result<[f64; 6], String> {
    matrix_2x2_solve_vjp_inner(values, &[1.0; 2])
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
    for index in 0..dimension {
        gradient[index] = values[dimension + index];
        gradient[dimension + index] = values[index];
    }
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

/// PyO3 wrapper for bounded Rust eigensystem value evaluation.
#[pyfunction]
pub fn matrix_2x2_eigensystem_value<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_2x2_eigensystem_value_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust eigensystem JVP evaluation.
#[pyfunction]
pub fn matrix_2x2_eigensystem_jvp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let tangent = validate_contiguous_slice(&tangent, "tangent")?;
    validate_finite(values, "values")?;
    validate_finite(tangent, "tangent")?;
    let result = matrix_2x2_eigensystem_jvp_inner(values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust eigensystem VJP evaluation.
#[pyfunction]
pub fn matrix_2x2_eigensystem_vjp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let cotangent = validate_contiguous_slice(&cotangent, "cotangent")?;
    validate_finite(values, "values")?;
    validate_finite(cotangent, "cotangent")?;
    let result = matrix_2x2_eigensystem_vjp_inner(values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust sum-output gradient provenance.
#[pyfunction]
pub fn matrix_2x2_eigensystem_sum_gradient<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_2x2_eigensystem_sum_gradient_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust matrix quadratic-form value evaluation.
#[pyfunction]
pub fn matrix_quadratic_form_value<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_quadratic_form_value_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust matrix quadratic-form JVP evaluation.
#[pyfunction]
pub fn matrix_quadratic_form_jvp<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let tangent = validate_contiguous_slice(&tangent, "tangent")?;
    validate_finite(values, "values")?;
    validate_finite(tangent, "tangent")?;
    let result =
        matrix_quadratic_form_jvp_inner(dimension, values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust matrix quadratic-form VJP evaluation.
#[pyfunction]
pub fn matrix_quadratic_form_vjp<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let cotangent = validate_contiguous_slice(&cotangent, "cotangent")?;
    validate_finite(values, "values")?;
    validate_finite(cotangent, "cotangent")?;
    let result =
        matrix_quadratic_form_vjp_inner(dimension, values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust matrix quadratic-form gradient evaluation.
#[pyfunction]
pub fn matrix_quadratic_form_gradient<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_quadratic_form_gradient_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust matrix trace value evaluation.
#[pyfunction]
pub fn matrix_trace_value<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_trace_value_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust matrix trace JVP evaluation.
#[pyfunction]
pub fn matrix_trace_jvp<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let tangent = validate_contiguous_slice(&tangent, "tangent")?;
    validate_finite(values, "values")?;
    validate_finite(tangent, "tangent")?;
    let result = matrix_trace_jvp_inner(dimension, values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust matrix trace VJP evaluation.
#[pyfunction]
pub fn matrix_trace_vjp<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let cotangent = validate_contiguous_slice(&cotangent, "cotangent")?;
    validate_finite(values, "values")?;
    validate_finite(cotangent, "cotangent")?;
    let result = matrix_trace_vjp_inner(dimension, values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust matrix trace gradient evaluation.
#[pyfunction]
pub fn matrix_trace_gradient<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_trace_gradient_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust matrix Frobenius-squared value evaluation.
#[pyfunction]
pub fn matrix_frobenius_norm_squared_value<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result =
        matrix_frobenius_norm_squared_value_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust matrix Frobenius-squared JVP evaluation.
#[pyfunction]
pub fn matrix_frobenius_norm_squared_jvp<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let tangent = validate_contiguous_slice(&tangent, "tangent")?;
    validate_finite(values, "values")?;
    validate_finite(tangent, "tangent")?;
    let result = matrix_frobenius_norm_squared_jvp_inner(dimension, values, tangent)
        .map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust matrix Frobenius-squared VJP evaluation.
#[pyfunction]
pub fn matrix_frobenius_norm_squared_vjp<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let cotangent = validate_contiguous_slice(&cotangent, "cotangent")?;
    validate_finite(values, "values")?;
    validate_finite(cotangent, "cotangent")?;
    let result = matrix_frobenius_norm_squared_vjp_inner(dimension, values, cotangent)
        .map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust matrix Frobenius-squared gradient evaluation.
#[pyfunction]
pub fn matrix_frobenius_norm_squared_gradient<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result =
        matrix_frobenius_norm_squared_gradient_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust 2x2 determinant value evaluation.
#[pyfunction]
pub fn matrix_2x2_determinant_value<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_2x2_determinant_value_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust 2x2 determinant JVP evaluation.
#[pyfunction]
pub fn matrix_2x2_determinant_jvp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let tangent = validate_contiguous_slice(&tangent, "tangent")?;
    validate_finite(values, "values")?;
    validate_finite(tangent, "tangent")?;
    let result = matrix_2x2_determinant_jvp_inner(values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust 2x2 determinant VJP evaluation.
#[pyfunction]
pub fn matrix_2x2_determinant_vjp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let cotangent = validate_contiguous_slice(&cotangent, "cotangent")?;
    validate_finite(values, "values")?;
    validate_finite(cotangent, "cotangent")?;
    let result = matrix_2x2_determinant_vjp_inner(values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust 2x2 determinant gradient evaluation.
#[pyfunction]
pub fn matrix_2x2_determinant_gradient<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_2x2_determinant_gradient_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust 2x2 inverse value evaluation.
#[pyfunction]
pub fn matrix_2x2_inverse_value<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_2x2_inverse_value_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust 2x2 inverse JVP evaluation.
#[pyfunction]
pub fn matrix_2x2_inverse_jvp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let tangent = validate_contiguous_slice(&tangent, "tangent")?;
    validate_finite(values, "values")?;
    validate_finite(tangent, "tangent")?;
    let result = matrix_2x2_inverse_jvp_inner(values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust 2x2 inverse VJP evaluation.
#[pyfunction]
pub fn matrix_2x2_inverse_vjp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let cotangent = validate_contiguous_slice(&cotangent, "cotangent")?;
    validate_finite(values, "values")?;
    validate_finite(cotangent, "cotangent")?;
    let result = matrix_2x2_inverse_vjp_inner(values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust 2x2 inverse sum-output gradient provenance.
#[pyfunction]
pub fn matrix_2x2_inverse_sum_gradient<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_2x2_inverse_sum_gradient_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust 2x2 solve value evaluation.
#[pyfunction]
pub fn matrix_2x2_solve_value<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_2x2_solve_value_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust 2x2 solve JVP evaluation.
#[pyfunction]
pub fn matrix_2x2_solve_jvp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let tangent = validate_contiguous_slice(&tangent, "tangent")?;
    validate_finite(values, "values")?;
    validate_finite(tangent, "tangent")?;
    let result = matrix_2x2_solve_jvp_inner(values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust 2x2 solve VJP evaluation.
#[pyfunction]
pub fn matrix_2x2_solve_vjp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let cotangent = validate_contiguous_slice(&cotangent, "cotangent")?;
    validate_finite(values, "values")?;
    validate_finite(cotangent, "cotangent")?;
    let result = matrix_2x2_solve_vjp_inner(values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust 2x2 solve sum-output gradient provenance.
#[pyfunction]
pub fn matrix_2x2_solve_sum_gradient<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_2x2_solve_sum_gradient_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust vector dot value evaluation.
#[pyfunction]
pub fn vector_dot_value<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = vector_dot_value_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust vector dot JVP evaluation.
#[pyfunction]
pub fn vector_dot_jvp<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let tangent = validate_contiguous_slice(&tangent, "tangent")?;
    validate_finite(values, "values")?;
    validate_finite(tangent, "tangent")?;
    let result = vector_dot_jvp_inner(dimension, values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust vector dot VJP evaluation.
#[pyfunction]
pub fn vector_dot_vjp<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let cotangent = validate_contiguous_slice(&cotangent, "cotangent")?;
    validate_finite(values, "values")?;
    validate_finite(cotangent, "cotangent")?;
    let result = vector_dot_vjp_inner(dimension, values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust vector dot gradient evaluation.
#[pyfunction]
pub fn vector_dot_gradient<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = vector_dot_gradient_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust vector squared-norm value evaluation.
#[pyfunction]
pub fn vector_squared_norm_value<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = vector_squared_norm_value_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust vector squared-norm JVP evaluation.
#[pyfunction]
pub fn vector_squared_norm_jvp<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let tangent = validate_contiguous_slice(&tangent, "tangent")?;
    validate_finite(values, "values")?;
    validate_finite(tangent, "tangent")?;
    let result =
        vector_squared_norm_jvp_inner(dimension, values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for Rust vector squared-norm VJP evaluation.
#[pyfunction]
pub fn vector_squared_norm_vjp<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let cotangent = validate_contiguous_slice(&cotangent, "cotangent")?;
    validate_finite(values, "values")?;
    validate_finite(cotangent, "cotangent")?;
    let result =
        vector_squared_norm_vjp_inner(dimension, values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust vector squared-norm gradient evaluation.
#[pyfunction]
pub fn vector_squared_norm_gradient<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = vector_squared_norm_gradient_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
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
    fn matrix_2x2_eigensystem_value_matches_closed_form() {
        let values = [2.0, 0.25, 0.75, 1.0];
        let result = matrix_2x2_eigensystem_value_inner(&values).unwrap();
        assert_close(
            &result,
            &[
                0.838_562_172_233_852_3,
                2.161_437_827_766_147_5,
                0.210_430_715_716_423_36,
                0.840_070_779_091_305_9,
                -0.977_608_773_427_833_9,
                0.542_476_806_985_263,
            ],
        );
    }

    #[test]
    fn matrix_2x2_eigensystem_jvp_matches_closed_form() {
        let values = [2.0, 0.25, 0.75, 1.0];
        let tangent = [0.1, -0.2, 0.4, -0.3];
        let result = matrix_2x2_eigensystem_jvp_inner(&values, &tangent).unwrap();
        assert_close(
            &result,
            &[
                -0.213_389_341_902_768_15,
                0.013_389_341_902_768_165,
                -0.215_156_061_857_423_42,
                -0.065_142_791_863_097_92,
                -0.046_312_436_342_641_75,
                0.100_879_070_234_798_1,
            ],
        );
    }

    #[test]
    fn matrix_2x2_eigensystem_vjp_matches_jvp_transpose_reference() {
        let values = [2.0, 0.25, 0.75, 1.0];
        let cotangent = [1.25, -0.75, 0.5, -0.25, 0.3, -0.6];
        let result = matrix_2x2_eigensystem_vjp_inner(&values, &cotangent).unwrap();
        assert_close(
            &result,
            &[
                -0.464_840_980_381_554_03,
                -0.653_188_087_262_615_7,
                -0.592_983_537_780_116_1,
                0.964_840_980_381_554_1,
            ],
        );
    }

    #[test]
    fn matrix_2x2_eigensystem_boundaries_fail_closed() {
        let nonreal = matrix_2x2_eigensystem_value_inner(&[0.0, -1.0, 1.0, 0.0]).unwrap_err();
        assert!(nonreal.contains("real distinct eigenvalues"));
        let repeated = matrix_2x2_eigensystem_value_inner(&[1.0, 0.0, 0.0, 1.0]).unwrap_err();
        assert!(repeated.contains("real distinct eigenvalues"));
        let zero_chart = matrix_2x2_eigensystem_value_inner(&[2.0, 0.0, 1.0, 1.0]).unwrap_err();
        assert!(zero_chart.contains("upper off-diagonal eigenvector chart"));
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
    fn matrix_2x2_determinant_value_jvp_vjp_and_gradient_match_closed_form() {
        let values = [2.0, -1.0, 0.5, 3.0];
        let tangent = [0.1, -0.2, 0.3, 0.4];
        let cotangent = [1.25];

        assert_close(
            &matrix_2x2_determinant_value_inner(&values).unwrap(),
            &[6.5],
        );
        assert_close(
            &matrix_2x2_determinant_jvp_inner(&values, &tangent).unwrap(),
            &[1.5],
        );
        assert_close(
            &matrix_2x2_determinant_vjp_inner(&values, &cotangent).unwrap(),
            &[3.75, -0.625, 1.25, 2.5],
        );
        assert_close(
            &matrix_2x2_determinant_gradient_inner(&values).unwrap(),
            &[3.0, -0.5, 1.0, 2.0],
        );
    }

    #[test]
    fn matrix_2x2_determinant_boundaries_fail_closed() {
        let wrong_count = matrix_2x2_determinant_value_inner(&[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("row-major 2x2 matrix values"));
        let non_finite =
            matrix_2x2_determinant_gradient_inner(&[2.0, f64::NAN, 0.5, 3.0]).unwrap_err();
        assert!(non_finite.contains("not finite"));
        let tangent_count =
            matrix_2x2_determinant_jvp_inner(&[2.0, -1.0, 0.5, 3.0], &[1.0]).unwrap_err();
        assert!(tangent_count.contains("row-major 2x2 matrix tangent values"));
        let cotangent_count =
            matrix_2x2_determinant_vjp_inner(&[2.0, -1.0, 0.5, 3.0], &[1.0, 2.0]).unwrap_err();
        assert!(cotangent_count.contains("requires 1 cotangent value"));
    }

    #[test]
    fn matrix_2x2_inverse_value_jvp_vjp_and_sum_gradient_match_closed_form() {
        let values = [2.0, -1.0, 0.5, 3.0];
        let tangent = [0.1, -0.2, 0.3, 0.4];
        let cotangent = [0.75, -1.25, 0.5, 2.0];

        assert_close(
            &matrix_2x2_inverse_value_inner(&values).unwrap(),
            &[
                0.461_538_461_538_461_56,
                0.153_846_153_846_153_85,
                -0.076_923_076_923_076_93,
                0.307_692_307_692_307_7,
            ],
        );
        assert_close(
            &matrix_2x2_inverse_jvp_inner(&values, &tangent).unwrap(),
            &[
                -0.044_970_414_201_183_43,
                -0.004_733_727_810_650_887,
                -0.028_402_366_863_905_325,
                -0.055_621_301_775_147_93,
            ],
        );
        assert_close(
            &matrix_2x2_inverse_vjp_inner(&values, &cotangent).unwrap(),
            &[
                -0.029_585_798_816_568_046,
                0.248_520_710_059_171_6,
                -0.189_349_112_426_035_5,
                -0.109_467_455_621_301_78,
            ],
        );
        assert_close(
            &matrix_2x2_inverse_sum_gradient_inner(&values).unwrap(),
            &[
                -0.236_686_390_532_544_37,
                -0.088_757_396_449_704_14,
                -0.284_023_668_639_053_26,
                -0.106_508_875_739_644_97,
            ],
        );
    }

    #[test]
    fn matrix_2x2_inverse_boundaries_fail_closed() {
        let wrong_count = matrix_2x2_inverse_value_inner(&[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("row-major 2x2 matrix values"));
        let non_finite =
            matrix_2x2_inverse_sum_gradient_inner(&[2.0, f64::NAN, 0.5, 3.0]).unwrap_err();
        assert!(non_finite.contains("not finite"));
        let singular = matrix_2x2_inverse_value_inner(&[1.0, 2.0, 2.0, 4.0]).unwrap_err();
        assert!(singular.contains("nonsingular row-major 2x2 matrix"));
        let tangent_count =
            matrix_2x2_inverse_jvp_inner(&[2.0, -1.0, 0.5, 3.0], &[1.0]).unwrap_err();
        assert!(tangent_count.contains("row-major 2x2 matrix tangent values"));
        let cotangent_count =
            matrix_2x2_inverse_vjp_inner(&[2.0, -1.0, 0.5, 3.0], &[1.0, 2.0]).unwrap_err();
        assert!(cotangent_count.contains("requires 4 cotangent value"));
    }

    #[test]
    fn matrix_2x2_solve_value_jvp_vjp_and_sum_gradient_match_closed_form() {
        let values = [2.0, -1.0, 0.5, 3.0, 1.5, -2.0];
        let tangent = [0.1, -0.2, 0.3, 0.4, -0.5, 0.75];
        let cotangent = [1.25, -0.75];

        assert_close(
            &matrix_2x2_solve_value_inner(&values).unwrap(),
            &[0.384_615_384_615_384_64, -0.730_769_230_769_230_7],
        );
        assert_close(
            &matrix_2x2_solve_jvp_inner(&values, &tangent).unwrap(),
            &[-0.173_372_781_065_088_76, 0.337_869_822_485_207_1],
        );
        assert_close(
            &matrix_2x2_solve_vjp_inner(&values, &cotangent).unwrap(),
            &[
                -0.244_082_840_236_686_4,
                0.463_757_396_449_704_15,
                0.014_792_899_408_284_023,
                -0.028_106_508_875_739_646,
                0.634_615_384_615_384_6,
                -0.038_461_538_461_538_464,
            ],
        );
        assert_close(
            &matrix_2x2_solve_sum_gradient_inner(&values).unwrap(),
            &[
                -0.147_928_994_082_840_24,
                0.281_065_088_757_396_44,
                -0.177_514_792_899_408_27,
                0.337_278_106_508_875_74,
                0.384_615_384_615_384_64,
                0.461_538_461_538_461_56,
            ],
        );
    }

    #[test]
    fn matrix_2x2_solve_boundaries_fail_closed() {
        let wrong_count = matrix_2x2_solve_value_inner(&[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("matrix plus rhs values"));
        let non_finite =
            matrix_2x2_solve_sum_gradient_inner(&[2.0, -1.0, 0.5, 3.0, f64::NAN, -2.0])
                .unwrap_err();
        assert!(non_finite.contains("not finite"));
        let singular = matrix_2x2_solve_value_inner(&[1.0, 2.0, 2.0, 4.0, 1.0, -1.0]).unwrap_err();
        assert!(singular.contains("nonsingular row-major 2x2 matrix"));
        let tangent_count =
            matrix_2x2_solve_jvp_inner(&[2.0, -1.0, 0.5, 3.0, 1.5, -2.0], &[1.0]).unwrap_err();
        assert!(tangent_count.contains("matrix plus rhs tangent values"));
        let cotangent_count =
            matrix_2x2_solve_vjp_inner(&[2.0, -1.0, 0.5, 3.0, 1.5, -2.0], &[1.0]).unwrap_err();
        assert!(cotangent_count.contains("requires 2 cotangent value"));
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
