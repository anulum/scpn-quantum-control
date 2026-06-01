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
}
