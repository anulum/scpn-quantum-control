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

mod kernels;

pub use kernels::{
    matrix_2x2_determinant_gradient_inner, matrix_2x2_determinant_jvp_inner,
    matrix_2x2_determinant_value_inner, matrix_2x2_determinant_vjp_inner,
    matrix_2x2_eigensystem_jvp_inner, matrix_2x2_eigensystem_sum_gradient_inner,
    matrix_2x2_eigensystem_value_inner, matrix_2x2_eigensystem_vjp_inner,
    matrix_2x2_eigenvalues_jvp_inner, matrix_2x2_eigenvalues_sum_gradient_inner,
    matrix_2x2_eigenvalues_value_inner, matrix_2x2_eigenvalues_vjp_inner,
    matrix_2x2_inverse_jvp_inner, matrix_2x2_inverse_sum_gradient_inner,
    matrix_2x2_inverse_value_inner, matrix_2x2_inverse_vjp_inner, matrix_2x2_solve_jvp_inner,
    matrix_2x2_solve_sum_gradient_inner, matrix_2x2_solve_value_inner, matrix_2x2_solve_vjp_inner,
    matrix_frobenius_norm_squared_gradient_inner, matrix_frobenius_norm_squared_jvp_inner,
    matrix_frobenius_norm_squared_value_inner, matrix_frobenius_norm_squared_vjp_inner,
    matrix_matrix_product_jvp_inner, matrix_matrix_product_sum_gradient_inner,
    matrix_matrix_product_value_inner, matrix_matrix_product_vjp_inner,
    matrix_quadratic_form_gradient_inner, matrix_quadratic_form_jvp_inner,
    matrix_quadratic_form_value_inner, matrix_quadratic_form_vjp_inner,
    matrix_trace_gradient_inner, matrix_trace_jvp_inner, matrix_trace_value_inner,
    matrix_trace_vjp_inner, matrix_vector_product_jvp_inner,
    matrix_vector_product_sum_gradient_inner, matrix_vector_product_value_inner,
    matrix_vector_product_vjp_inner, symmetric_2x2_cholesky_jvp_inner,
    symmetric_2x2_cholesky_sum_gradient_inner, symmetric_2x2_cholesky_value_inner,
    symmetric_2x2_cholesky_vjp_inner, symmetric_2x2_eigenvalues_jvp_inner,
    symmetric_2x2_eigenvalues_sum_gradient_inner, symmetric_2x2_eigenvalues_value_inner,
    symmetric_2x2_eigenvalues_vjp_inner, vector_dot_gradient_inner, vector_dot_jvp_inner,
    vector_dot_value_inner, vector_dot_vjp_inner, vector_squared_norm_gradient_inner,
    vector_squared_norm_jvp_inner, vector_squared_norm_value_inner, vector_squared_norm_vjp_inner,
};

fn py_value_error(error: String) -> PyErr {
    PyValueError::new_err(error)
}

/// PyO3 wrapper for bounded Rust eigenvalue evaluation.
#[pyfunction]
pub fn matrix_2x2_eigenvalues_value<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_2x2_eigenvalues_value_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust eigenvalue JVP evaluation.
#[pyfunction]
pub fn matrix_2x2_eigenvalues_jvp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let tangent = validate_contiguous_slice(&tangent, "tangent")?;
    validate_finite(values, "values")?;
    validate_finite(tangent, "tangent")?;
    let result = matrix_2x2_eigenvalues_jvp_inner(values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust eigenvalue VJP evaluation.
#[pyfunction]
pub fn matrix_2x2_eigenvalues_vjp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let cotangent = validate_contiguous_slice(&cotangent, "cotangent")?;
    validate_finite(values, "values")?;
    validate_finite(cotangent, "cotangent")?;
    let result = matrix_2x2_eigenvalues_vjp_inner(values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust eigenvalue sum-output gradient provenance.
#[pyfunction]
pub fn matrix_2x2_eigenvalues_sum_gradient<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_2x2_eigenvalues_sum_gradient_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
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

/// PyO3 wrapper for Rust matrix-vector product value evaluation.
#[pyfunction]
pub fn matrix_vector_product_value<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_vector_product_value_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust matrix-vector product JVP evaluation.
#[pyfunction]
pub fn matrix_vector_product_jvp<'py>(
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
        matrix_vector_product_jvp_inner(dimension, values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust matrix-vector product VJP evaluation.
#[pyfunction]
pub fn matrix_vector_product_vjp<'py>(
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
        matrix_vector_product_vjp_inner(dimension, values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust matrix-vector product sum-output gradient provenance.
#[pyfunction]
pub fn matrix_vector_product_sum_gradient<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result =
        matrix_vector_product_sum_gradient_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust matrix-matrix product value evaluation.
#[pyfunction]
pub fn matrix_matrix_product_value<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = matrix_matrix_product_value_inner(dimension, values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust matrix-matrix product JVP evaluation.
#[pyfunction]
pub fn matrix_matrix_product_jvp<'py>(
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
        matrix_matrix_product_jvp_inner(dimension, values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust matrix-matrix product VJP evaluation.
#[pyfunction]
pub fn matrix_matrix_product_vjp<'py>(
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
        matrix_matrix_product_vjp_inner(dimension, values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result))
}

/// PyO3 wrapper for Rust matrix-matrix product sum-output gradient provenance.
#[pyfunction]
pub fn matrix_matrix_product_sum_gradient<'py>(
    py: Python<'py>,
    dimension: usize,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result =
        matrix_matrix_product_sum_gradient_inner(dimension, values).map_err(py_value_error)?;
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

/// PyO3 wrapper for bounded Rust SPD symmetric 2x2 Cholesky value evaluation.
#[pyfunction]
pub fn symmetric_2x2_cholesky_value<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = symmetric_2x2_cholesky_value_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust SPD symmetric 2x2 Cholesky JVP evaluation.
#[pyfunction]
pub fn symmetric_2x2_cholesky_jvp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let tangent = validate_contiguous_slice(&tangent, "tangent")?;
    validate_finite(values, "values")?;
    validate_finite(tangent, "tangent")?;
    let result = symmetric_2x2_cholesky_jvp_inner(values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust SPD symmetric 2x2 Cholesky VJP evaluation.
#[pyfunction]
pub fn symmetric_2x2_cholesky_vjp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let cotangent = validate_contiguous_slice(&cotangent, "cotangent")?;
    validate_finite(values, "values")?;
    validate_finite(cotangent, "cotangent")?;
    let result = symmetric_2x2_cholesky_vjp_inner(values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust SPD symmetric 2x2 Cholesky sum-gradient provenance.
#[pyfunction]
pub fn symmetric_2x2_cholesky_sum_gradient<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = symmetric_2x2_cholesky_sum_gradient_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust distinct symmetric 2x2 eigenvalue evaluation.
#[pyfunction]
pub fn symmetric_2x2_eigenvalues_value<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = symmetric_2x2_eigenvalues_value_inner(values).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust distinct symmetric 2x2 eigenvalue JVP evaluation.
#[pyfunction]
pub fn symmetric_2x2_eigenvalues_jvp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let tangent = validate_contiguous_slice(&tangent, "tangent")?;
    validate_finite(values, "values")?;
    validate_finite(tangent, "tangent")?;
    let result = symmetric_2x2_eigenvalues_jvp_inner(values, tangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust distinct symmetric 2x2 eigenvalue VJP evaluation.
#[pyfunction]
pub fn symmetric_2x2_eigenvalues_vjp<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    let cotangent = validate_contiguous_slice(&cotangent, "cotangent")?;
    validate_finite(values, "values")?;
    validate_finite(cotangent, "cotangent")?;
    let result = symmetric_2x2_eigenvalues_vjp_inner(values, cotangent).map_err(py_value_error)?;
    Ok(PyArray1::from_vec(py, result.to_vec()))
}

/// PyO3 wrapper for bounded Rust symmetric 2x2 eigenvalue sum-gradient provenance.
#[pyfunction]
pub fn symmetric_2x2_eigenvalues_sum_gradient<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = validate_contiguous_slice(&values, "values")?;
    validate_finite(values, "values")?;
    let result = symmetric_2x2_eigenvalues_sum_gradient_inner(values).map_err(py_value_error)?;
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
