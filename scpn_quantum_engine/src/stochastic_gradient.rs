// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — stochastic gradient uncertainty kernels

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct StochasticGradientUncertaintyResult {
    pub gradient: Vec<f64>,
    pub standard_error: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub confidence_radius: Vec<f64>,
}

fn ensure_finite_matrix(name: &str, values: &[Vec<f64>]) -> Result<(usize, usize), String> {
    if values.is_empty() {
        return Err(format!("{name} must have at least one shift term"));
    }
    let columns = values[0].len();
    if columns == 0 {
        return Err(format!("{name} must have at least one parameter column"));
    }
    for (row_index, row) in values.iter().enumerate() {
        if row.len() != columns {
            return Err(format!(
                "{name} rows must share width {columns}; row {row_index} has width {}",
                row.len()
            ));
        }
        for (column_index, value) in row.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "{name}[{row_index}][{column_index}] must be finite, got {value}"
                ));
            }
        }
    }
    Ok((values.len(), columns))
}

fn ensure_same_shape(
    name: &str,
    values: &[Vec<f64>],
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(), String> {
    let (rows, columns) = ensure_finite_matrix(name, values)?;
    if rows != expected_rows || columns != expected_columns {
        return Err(format!(
            "{name} shape must be ({expected_rows}, {expected_columns}), got ({rows}, {columns})"
        ));
    }
    Ok(())
}

fn validate_variances(name: &str, values: &[Vec<f64>]) -> Result<(), String> {
    for (row_index, row) in values.iter().enumerate() {
        for (column_index, value) in row.iter().copied().enumerate() {
            if value < 0.0 {
                return Err(format!(
                    "{name}[{row_index}][{column_index}] must be non-negative, got {value}"
                ));
            }
        }
    }
    Ok(())
}

fn validate_shots(name: &str, values: &[Vec<f64>]) -> Result<(), String> {
    for (row_index, row) in values.iter().enumerate() {
        for (column_index, value) in row.iter().copied().enumerate() {
            if value <= 0.0 || (value - value.round()).abs() > 0.0 {
                return Err(format!(
                    "{name}[{row_index}][{column_index}] must be a positive integer count, got {value}"
                ));
            }
        }
    }
    Ok(())
}

pub fn stochastic_parameter_shift_uncertainty_inner(
    plus_values: &[Vec<f64>],
    minus_values: &[Vec<f64>],
    plus_variances: &[Vec<f64>],
    minus_variances: &[Vec<f64>],
    plus_shots: &[Vec<f64>],
    minus_shots: &[Vec<f64>],
    coefficients: &[f64],
    trainable: &[bool],
    confidence_z: f64,
) -> Result<StochasticGradientUncertaintyResult, String> {
    let (term_count, parameter_count) = ensure_finite_matrix("plus_values", plus_values)?;
    ensure_same_shape("minus_values", minus_values, term_count, parameter_count)?;
    ensure_same_shape(
        "plus_variances",
        plus_variances,
        term_count,
        parameter_count,
    )?;
    ensure_same_shape(
        "minus_variances",
        minus_variances,
        term_count,
        parameter_count,
    )?;
    ensure_same_shape("plus_shots", plus_shots, term_count, parameter_count)?;
    ensure_same_shape("minus_shots", minus_shots, term_count, parameter_count)?;
    if coefficients.len() != term_count {
        return Err(format!(
            "coefficients length must match shift term count: {} != {term_count}",
            coefficients.len()
        ));
    }
    if trainable.len() != parameter_count {
        return Err(format!(
            "trainable length must match parameter count: {} != {parameter_count}",
            trainable.len()
        ));
    }
    for (index, coefficient) in coefficients.iter().copied().enumerate() {
        if !coefficient.is_finite() {
            return Err(format!(
                "coefficients[{index}] must be finite, got {coefficient}"
            ));
        }
    }
    if !confidence_z.is_finite() || confidence_z <= 0.0 {
        return Err(format!(
            "confidence_z must be a finite positive value, got {confidence_z}"
        ));
    }
    validate_variances("plus_variances", plus_variances)?;
    validate_variances("minus_variances", minus_variances)?;
    validate_shots("plus_shots", plus_shots)?;
    validate_shots("minus_shots", minus_shots)?;

    let mut gradient = vec![0.0; parameter_count];
    let mut variance = vec![0.0; parameter_count];
    for parameter in 0..parameter_count {
        if !trainable[parameter] {
            continue;
        }
        for term in 0..term_count {
            let coefficient = coefficients[term];
            gradient[parameter] +=
                coefficient * (plus_values[term][parameter] - minus_values[term][parameter]);
            variance[parameter] += coefficient
                * coefficient
                * (plus_variances[term][parameter] / plus_shots[term][parameter]
                    + minus_variances[term][parameter] / minus_shots[term][parameter]);
        }
    }
    let standard_error: Vec<f64> = variance.iter().map(|value| value.sqrt()).collect();
    let confidence_radius: Vec<f64> = standard_error
        .iter()
        .map(|value| confidence_z * value)
        .collect();
    let mut covariance = vec![vec![0.0; parameter_count]; parameter_count];
    for index in 0..parameter_count {
        covariance[index][index] = variance[index];
    }
    Ok(StochasticGradientUncertaintyResult {
        gradient,
        standard_error,
        covariance,
        confidence_radius,
    })
}

fn read_array2_rows(values: PyReadonlyArray2<'_, f64>) -> Vec<Vec<f64>> {
    values
        .as_array()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect()
}

fn nested_to_array2(values: Vec<Vec<f64>>) -> Result<Array2<f64>, String> {
    let rows = values.len();
    let columns = values.first().map_or(0, Vec::len);
    let flat: Vec<f64> = values.into_iter().flatten().collect();
    Array2::from_shape_vec((rows, columns), flat).map_err(|err| err.to_string())
}

#[pyfunction]
#[pyo3(signature = (
    plus_values,
    minus_values,
    plus_variances,
    minus_variances,
    plus_shots,
    minus_shots,
    coefficients,
    trainable,
    confidence_z=1.959963984540054
))]
pub fn parameter_shift_gradient_uncertainty_rust<'py>(
    py: Python<'py>,
    plus_values: PyReadonlyArray2<'_, f64>,
    minus_values: PyReadonlyArray2<'_, f64>,
    plus_variances: PyReadonlyArray2<'_, f64>,
    minus_variances: PyReadonlyArray2<'_, f64>,
    plus_shots: PyReadonlyArray2<'_, f64>,
    minus_shots: PyReadonlyArray2<'_, f64>,
    coefficients: PyReadonlyArray1<'_, f64>,
    trainable: PyReadonlyArray1<'_, bool>,
    confidence_z: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let result = stochastic_parameter_shift_uncertainty_inner(
        &read_array2_rows(plus_values),
        &read_array2_rows(minus_values),
        &read_array2_rows(plus_variances),
        &read_array2_rows(minus_variances),
        &read_array2_rows(plus_shots),
        &read_array2_rows(minus_shots),
        coefficients.as_slice()?,
        trainable.as_slice()?,
        confidence_z,
    )
    .map_err(PyValueError::new_err)?;
    let covariance = nested_to_array2(result.covariance).map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, result.gradient),
        PyArray1::from_vec(py, result.standard_error),
        PyArray2::from_owned_array(py, covariance),
        PyArray1::from_vec(py, result.confidence_radius),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-12, "{actual} != {expected}");
    }

    #[test]
    fn stochastic_parameter_shift_uncertainty_matches_single_term_reference() {
        let plus_values = [vec![0.8, 0.1]];
        let minus_values = [vec![0.2, -0.3]];
        let plus_variances = [vec![0.36, 0.25]];
        let minus_variances = [vec![0.16, 0.09]];
        let plus_shots = [vec![900.0, 400.0]];
        let minus_shots = [vec![400.0, 100.0]];
        let result = stochastic_parameter_shift_uncertainty_inner(
            &plus_values,
            &minus_values,
            &plus_variances,
            &minus_variances,
            &plus_shots,
            &minus_shots,
            &[0.5],
            &[true, false],
            1.959963984540054,
        )
        .unwrap();

        let expected_variance = 0.5_f64.powi(2) * (0.36 / 900.0 + 0.16 / 400.0);
        assert_close(result.gradient[0], 0.3);
        assert_close(result.gradient[1], 0.0);
        assert_close(result.standard_error[0], expected_variance.sqrt());
        assert_close(result.standard_error[1], 0.0);
        assert_close(result.covariance[0][0], expected_variance);
        assert_close(result.covariance[1][1], 0.0);
    }

    #[test]
    fn stochastic_parameter_shift_uncertainty_rejects_invalid_contracts() {
        let valid = [vec![1.0]];
        let invalid_variance = [vec![-0.1]];
        let invalid_shots = [vec![0.5]];
        assert!(stochastic_parameter_shift_uncertainty_inner(
            &valid,
            &valid,
            &invalid_variance,
            &valid,
            &valid,
            &valid,
            &[0.5],
            &[true],
            1.0,
        )
        .is_err());
        assert!(stochastic_parameter_shift_uncertainty_inner(
            &valid,
            &valid,
            &valid,
            &valid,
            &invalid_shots,
            &valid,
            &[0.5],
            &[true],
            1.0,
        )
        .is_err());
    }
}
