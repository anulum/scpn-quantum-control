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

#[derive(Debug, Clone, PartialEq)]
pub struct SPSAGradientKernelResult {
    pub gradient: Vec<f64>,
    pub standard_error: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub confidence_radius: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoreFunctionGradientKernelResult {
    pub gradient: Vec<f64>,
    pub standard_error: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub confidence_radius: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GradientConfidenceIntervalKernelResult {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub status: String,
    pub failure_reasons: Vec<String>,
}

type StochasticGradientPyResult<'py> = PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)>;

type GradientConfidenceIntervalPyResult<'py> = PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    String,
    Vec<String>,
)>;

fn ensure_finite_vector(name: &str, values: &[f64]) -> Result<(), String> {
    if values.is_empty() {
        return Err(format!("{name} must be non-empty"));
    }
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(format!("{name}[{index}] must be finite, got {value}"));
        }
    }
    Ok(())
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

#[expect(
    clippy::too_many_arguments,
    reason = "kernel inputs mirror the finite-shot parameter-shift evidence schema"
)]
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

#[expect(
    clippy::too_many_arguments,
    reason = "kernel inputs mirror the SPSA evidence schema"
)]
pub fn spsa_gradient_inner(
    plus_values: &[f64],
    minus_values: &[f64],
    perturbations: &[Vec<f64>],
    plus_variances: Option<&[f64]>,
    minus_variances: Option<&[f64]>,
    plus_shots: Option<&[f64]>,
    minus_shots: Option<&[f64]>,
    trainable: &[bool],
    perturbation_radius: f64,
    confidence_z: f64,
) -> Result<SPSAGradientKernelResult, String> {
    ensure_finite_vector("plus_values", plus_values)?;
    ensure_finite_vector("minus_values", minus_values)?;
    if plus_values.len() != minus_values.len() {
        return Err(format!(
            "minus_values length must match plus_values length: {} != {}",
            minus_values.len(),
            plus_values.len()
        ));
    }
    let repetitions = plus_values.len();
    if perturbations.len() != repetitions {
        return Err(format!(
            "perturbation row count must match repetitions: {} != {repetitions}",
            perturbations.len()
        ));
    }
    if !perturbation_radius.is_finite() || perturbation_radius <= 0.0 {
        return Err(format!(
            "perturbation_radius must be a finite positive value, got {perturbation_radius}"
        ));
    }
    if !confidence_z.is_finite() || confidence_z <= 0.0 {
        return Err(format!(
            "confidence_z must be a finite positive value, got {confidence_z}"
        ));
    }
    if trainable.is_empty() {
        return Err("trainable mask must be non-empty".to_string());
    }
    let parameter_count = trainable.len();
    for (row_index, row) in perturbations.iter().enumerate() {
        if row.len() != parameter_count {
            return Err(format!(
                "perturbation row {row_index} width must match trainable mask: {} != {parameter_count}",
                row.len()
            ));
        }
        for (column_index, value) in row.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "perturbations[{row_index}][{column_index}] must be finite, got {value}"
                ));
            }
            if trainable[column_index] && value.abs() != 1.0 {
                return Err(format!(
                    "trainable perturbations[{row_index}][{column_index}] must be +/-1, got {value}"
                ));
            }
            if !trainable[column_index] && value != 0.0 {
                return Err(format!(
                    "frozen perturbations[{row_index}][{column_index}] must be 0, got {value}"
                ));
            }
        }
    }

    let finite_shot = plus_variances.is_some()
        || minus_variances.is_some()
        || plus_shots.is_some()
        || minus_shots.is_some();
    let mut shot_variance = vec![0.0; parameter_count];
    let plus_variances = match (finite_shot, plus_variances) {
        (true, Some(values)) => {
            ensure_finite_vector("plus_variances", values)?;
            Some(values)
        }
        (true, None) => return Err("plus_variances are required for finite-shot SPSA".to_string()),
        (false, _) => None,
    };
    let minus_variances = match (finite_shot, minus_variances) {
        (true, Some(values)) => {
            ensure_finite_vector("minus_variances", values)?;
            Some(values)
        }
        (true, None) => return Err("minus_variances are required for finite-shot SPSA".to_string()),
        (false, _) => None,
    };
    let plus_shots = match (finite_shot, plus_shots) {
        (true, Some(values)) => {
            ensure_finite_vector("plus_shots", values)?;
            validate_shot_vector("plus_shots", values)?;
            Some(values)
        }
        (true, None) => return Err("plus_shots are required for finite-shot SPSA".to_string()),
        (false, _) => None,
    };
    let minus_shots = match (finite_shot, minus_shots) {
        (true, Some(values)) => {
            ensure_finite_vector("minus_shots", values)?;
            validate_shot_vector("minus_shots", values)?;
            Some(values)
        }
        (true, None) => return Err("minus_shots are required for finite-shot SPSA".to_string()),
        (false, _) => None,
    };
    if finite_shot {
        for (name, values) in [
            ("plus_variances", plus_variances.unwrap()),
            ("minus_variances", minus_variances.unwrap()),
            ("plus_shots", plus_shots.unwrap()),
            ("minus_shots", minus_shots.unwrap()),
        ] {
            if values.len() != repetitions {
                return Err(format!(
                    "{name} length must match repetitions: {} != {repetitions}",
                    values.len()
                ));
            }
        }
        validate_variance_vector("plus_variances", plus_variances.unwrap())?;
        validate_variance_vector("minus_variances", minus_variances.unwrap())?;
    }

    let mut estimates = vec![vec![0.0; parameter_count]; repetitions];
    for repetition in 0..repetitions {
        let difference = plus_values[repetition] - minus_values[repetition];
        for parameter in 0..parameter_count {
            if !trainable[parameter] {
                continue;
            }
            let delta = perturbations[repetition][parameter];
            estimates[repetition][parameter] = difference / (2.0 * perturbation_radius * delta);
            if finite_shot {
                shot_variance[parameter] += (plus_variances.unwrap()[repetition]
                    / plus_shots.unwrap()[repetition]
                    + minus_variances.unwrap()[repetition] / minus_shots.unwrap()[repetition])
                    / (4.0 * perturbation_radius * perturbation_radius);
            }
        }
    }

    let mut gradient = vec![0.0; parameter_count];
    for parameter in 0..parameter_count {
        gradient[parameter] =
            estimates.iter().map(|row| row[parameter]).sum::<f64>() / repetitions as f64;
    }
    let mut variance = vec![0.0; parameter_count];
    if repetitions > 1 {
        for parameter in 0..parameter_count {
            let sum_sq: f64 = estimates
                .iter()
                .map(|row| {
                    let residual = row[parameter] - gradient[parameter];
                    residual * residual
                })
                .sum();
            variance[parameter] += sum_sq / ((repetitions - 1) as f64 * repetitions as f64);
        }
    }
    if finite_shot {
        for parameter in 0..parameter_count {
            variance[parameter] += shot_variance[parameter] / (repetitions * repetitions) as f64;
        }
    }
    for parameter in 0..parameter_count {
        if !trainable[parameter] {
            gradient[parameter] = 0.0;
            variance[parameter] = 0.0;
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
    Ok(SPSAGradientKernelResult {
        gradient,
        standard_error,
        covariance,
        confidence_radius,
    })
}

pub fn score_function_gradient_inner(
    rewards: &[f64],
    score_vectors: &[Vec<f64>],
    trainable: &[bool],
    baseline: f64,
    confidence_z: f64,
) -> Result<ScoreFunctionGradientKernelResult, String> {
    ensure_finite_vector("rewards", rewards)?;
    if rewards.len() < 2 {
        return Err("score-function estimator requires at least two rewards".to_string());
    }
    let (sample_count, parameter_count) = ensure_finite_matrix("score_vectors", score_vectors)?;
    if sample_count != rewards.len() {
        return Err(format!(
            "score_vectors row count must match rewards length: {sample_count} != {}",
            rewards.len()
        ));
    }
    if trainable.len() != parameter_count {
        return Err(format!(
            "trainable length must match parameter count: {} != {parameter_count}",
            trainable.len()
        ));
    }
    if !trainable.iter().any(|flag| *flag) {
        return Err(
            "score-function estimator requires at least one trainable parameter".to_string(),
        );
    }
    if !baseline.is_finite() {
        return Err(format!("baseline must be finite, got {baseline}"));
    }
    if !confidence_z.is_finite() || confidence_z <= 0.0 {
        return Err(format!(
            "confidence_z must be a finite positive value, got {confidence_z}"
        ));
    }

    let mut estimates = vec![vec![0.0; parameter_count]; sample_count];
    for sample in 0..sample_count {
        let centred_reward = rewards[sample] - baseline;
        for parameter in 0..parameter_count {
            if trainable[parameter] {
                estimates[sample][parameter] = centred_reward * score_vectors[sample][parameter];
            }
        }
    }

    let mut gradient = vec![0.0; parameter_count];
    for parameter in 0..parameter_count {
        gradient[parameter] =
            estimates.iter().map(|row| row[parameter]).sum::<f64>() / sample_count as f64;
    }
    let mut covariance = vec![vec![0.0; parameter_count]; parameter_count];
    let scale = 1.0 / ((sample_count - 1) as f64 * sample_count as f64);
    for row in estimates.iter() {
        for i in 0..parameter_count {
            if !trainable[i] {
                continue;
            }
            let left = row[i] - gradient[i];
            for j in 0..parameter_count {
                if trainable[j] {
                    covariance[i][j] += left * (row[j] - gradient[j]) * scale;
                }
            }
        }
    }
    let standard_error: Vec<f64> = (0..parameter_count)
        .map(|index| covariance[index][index].sqrt())
        .collect();
    let confidence_radius: Vec<f64> = standard_error
        .iter()
        .map(|value| confidence_z * value)
        .collect();
    Ok(ScoreFunctionGradientKernelResult {
        gradient,
        standard_error,
        covariance,
        confidence_radius,
    })
}

pub fn gradient_confidence_interval_inner(
    gradient: &[f64],
    standard_error: &[f64],
    trainable: &[bool],
    confidence_z: f64,
    max_standard_error: Option<f64>,
    max_confidence_radius: Option<f64>,
) -> Result<GradientConfidenceIntervalKernelResult, String> {
    ensure_finite_vector("gradient", gradient)?;
    ensure_finite_vector("standard_error", standard_error)?;
    if gradient.len() != standard_error.len() {
        return Err(format!(
            "standard_error length must match gradient length: {} != {}",
            standard_error.len(),
            gradient.len()
        ));
    }
    if trainable.len() != gradient.len() {
        return Err(format!(
            "trainable length must match gradient length: {} != {}",
            trainable.len(),
            gradient.len()
        ));
    }
    if !trainable.iter().any(|flag| *flag) {
        return Err("trainable mask must include at least one active parameter".to_string());
    }
    if !confidence_z.is_finite() || confidence_z <= 0.0 {
        return Err(format!(
            "confidence_z must be a finite positive value, got {confidence_z}"
        ));
    }
    if let Some(limit) = max_standard_error {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(format!(
                "max_standard_error must be a finite positive value, got {limit}"
            ));
        }
    }
    if let Some(limit) = max_confidence_radius {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(format!(
                "max_confidence_radius must be a finite positive value, got {limit}"
            ));
        }
    }
    for (index, value) in standard_error.iter().copied().enumerate() {
        if value < 0.0 {
            return Err(format!(
                "standard_error[{index}] must be non-negative, got {value}"
            ));
        }
    }

    let mut lower = Vec::with_capacity(gradient.len());
    let mut upper = Vec::with_capacity(gradient.len());
    let mut failure_reasons = Vec::new();
    for (index, (&mean, &stderr)) in gradient.iter().zip(standard_error.iter()).enumerate() {
        let radius = confidence_z * stderr;
        lower.push(mean - radius);
        upper.push(mean + radius);
        if !trainable[index] {
            continue;
        }
        if let Some(limit) = max_standard_error {
            if stderr > limit {
                failure_reasons.push(format!(
                    "standard_error[{index}]={stderr} exceeds max_standard_error={limit}"
                ));
            }
        }
        if let Some(limit) = max_confidence_radius {
            if radius > limit {
                failure_reasons.push(format!(
                    "confidence_radius[{index}]={radius} exceeds max_confidence_radius={limit}"
                ));
            }
        }
    }
    let status = if failure_reasons.is_empty() {
        "passed".to_string()
    } else {
        "failed".to_string()
    };
    Ok(GradientConfidenceIntervalKernelResult {
        lower,
        upper,
        status,
        failure_reasons,
    })
}

fn validate_variance_vector(name: &str, values: &[f64]) -> Result<(), String> {
    for (index, value) in values.iter().copied().enumerate() {
        if value < 0.0 {
            return Err(format!("{name}[{index}] must be non-negative, got {value}"));
        }
    }
    Ok(())
}

fn validate_shot_vector(name: &str, values: &[f64]) -> Result<(), String> {
    for (index, value) in values.iter().copied().enumerate() {
        if value <= 0.0 || (value - value.round()).abs() > 0.0 {
            return Err(format!(
                "{name}[{index}] must be a positive integer count, got {value}"
            ));
        }
    }
    Ok(())
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
#[expect(
    clippy::too_many_arguments,
    reason = "public PyO3 ABI mirrors the finite-shot parameter-shift evidence schema"
)]
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
) -> StochasticGradientPyResult<'py> {
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

#[pyfunction]
#[pyo3(signature = (
    plus_values,
    minus_values,
    perturbations,
    plus_variances,
    minus_variances,
    plus_shots,
    minus_shots,
    trainable,
    perturbation_radius,
    confidence_z=1.959963984540054
))]
#[expect(
    clippy::too_many_arguments,
    reason = "public PyO3 ABI mirrors the SPSA evidence schema"
)]
pub fn spsa_gradient_rust<'py>(
    py: Python<'py>,
    plus_values: PyReadonlyArray1<'_, f64>,
    minus_values: PyReadonlyArray1<'_, f64>,
    perturbations: PyReadonlyArray2<'_, f64>,
    plus_variances: PyReadonlyArray1<'_, f64>,
    minus_variances: PyReadonlyArray1<'_, f64>,
    plus_shots: PyReadonlyArray1<'_, f64>,
    minus_shots: PyReadonlyArray1<'_, f64>,
    trainable: PyReadonlyArray1<'_, bool>,
    perturbation_radius: f64,
    confidence_z: f64,
) -> StochasticGradientPyResult<'py> {
    let perturbations = read_array2_rows(perturbations);
    let result = spsa_gradient_inner(
        plus_values.as_slice()?,
        minus_values.as_slice()?,
        &perturbations,
        Some(plus_variances.as_slice()?),
        Some(minus_variances.as_slice()?),
        Some(plus_shots.as_slice()?),
        Some(minus_shots.as_slice()?),
        trainable.as_slice()?,
        perturbation_radius,
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

#[pyfunction]
#[pyo3(signature = (
    rewards,
    score_vectors,
    trainable,
    baseline=0.0,
    confidence_z=1.959963984540054
))]
pub fn score_function_gradient_rust<'py>(
    py: Python<'py>,
    rewards: PyReadonlyArray1<'_, f64>,
    score_vectors: PyReadonlyArray2<'_, f64>,
    trainable: PyReadonlyArray1<'_, bool>,
    baseline: f64,
    confidence_z: f64,
) -> StochasticGradientPyResult<'py> {
    let score_vectors = read_array2_rows(score_vectors);
    let result = score_function_gradient_inner(
        rewards.as_slice()?,
        &score_vectors,
        trainable.as_slice()?,
        baseline,
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

#[pyfunction]
#[pyo3(signature = (
    gradient,
    standard_error,
    trainable,
    confidence_z=1.959963984540054,
    max_standard_error=None,
    max_confidence_radius=None
))]
pub fn gradient_confidence_interval_rust<'py>(
    py: Python<'py>,
    gradient: PyReadonlyArray1<'_, f64>,
    standard_error: PyReadonlyArray1<'_, f64>,
    trainable: PyReadonlyArray1<'_, bool>,
    confidence_z: f64,
    max_standard_error: Option<f64>,
    max_confidence_radius: Option<f64>,
) -> GradientConfidenceIntervalPyResult<'py> {
    let result = gradient_confidence_interval_inner(
        gradient.as_slice()?,
        standard_error.as_slice()?,
        trainable.as_slice()?,
        confidence_z,
        max_standard_error,
        max_confidence_radius,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        PyArray1::from_vec(py, result.lower),
        PyArray1::from_vec(py, result.upper),
        result.status,
        result.failure_reasons,
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

    #[test]
    fn spsa_gradient_kernel_averages_perturbation_records() {
        let plus_values = [1.0, 0.25];
        let minus_values = [0.0, -0.75];
        let perturbations = [vec![1.0, 0.0], vec![-1.0, 0.0]];
        let plus_variances = [0.04, 0.09];
        let minus_variances = [0.04, 0.09];
        let plus_shots = [400.0, 500.0];
        let minus_shots = [400.0, 500.0];

        let result = spsa_gradient_inner(
            &plus_values,
            &minus_values,
            &perturbations,
            Some(&plus_variances),
            Some(&minus_variances),
            Some(&plus_shots),
            Some(&minus_shots),
            &[true, false],
            0.5,
            2.0,
        )
        .unwrap();

        assert_close(result.gradient[0], 0.0);
        assert_close(result.gradient[1], 0.0);
        assert!(result.standard_error[0] > 0.0);
        assert_close(result.standard_error[1], 0.0);
        assert_close(result.confidence_radius[0], 2.0 * result.standard_error[0]);
    }

    #[test]
    fn spsa_gradient_kernel_rejects_invalid_contracts() {
        let plus_values = [1.0];
        let minus_values = [0.0];
        let valid_perturbations = [vec![1.0]];
        let variances = [0.04];
        let shots = [400.0];

        assert!(spsa_gradient_inner(
            &plus_values,
            &minus_values,
            &[vec![0.0]],
            Some(&variances),
            Some(&variances),
            Some(&shots),
            Some(&shots),
            &[true],
            0.5,
            2.0,
        )
        .is_err());
        assert!(spsa_gradient_inner(
            &plus_values,
            &minus_values,
            &valid_perturbations,
            Some(&variances),
            None,
            Some(&shots),
            Some(&shots),
            &[true],
            0.5,
            2.0,
        )
        .is_err());
        assert!(spsa_gradient_inner(
            &plus_values,
            &minus_values,
            &valid_perturbations,
            Some(&[-0.01]),
            Some(&variances),
            Some(&shots),
            Some(&shots),
            &[true],
            0.5,
            2.0,
        )
        .is_err());
    }

    #[test]
    fn score_function_gradient_kernel_matches_likelihood_ratio_reference() {
        let rewards = [2.0, 0.0, 4.0];
        let score_vectors = [vec![1.0, 2.0], vec![-1.0, 0.0], vec![0.0, 1.0]];

        let result =
            score_function_gradient_inner(&rewards, &score_vectors, &[true, true], 1.0, 2.0)
                .unwrap();

        assert_close(result.gradient[0], 2.0 / 3.0);
        assert_close(result.gradient[1], 5.0 / 3.0);
        assert!(result.standard_error[0] > 0.0);
        assert!(result.standard_error[1] > 0.0);
        assert_close(result.confidence_radius[0], 2.0 * result.standard_error[0]);
    }

    #[test]
    fn score_function_gradient_kernel_rejects_invalid_contracts() {
        let rewards = [1.0, 2.0];
        let score_vectors = [vec![0.5], vec![0.25]];

        assert!(score_function_gradient_inner(&[1.0], &[vec![0.5]], &[true], 0.0, 2.0).is_err());
        assert!(score_function_gradient_inner(&rewards, &[vec![0.5]], &[true], 0.0, 2.0).is_err());
        assert!(
            score_function_gradient_inner(&rewards, &score_vectors, &[false], 0.0, 2.0).is_err()
        );
        assert!(
            score_function_gradient_inner(&rewards, &score_vectors, &[true], f64::NAN, 2.0)
                .is_err()
        );
    }

    #[test]
    fn gradient_confidence_interval_kernel_applies_failure_policy() {
        let result = gradient_confidence_interval_inner(
            &[1.0, -2.0],
            &[0.2, 0.0],
            &[true, false],
            2.0,
            Some(0.1),
            None,
        )
        .unwrap();

        assert_close(result.lower[0], 0.6);
        assert_close(result.upper[0], 1.4);
        assert_close(result.lower[1], -2.0);
        assert_close(result.upper[1], -2.0);
        assert_eq!(result.status, "failed");
        assert_eq!(result.failure_reasons.len(), 1);
    }
}
