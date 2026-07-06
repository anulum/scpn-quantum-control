// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Phase-QNode metric and transform kernels

use ndarray::{Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug, Clone, PartialEq)]
pub struct FubiniStudyMetricResult {
    pub fubini_study_metric: Vec<Vec<f64>>,
    pub quantum_fisher_information: Vec<Vec<f64>>,
    pub derivative_norms: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComputationalBasisFisherResult {
    pub classical_fisher_information: Vec<Vec<f64>>,
    pub probabilities: Vec<f64>,
    pub probability_derivatives: Vec<Vec<f64>>,
}

type FubiniStudyMetricPyResult<'py> = PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)>;

type ComputationalBasisFisherPyResult<'py> = PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
)>;

fn ensure_finite_slice(name: &str, values: &[f64]) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!("{name}[{index}] must be finite, got {value}"));
    }
    Ok(())
}

fn validate_state_and_derivatives<R, I>(
    state_re: &[f64],
    state_im: &[f64],
    derivatives_re: &[R],
    derivatives_im: &[I],
) -> Result<(), String>
where
    R: AsRef<[f64]>,
    I: AsRef<[f64]>,
{
    if state_re.is_empty() {
        return Err("state vector must be non-empty".to_string());
    }
    if state_re.len() != state_im.len() {
        return Err(format!(
            "state real/imaginary lengths must match: {} != {}",
            state_re.len(),
            state_im.len()
        ));
    }
    if derivatives_re.is_empty() {
        return Err("at least one parameter derivative row is required".to_string());
    }
    if derivatives_re.len() != derivatives_im.len() {
        return Err(format!(
            "derivative real/imaginary row counts must match: {} != {}",
            derivatives_re.len(),
            derivatives_im.len()
        ));
    }
    ensure_finite_slice("state_re", state_re)?;
    ensure_finite_slice("state_im", state_im)?;
    let norm_sq: f64 = state_re
        .iter()
        .zip(state_im.iter())
        .map(|(re, im)| re.mul_add(*re, im * im))
        .sum();
    if norm_sq <= 0.0 {
        return Err("state vector norm must be positive".to_string());
    }
    for (row_index, (row_re, row_im)) in
        derivatives_re.iter().zip(derivatives_im.iter()).enumerate()
    {
        let row_re = row_re.as_ref();
        let row_im = row_im.as_ref();
        if row_re.len() != state_re.len() || row_im.len() != state_re.len() {
            return Err(format!(
                "derivative row {row_index} must have state width {}; got real {} imaginary {}",
                state_re.len(),
                row_re.len(),
                row_im.len()
            ));
        }
        ensure_finite_slice(&format!("derivatives_re[{row_index}]"), row_re)?;
        ensure_finite_slice(&format!("derivatives_im[{row_index}]"), row_im)?;
    }
    Ok(())
}

fn complex_inner_product(
    left_re: &[f64],
    left_im: &[f64],
    right_re: &[f64],
    right_im: &[f64],
) -> (f64, f64) {
    let mut real = 0.0;
    let mut imag = 0.0;
    for (((left_re, left_im), right_re), right_im) in left_re
        .iter()
        .zip(left_im.iter())
        .zip(right_re.iter())
        .zip(right_im.iter())
    {
        real += left_re * right_re + left_im * right_im;
        imag += left_re * right_im - left_im * right_re;
    }
    (real, imag)
}

pub fn fubini_study_metric_inner<R, I>(
    state_re: &[f64],
    state_im: &[f64],
    derivatives_re: &[R],
    derivatives_im: &[I],
) -> Result<FubiniStudyMetricResult, String>
where
    R: AsRef<[f64]>,
    I: AsRef<[f64]>,
{
    validate_state_and_derivatives(state_re, state_im, derivatives_re, derivatives_im)?;
    let width = derivatives_re.len();
    let mut metric = vec![vec![0.0; width]; width];
    let mut qfi = vec![vec![0.0; width]; width];
    let mut derivative_norms = vec![0.0; width];
    let overlaps: Vec<(f64, f64)> = derivatives_re
        .iter()
        .zip(derivatives_im.iter())
        .map(|(row_re, row_im)| {
            complex_inner_product(state_re, state_im, row_re.as_ref(), row_im.as_ref())
        })
        .collect();

    for row in 0..width {
        let row_re = derivatives_re[row].as_ref();
        let row_im = derivatives_im[row].as_ref();
        let (norm_real, _) = complex_inner_product(row_re, row_im, row_re, row_im);
        derivative_norms[row] = norm_real.max(0.0).sqrt();
        for column in row..width {
            let column_re = derivatives_re[column].as_ref();
            let column_im = derivatives_im[column].as_ref();
            let (raw_real, _) = complex_inner_product(row_re, row_im, column_re, column_im);
            let (overlap_row_real, overlap_row_imag) = overlaps[row];
            let (overlap_column_real, overlap_column_imag) = overlaps[column];
            let projected_real = overlap_row_real
                .mul_add(overlap_column_real, overlap_row_imag * overlap_column_imag);
            let value = raw_real - projected_real;
            metric[row][column] = value;
            metric[column][row] = value;
            qfi[row][column] = 4.0 * value;
            qfi[column][row] = 4.0 * value;
        }
    }
    Ok(FubiniStudyMetricResult {
        fubini_study_metric: metric,
        quantum_fisher_information: qfi,
        derivative_norms,
    })
}

pub fn computational_basis_fisher_inner<R, I>(
    state_re: &[f64],
    state_im: &[f64],
    derivatives_re: &[R],
    derivatives_im: &[I],
    min_probability: f64,
) -> Result<ComputationalBasisFisherResult, String>
where
    R: AsRef<[f64]>,
    I: AsRef<[f64]>,
{
    validate_state_and_derivatives(state_re, state_im, derivatives_re, derivatives_im)?;
    if !min_probability.is_finite() || min_probability < 0.0 {
        return Err(format!(
            "min_probability must be a finite non-negative value, got {min_probability}"
        ));
    }
    let probabilities: Vec<f64> = state_re
        .iter()
        .zip(state_im.iter())
        .map(|(re, im)| re.mul_add(*re, im * im))
        .collect();
    if probabilities
        .iter()
        .any(|probability| *probability <= min_probability)
    {
        return Err(
            "computational-basis Fisher information is singular at a zero-probability outcome"
                .to_string(),
        );
    }
    let parameter_count = derivatives_re.len();
    let state_width = state_re.len();
    let mut probability_derivatives = vec![vec![0.0; state_width]; parameter_count];
    for param in 0..parameter_count {
        let row_re = derivatives_re[param].as_ref();
        let row_im = derivatives_im[param].as_ref();
        for basis in 0..state_width {
            probability_derivatives[param][basis] =
                2.0 * (state_re[basis] * row_re[basis] + state_im[basis] * row_im[basis]);
        }
    }
    let mut fisher = vec![vec![0.0; parameter_count]; parameter_count];
    for row in 0..parameter_count {
        for column in row..parameter_count {
            let mut value = 0.0;
            for (basis, probability) in probabilities.iter().copied().enumerate() {
                value += probability_derivatives[row][basis]
                    * probability_derivatives[column][basis]
                    / probability;
            }
            fisher[row][column] = value;
            fisher[column][row] = value;
        }
    }
    Ok(ComputationalBasisFisherResult {
        classical_fisher_information: fisher,
        probabilities,
        probability_derivatives,
    })
}

fn validate_matrix<R>(name: &str, matrix: &[R]) -> Result<(usize, usize), String>
where
    R: AsRef<[f64]>,
{
    if matrix.is_empty() {
        return Err(format!("{name} must have at least one row"));
    }
    let columns = matrix[0].as_ref().len();
    if columns == 0 {
        return Err(format!("{name} must have at least one column"));
    }
    for (row_index, row) in matrix.iter().enumerate() {
        let row = row.as_ref();
        if row.len() != columns {
            return Err(format!(
                "{name} rows must share width {columns}; row {row_index} has width {}",
                row.len()
            ));
        }
        ensure_finite_slice(&format!("{name}[{row_index}]"), row)?;
    }
    Ok((matrix.len(), columns))
}

pub fn vector_jvp_inner<R>(jacobian: &[R], tangent: &[f64]) -> Result<Vec<f64>, String>
where
    R: AsRef<[f64]>,
{
    let (_, columns) = validate_matrix("jacobian", jacobian)?;
    if tangent.len() != columns {
        return Err(format!(
            "tangent length must match Jacobian column count: {} != {columns}",
            tangent.len()
        ));
    }
    ensure_finite_slice("tangent", tangent)?;
    Ok(jacobian
        .iter()
        .map(|row| {
            row.as_ref()
                .iter()
                .zip(tangent.iter())
                .map(|(jacobian_value, tangent_value)| jacobian_value * tangent_value)
                .sum()
        })
        .collect())
}

pub fn vector_vjp_inner<R>(jacobian: &[R], cotangent: &[f64]) -> Result<Vec<f64>, String>
where
    R: AsRef<[f64]>,
{
    let (rows, columns) = validate_matrix("jacobian", jacobian)?;
    if cotangent.len() != rows {
        return Err(format!(
            "cotangent length must match Jacobian row count: {} != {rows}",
            cotangent.len()
        ));
    }
    ensure_finite_slice("cotangent", cotangent)?;
    let mut result = vec![0.0; columns];
    for (row_index, row) in jacobian.iter().enumerate() {
        for (column, value) in row.as_ref().iter().enumerate() {
            result[column] += value * cotangent[row_index];
        }
    }
    Ok(result)
}

pub fn hessian_vector_product_inner<R>(hessian: &[R], vector: &[f64]) -> Result<Vec<f64>, String>
where
    R: AsRef<[f64]>,
{
    let (rows, columns) = validate_matrix("hessian", hessian)?;
    if rows != columns {
        return Err(format!("hessian must be square, got {rows}x{columns}"));
    }
    if vector.len() != columns {
        return Err(format!(
            "vector length must match Hessian width: {} != {columns}",
            vector.len()
        ));
    }
    ensure_finite_slice("vector", vector)?;
    Ok(hessian
        .iter()
        .map(|row| {
            row.as_ref()
                .iter()
                .zip(vector.iter())
                .map(|(hessian_value, vector_value)| hessian_value * vector_value)
                .sum()
        })
        .collect())
}

pub fn vector_hessian_tensor_inner(
    tensor: &[Vec<Vec<f64>>],
    symmetry_tolerance: f64,
) -> Result<Vec<Vec<Vec<f64>>>, String> {
    if tensor.is_empty() {
        return Err("hessian tensor must have at least one output component".to_string());
    }
    if !symmetry_tolerance.is_finite() || symmetry_tolerance < 0.0 {
        return Err(format!(
            "symmetry_tolerance must be a finite non-negative value, got {symmetry_tolerance}"
        ));
    }
    let parameter_count = tensor[0].len();
    if parameter_count == 0 {
        return Err("hessian tensor must have a non-empty parameter axis".to_string());
    }
    let mut result = vec![vec![vec![0.0; parameter_count]; parameter_count]; tensor.len()];
    for (component_index, component) in tensor.iter().enumerate() {
        if component.len() != parameter_count {
            return Err(format!(
                "hessian tensor component {component_index} has row count {}, expected {parameter_count}",
                component.len()
            ));
        }
        for (row_index, row) in component.iter().enumerate() {
            if row.len() != parameter_count {
                return Err(format!(
                    "hessian tensor component {component_index} row {row_index} has width {}, expected {parameter_count}",
                    row.len()
                ));
            }
            ensure_finite_slice(
                &format!("hessian_tensor[{component_index}][{row_index}]"),
                row,
            )?;
        }
        for row in 0..parameter_count {
            for column in row..parameter_count {
                let forward = component[row][column];
                let reverse = component[column][row];
                if (forward - reverse).abs() > symmetry_tolerance {
                    return Err(format!(
                        "hessian tensor component {component_index} is not symmetric at ({row}, {column}): {forward} != {reverse}"
                    ));
                }
                let value = 0.5 * (forward + reverse);
                result[component_index][row][column] = value;
                result[component_index][column][row] = value;
            }
        }
    }
    Ok(result)
}

fn nested_to_array2(values: Vec<Vec<f64>>) -> Result<Array2<f64>, String> {
    let rows = values.len();
    let columns = values.first().map_or(0, Vec::len);
    let flat: Vec<f64> = values.into_iter().flatten().collect();
    Array2::from_shape_vec((rows, columns), flat).map_err(|err| err.to_string())
}

fn nested_to_array3(values: Vec<Vec<Vec<f64>>>) -> Result<Array3<f64>, String> {
    let components = values.len();
    let rows = values.first().map_or(0, Vec::len);
    let columns = values
        .first()
        .and_then(|component| component.first())
        .map_or(0, Vec::len);
    let flat: Vec<f64> = values.into_iter().flatten().flatten().collect();
    Array3::from_shape_vec((components, rows, columns), flat).map_err(|err| err.to_string())
}

fn read_array2_rows(values: PyReadonlyArray2<'_, f64>) -> Vec<Vec<f64>> {
    values
        .as_array()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect()
}

fn read_array3_components(values: PyReadonlyArray3<'_, f64>) -> Vec<Vec<Vec<f64>>> {
    let array = values.as_array();
    let shape = array.shape();
    let output_dim = shape[0];
    let rows = shape[1];
    let columns = shape[2];
    let mut tensor = vec![vec![vec![0.0; columns]; rows]; output_dim];
    for output in 0..output_dim {
        for row in 0..rows {
            for column in 0..columns {
                tensor[output][row][column] = array[[output, row, column]];
            }
        }
    }
    tensor
}

#[pyfunction]
pub fn phase_qnode_fubini_study_metric_rust<'py>(
    py: Python<'py>,
    state_re: PyReadonlyArray1<'_, f64>,
    state_im: PyReadonlyArray1<'_, f64>,
    derivatives_re: PyReadonlyArray2<'_, f64>,
    derivatives_im: PyReadonlyArray2<'_, f64>,
) -> FubiniStudyMetricPyResult<'py> {
    let state_re = state_re.as_slice()?;
    let state_im = state_im.as_slice()?;
    let derivatives_re = read_array2_rows(derivatives_re);
    let derivatives_im = read_array2_rows(derivatives_im);
    let result = fubini_study_metric_inner(state_re, state_im, &derivatives_re, &derivatives_im)
        .map_err(PyValueError::new_err)?;
    let metric = nested_to_array2(result.fubini_study_metric).map_err(PyValueError::new_err)?;
    let qfi = nested_to_array2(result.quantum_fisher_information).map_err(PyValueError::new_err)?;
    Ok((
        PyArray2::from_owned_array(py, metric),
        PyArray2::from_owned_array(py, qfi),
        PyArray1::from_vec(py, result.derivative_norms),
    ))
}

#[pyfunction]
pub fn phase_qnode_computational_basis_fisher_rust<'py>(
    py: Python<'py>,
    state_re: PyReadonlyArray1<'_, f64>,
    state_im: PyReadonlyArray1<'_, f64>,
    derivatives_re: PyReadonlyArray2<'_, f64>,
    derivatives_im: PyReadonlyArray2<'_, f64>,
    min_probability: f64,
) -> ComputationalBasisFisherPyResult<'py> {
    let state_re = state_re.as_slice()?;
    let state_im = state_im.as_slice()?;
    let derivatives_re = read_array2_rows(derivatives_re);
    let derivatives_im = read_array2_rows(derivatives_im);
    let result = computational_basis_fisher_inner(
        state_re,
        state_im,
        &derivatives_re,
        &derivatives_im,
        min_probability,
    )
    .map_err(PyValueError::new_err)?;
    let fisher =
        nested_to_array2(result.classical_fisher_information).map_err(PyValueError::new_err)?;
    let probability_derivatives =
        nested_to_array2(result.probability_derivatives).map_err(PyValueError::new_err)?;
    Ok((
        PyArray2::from_owned_array(py, fisher),
        PyArray1::from_vec(py, result.probabilities),
        PyArray2::from_owned_array(py, probability_derivatives),
    ))
}

#[pyfunction]
pub fn phase_qnode_vector_jvp_rust<'py>(
    py: Python<'py>,
    jacobian: PyReadonlyArray2<'_, f64>,
    tangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let jacobian = read_array2_rows(jacobian);
    let result = vector_jvp_inner(&jacobian, tangent.as_slice()?).map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
pub fn phase_qnode_vector_vjp_rust<'py>(
    py: Python<'py>,
    jacobian: PyReadonlyArray2<'_, f64>,
    cotangent: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let jacobian = read_array2_rows(jacobian);
    let result =
        vector_vjp_inner(&jacobian, cotangent.as_slice()?).map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
pub fn phase_qnode_hessian_vector_product_rust<'py>(
    py: Python<'py>,
    hessian: PyReadonlyArray2<'_, f64>,
    vector: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let hessian = read_array2_rows(hessian);
    let result = hessian_vector_product_inner(&hessian, vector.as_slice()?)
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, result))
}

#[pyfunction]
#[pyo3(signature = (hessian_tensor, symmetry_tolerance=1e-12))]
pub fn phase_qnode_vector_hessian_tensor_rust<'py>(
    py: Python<'py>,
    hessian_tensor: PyReadonlyArray3<'_, f64>,
    symmetry_tolerance: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let tensor = read_array3_components(hessian_tensor);
    let result =
        vector_hessian_tensor_inner(&tensor, symmetry_tolerance).map_err(PyValueError::new_err)?;
    let array = nested_to_array3(result).map_err(PyValueError::new_err)?;
    Ok(PyArray3::from_owned_array(py, array))
}

#[pyfunction]
pub fn phase_qnode_complex_derivative_contract_rust<'py>(
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let contract = PyDict::new(py);
    contract.set_item("parameter_domain", "real")?;
    contract.set_item("accepts_complex_parameters", false)?;
    contract.set_item("accepts_complex_tangents", false)?;
    contract.set_item("holomorphic_derivatives", false)?;
    contract.set_item("wirtinger_partials", false)?;
    contract.set_item("complex_state_amplitudes", "internal_statevector_only")?;
    contract.set_item(
        "claim_boundary",
        "real-valued parameter, tangent, cotangent, vector, and output arrays only",
    )?;
    Ok(contract)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-12, "{actual} != {expected}");
    }

    #[test]
    fn fubini_study_kernel_matches_single_ry_reference() {
        let theta = 0.31_f64;
        let state_re = [(theta / 2.0).cos(), (theta / 2.0).sin()];
        let state_im = [0.0, 0.0];
        let derivatives_re = [[-0.5 * (theta / 2.0).sin(), 0.5 * (theta / 2.0).cos()]];
        let derivatives_im = [[0.0, 0.0]];

        let result =
            fubini_study_metric_inner(&state_re, &state_im, &derivatives_re, &derivatives_im)
                .unwrap();

        assert_close(result.fubini_study_metric[0][0], 0.25);
        assert_close(result.quantum_fisher_information[0][0], 1.0);
        assert_close(result.derivative_norms[0], 0.5);
    }

    #[test]
    fn computational_basis_fisher_kernel_matches_single_ry_reference() {
        let theta = 0.31_f64;
        let state_re = [(theta / 2.0).cos(), (theta / 2.0).sin()];
        let state_im = [0.0, 0.0];
        let derivatives_re = [[-0.5 * (theta / 2.0).sin(), 0.5 * (theta / 2.0).cos()]];
        let derivatives_im = [[0.0, 0.0]];

        let result = computational_basis_fisher_inner(
            &state_re,
            &state_im,
            &derivatives_re,
            &derivatives_im,
            1e-15,
        )
        .unwrap();

        assert_close(result.classical_fisher_information[0][0], 1.0);
        assert_close(result.probabilities[0], state_re[0] * state_re[0]);
        assert_close(result.probability_derivatives[0][0], -0.5 * theta.sin());
    }

    #[test]
    fn directional_kernels_match_dense_linear_algebra() {
        let jacobian = [[1.0, -2.0, 0.5], [3.0, 4.0, -1.0]];
        let tangent = [0.25, -0.5, 2.0];
        let cotangent = [1.5, -0.25];
        let hessian = [[2.0, -1.0], [-1.0, 3.5]];
        let vector = [0.75, -2.0];

        assert_eq!(
            vector_jvp_inner(&jacobian, &tangent).unwrap(),
            vec![2.25, -3.25]
        );
        assert_eq!(
            vector_vjp_inner(&jacobian, &cotangent).unwrap(),
            vec![0.75, -4.0, 1.0]
        );
        assert_eq!(
            hessian_vector_product_inner(&hessian, &vector).unwrap(),
            vec![3.5, -7.75]
        );
    }

    #[test]
    fn vector_hessian_tensor_kernel_validates_and_symmetrizes_component_matrices() {
        let tensor = vec![
            vec![vec![1.0, 2.0], vec![2.0000000000000004, 3.0]],
            vec![vec![-0.5, 0.25], vec![0.2499999999999998, 0.75]],
        ];

        let result = vector_hessian_tensor_inner(&tensor, 1e-12).unwrap();

        assert_close(result[0][0][1], 2.0);
        assert_close(result[0][1][0], 2.0);
        assert_close(result[1][0][1], 0.25);
        assert_close(result[1][1][0], 0.25);
    }

    #[test]
    fn singular_classical_fisher_fails_closed() {
        let state_re = [1.0, 0.0];
        let state_im = [0.0, 0.0];
        let derivatives_re = [[0.0, 0.5]];
        let derivatives_im = [[0.0, 0.0]];

        let err = computational_basis_fisher_inner(
            &state_re,
            &state_im,
            &derivatives_re,
            &derivatives_im,
            1e-15,
        )
        .unwrap_err();

        assert!(err.contains("singular"));
    }

    #[test]
    fn complex_derivative_contract_is_real_only() {
        Python::initialize();
        Python::attach(|py| {
            let contract = phase_qnode_complex_derivative_contract_rust(py).unwrap();
            assert_eq!(
                contract
                    .get_item("parameter_domain")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "real"
            );
            assert!(!contract
                .get_item("accepts_complex_parameters")
                .unwrap()
                .unwrap()
                .extract::<bool>()
                .unwrap());
        });
    }
}
