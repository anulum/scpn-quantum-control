// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD PyO3 bindings

/// PyO3 wrapper returning a JSON metadata summary for a Program AD IR payload.
#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn program_ad_effect_ir_metadata_summary(serialization: &str) -> PyResult<String> {
    let ir = parse_program_ad_effect_ir(serialization).map_err(PyValueError::new_err)?;
    serde_json::to_string(&ir.metadata_summary()).map_err(|error| {
        PyValueError::new_err(format!("failed to encode Program AD IR summary: {error}"))
    })
}

/// PyO3 wrapper returning JSON for bounded Rust scalar Program AD interpretation.
#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn program_ad_effect_ir_interpret_forward(
    serialization: &str,
    inputs: Vec<f64>,
) -> PyResult<String> {
    let result = interpret_program_ad_effect_ir_forward(serialization, &inputs)
        .map_err(PyValueError::new_err)?;
    serde_json::to_string(&result).map_err(|error| {
        PyValueError::new_err(format!(
            "failed to encode Program AD IR interpreter result: {error}"
        ))
    })
}

/// PyO3 wrapper returning JSON for bounded Rust scalar Program AD value+gradient replay.
#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn program_ad_effect_ir_interpret_value_and_gradient(
    serialization: &str,
    inputs: Vec<f64>,
) -> PyResult<String> {
    let result = interpret_program_ad_effect_ir_value_and_gradient(serialization, &inputs)
        .map_err(PyValueError::new_err)?;
    serde_json::to_string(&result).map_err(|error| {
        PyValueError::new_err(format!(
            "failed to encode Program AD IR value+gradient result: {error}"
        ))
    })
}
