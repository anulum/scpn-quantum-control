// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD numeric linalg evaluation

fn operand_scalar_value(
    name: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<f64, String> {
    numeric_operand(name, values)?.scalar_value()
}

fn numeric_multi_dot(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    multi_dot_output_value(effect.index, operation, &input_values)
        .map(ProgramADNumericValue::scalar)
}

fn numeric_matrix_power(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    matrix_power_output_value(effect.index, operation, &input_values)
        .map(ProgramADNumericValue::scalar)
}

fn numeric_eigvalsh(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    eigvalsh_output_value(effect.index, operation, &input_values).map(ProgramADNumericValue::scalar)
}

fn numeric_eigvals(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    eigvals_output_value(effect.index, operation, &input_values).map(ProgramADNumericValue::scalar)
}

fn numeric_eig(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    eig_output_value(effect.index, operation, &input_values).map(ProgramADNumericValue::scalar)
}

fn numeric_eigh(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    eigh_output_value(effect.index, operation, &input_values).map(ProgramADNumericValue::scalar)
}

fn numeric_svdvals(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    svdvals_output_value(effect.index, operation, &input_values).map(ProgramADNumericValue::scalar)
}

fn numeric_diagflat(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    diagflat_output_value(effect.index, operation, &input_values).map(ProgramADNumericValue::scalar)
}

fn numeric_diag(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    diag_output_value(effect.index, operation, &input_values).map(ProgramADNumericValue::scalar)
}

fn numeric_pinv(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    pinv_output_value(effect.index, operation, &input_values).map(ProgramADNumericValue::scalar)
}

fn numeric_stencil(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    stencil_output_value(effect.index, operation, &input_values).map(ProgramADNumericValue::scalar)
}

fn numeric_unary(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    function: fn(f64) -> f64,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    ProgramADNumericValue::new(
        input.shape,
        input.values.into_iter().map(function).collect::<Vec<f64>>(),
    )
}

fn numeric_unary_checked(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    function: fn(f64) -> f64,
    finite_error: &str,
) -> Result<ProgramADNumericValue, String> {
    let value = numeric_unary(effect, values, function)?;
    if value.values.iter().all(|item| item.is_finite()) {
        Ok(value)
    } else {
        Err(finite_error.to_owned())
    }
}

fn numeric_unary_domain(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    predicate: fn(f64) -> bool,
    function: fn(f64) -> f64,
    domain_error: &str,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    if input.values.iter().any(|value| !predicate(*value)) {
        return Err(domain_error.to_owned());
    }
    ProgramADNumericValue::new(
        input.shape,
        input.values.into_iter().map(function).collect::<Vec<f64>>(),
    )
}
