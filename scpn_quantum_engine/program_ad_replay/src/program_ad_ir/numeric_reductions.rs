// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD numeric reduction evaluation

fn numeric_sum(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} sum requires one input", effect.index));
    }
    let target = target_shape(effect, shapes_by_target)?;
    let source = numeric_operand(&effect.inputs[0], values)?;
    if operation == "sum" {
        if !target.is_empty() {
            return Err(format!(
                "effect {} sum non-scalar target requires static axis metadata sum:axis:<int>",
                effect.index
            ));
        }
        return Ok(ProgramADNumericValue::scalar(source.values.iter().sum()));
    }
    reduce_axis_values(effect.index, operation, "sum", &source, &target, 1.0)
}

fn numeric_mean(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} mean requires one input", effect.index));
    }
    let target = target_shape(effect, shapes_by_target)?;
    let source = numeric_operand(&effect.inputs[0], values)?;
    if operation == "mean" {
        if !target.is_empty() {
            return Err(format!(
                "effect {} mean non-scalar target requires static axis metadata mean:axis:<int>",
                effect.index
            ));
        }
        let total: f64 = source.values.iter().sum();
        return Ok(ProgramADNumericValue::scalar(
            total / source.values.len() as f64,
        ));
    }
    let axis = parse_static_axis(operation, "mean", source.shape.len())?;
    let scale = 1.0 / source.shape[axis] as f64;
    reduce_axis_values(effect.index, operation, "mean", &source, &target, scale)
}

fn numeric_prod(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} prod requires one input", effect.index));
    }
    let target = target_shape(effect, shapes_by_target)?;
    let source = numeric_operand(&effect.inputs[0], values)?;
    if operation == "prod" {
        if !target.is_empty() {
            return Err(format!(
                "effect {} prod non-scalar target requires static axis metadata prod:axis:<int>",
                effect.index
            ));
        }
        return Ok(ProgramADNumericValue::scalar(product_all_value(
            effect.index,
            &source.values,
        )?));
    }
    let axis = parse_static_axis(operation, "prod", source.shape.len())?;
    let output = product_axis_values(effect.index, &source.shape, axis, &target, &source.values)?;
    ProgramADNumericValue::new(target, output)
}

fn numeric_variance(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} var requires one input", effect.index));
    }
    let target = target_shape(effect, shapes_by_target)?;
    let source = numeric_operand(&effect.inputs[0], values)?;
    let metadata = parse_moment_reduction_metadata(operation, "var", source.shape.len())?;
    match metadata.axis {
        Some(axis) => {
            let output = variance_axis_values(
                effect.index,
                &source.shape,
                axis,
                &target,
                &source.values,
                metadata.correction,
            )?;
            ProgramADNumericValue::new(target, output)
        }
        None => {
            if !target.is_empty() {
                return Err(format!(
                    "effect {} var non-scalar target requires static axis metadata var:axis:<int>",
                    effect.index
                ));
            }
            Ok(ProgramADNumericValue::scalar(variance_all_value(
                effect.index,
                &source.values,
                metadata.correction,
            )?))
        }
    }
}

fn numeric_standard_deviation(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} std requires one input", effect.index));
    }
    let target = target_shape(effect, shapes_by_target)?;
    let source = numeric_operand(&effect.inputs[0], values)?;
    let metadata = parse_moment_reduction_metadata(operation, "std", source.shape.len())?;
    match metadata.axis {
        Some(axis) => {
            let output = std_axis_values(
                effect.index,
                &source.shape,
                axis,
                &target,
                &source.values,
                metadata.correction,
            )?;
            ProgramADNumericValue::new(target, output)
        }
        None => {
            if !target.is_empty() {
                return Err(format!(
                    "effect {} std non-scalar target requires static axis metadata std:axis:<int>",
                    effect.index
                ));
            }
            Ok(ProgramADNumericValue::scalar(std_all_value(
                effect.index,
                &source.values,
                metadata.correction,
            )?))
        }
    }
}

fn numeric_order_statistic(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} {operation} requires one input",
            effect.index
        ));
    }
    let target = target_shape(effect, shapes_by_target)?;
    let source = numeric_operand(&effect.inputs[0], values)?;
    let output = order_statistic_values(
        effect.index,
        operation,
        &source.shape,
        &target,
        &source.values,
    )?;
    ProgramADNumericValue::new(target, output)
}

fn numeric_trapezoid(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} trapezoid requires one input",
            effect.index
        ));
    }
    let target = target_shape(effect, shapes_by_target)?;
    let source = numeric_operand(&effect.inputs[0], values)?;
    let output = trapezoid_values(
        effect.index,
        operation,
        &source.shape,
        &target,
        &source.values,
    )?;
    ProgramADNumericValue::new(target, output)
}

fn numeric_cumulative(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    Ok(ProgramADNumericValue::scalar(cumulative_output_value(
        effect.index,
        operation,
        &input_values,
    )?))
}

fn numeric_interpolation(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    Ok(ProgramADNumericValue::scalar(interpolation_output_value(
        effect.index,
        operation,
        &input_values,
    )?))
}

fn numeric_signal(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    Ok(ProgramADNumericValue::scalar(signal_output_value(
        effect.index,
        operation,
        &input_values,
    )?))
}
