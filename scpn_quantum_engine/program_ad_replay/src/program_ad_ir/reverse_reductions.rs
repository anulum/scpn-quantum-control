// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD reverse reduction accumulation

fn accumulate_sum(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} sum requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let contribution = if operation == "sum" {
        let scalar_cotangent = cotangent.scalar_value()?;
        ProgramADNumericValue::filled(&input.shape, scalar_cotangent)?
    } else {
        expand_axis_reduction_cotangent(effect.index, operation, "sum", &input, cotangent, 1.0)?
    };
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_mean(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} mean requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let contribution = if operation == "mean" {
        let scalar_cotangent = cotangent.scalar_value()?;
        let scale = scalar_cotangent / input.values.len() as f64;
        ProgramADNumericValue::filled(&input.shape, scale)?
    } else {
        let axis = parse_static_axis(operation, "mean", input.shape.len())?;
        let scale = 1.0 / input.shape[axis] as f64;
        expand_axis_reduction_cotangent(effect.index, operation, "mean", &input, cotangent, scale)?
    };
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_prod(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} prod requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let contribution_values = if operation == "prod" {
        let scalar_cotangent = cotangent.scalar_value()?;
        product_all_cotangent(effect.index, &input.values, scalar_cotangent)?
    } else {
        let axis = parse_static_axis(operation, "prod", input.shape.len())?;
        product_axis_cotangent(
            effect.index,
            &input.shape,
            axis,
            &cotangent.values,
            &input.values,
        )?
    };
    let contribution = ProgramADNumericValue::new(input.shape.clone(), contribution_values)?;
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_variance(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} var requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let metadata = parse_moment_reduction_metadata(operation, "var", input.shape.len())?;
    let contribution_values = if operation == "var" {
        let scalar_cotangent = cotangent.scalar_value()?;
        variance_all_cotangent(
            effect.index,
            &input.values,
            scalar_cotangent,
            metadata.correction,
        )?
    } else {
        match metadata.axis {
            Some(axis) => variance_axis_cotangent(
                effect.index,
                &input.shape,
                axis,
                &cotangent.values,
                &input.values,
                metadata.correction,
            )?,
            None => {
                let scalar_cotangent = cotangent.scalar_value()?;
                variance_all_cotangent(
                    effect.index,
                    &input.values,
                    scalar_cotangent,
                    metadata.correction,
                )?
            }
        }
    };
    let contribution = ProgramADNumericValue::new(input.shape.clone(), contribution_values)?;
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_standard_deviation(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} std requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let metadata = parse_moment_reduction_metadata(operation, "std", input.shape.len())?;
    let contribution_values = if operation == "std" {
        let scalar_cotangent = cotangent.scalar_value()?;
        std_all_cotangent(
            effect.index,
            &input.values,
            scalar_cotangent,
            metadata.correction,
        )?
    } else {
        match metadata.axis {
            Some(axis) => std_axis_cotangent(
                effect.index,
                &input.shape,
                axis,
                &cotangent.values,
                &input.values,
                metadata.correction,
            )?,
            None => {
                let scalar_cotangent = cotangent.scalar_value()?;
                std_all_cotangent(
                    effect.index,
                    &input.values,
                    scalar_cotangent,
                    metadata.correction,
                )?
            }
        }
    };
    let contribution = ProgramADNumericValue::new(input.shape.clone(), contribution_values)?;
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_order_statistic(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} {operation} requires one input",
            effect.index
        ));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let contribution_values = order_statistic_cotangent(
        effect.index,
        operation,
        &input.shape,
        &cotangent.values,
        &input.values,
    )?;
    let contribution = ProgramADNumericValue::new(input.shape.clone(), contribution_values)?;
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_trapezoid(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} trapezoid requires one input",
            effect.index
        ));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let contribution_values = trapezoid_cotangent(
        effect.index,
        operation,
        &input.shape,
        &cotangent.values,
        &input.values,
    )?;
    let contribution = ProgramADNumericValue::new(input.shape.clone(), contribution_values)?;
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_cumulative(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    let contributions =
        cumulative_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (input, contribution) in effect.inputs.iter().zip(contributions.iter()) {
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_interpolation(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    let contributions =
        interpolation_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (input, contribution) in effect.inputs.iter().zip(contributions.iter()) {
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_signal(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = effect
        .inputs
        .iter()
        .map(|input| operand_scalar_value(input, values))
        .collect::<Result<Vec<f64>, String>>()?;
    let contributions =
        signal_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (input, contribution) in effect.inputs.iter().zip(contributions.iter()) {
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}
