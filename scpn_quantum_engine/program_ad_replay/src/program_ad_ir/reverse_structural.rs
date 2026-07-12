// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD reverse structural accumulation

fn accumulate_reshape_like(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} reshape/ravel requires one input",
            effect.index
        ));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let reshaped = ProgramADNumericValue::new(input.shape.clone(), cotangent.values.clone())?;
    add_numeric_adjoint(&effect.inputs[0], reshaped, values, adjoints)
}

fn accumulate_broadcast_to(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} broadcast_to requires one input",
            effect.index
        ));
    }
    add_numeric_adjoint(&effect.inputs[0], cotangent.clone(), values, adjoints)
}

fn accumulate_transpose(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} transpose requires one input",
            effect.index
        ));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let contribution = transpose_reversed_axes(cotangent, &input.shape)?;
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_concatenate(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let operands = numeric_operands(effect, values)?;
    let contributions = split_concatenate_cotangent(effect.index, operation, &operands, cotangent)?;
    for (input, contribution) in effect.inputs.iter().zip(contributions) {
        add_numeric_adjoint(input, contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_stack(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let operands = numeric_operands(effect, values)?;
    let contributions = split_stack_cotangent(effect.index, operation, &operands, cotangent)?;
    for (input, contribution) in effect.inputs.iter().zip(contributions) {
        add_numeric_adjoint(input, contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_index_map(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} index_map requires one input",
            effect.index
        ));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let contribution_values = scatter_static_source_map_cotangent(
        effect.index,
        operation,
        input.values.len(),
        &cotangent.values,
    )?;
    let contribution = ProgramADNumericValue::new(input.shape.clone(), contribution_values)?;
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_add_sub(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
    lhs_sign: f64,
    rhs_sign: f64,
) -> Result<(), String> {
    if effect.inputs.len() != 2 {
        return Err(format!("effect {} requires two inputs", effect.index));
    }
    add_numeric_adjoint(
        &effect.inputs[0],
        scale_value(cotangent, lhs_sign)?,
        values,
        adjoints,
    )?;
    add_numeric_adjoint(
        &effect.inputs[1],
        scale_value(cotangent, rhs_sign)?,
        values,
        adjoints,
    )
}

fn accumulate_unary(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
    derivative: impl Fn(f64) -> f64,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let derivative_values = ProgramADNumericValue::new(
        input.shape.clone(),
        input
            .values
            .iter()
            .map(|value| derivative(*value))
            .collect(),
    )?;
    add_numeric_adjoint(
        &effect.inputs[0],
        elementwise_mul(cotangent, &derivative_values)?,
        values,
        adjoints,
    )
}

fn accumulate_unary_domain(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
    predicate: impl Fn(f64) -> bool,
    derivative: impl Fn(f64) -> f64,
    domain_error: &str,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    if input.values.iter().any(|value| !predicate(*value)) {
        return Err(domain_error.to_owned());
    }
    let derivative_values = ProgramADNumericValue::new(
        input.shape.clone(),
        input
            .values
            .iter()
            .map(|value| derivative(*value))
            .collect(),
    )?;
    add_numeric_adjoint(
        &effect.inputs[0],
        elementwise_mul(cotangent, &derivative_values)?,
        values,
        adjoints,
    )
}

fn add_scalar_adjoint(
    input: &str,
    contribution: f64,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
) -> Result<(), String> {
    add_numeric_adjoint(
        input,
        ProgramADNumericValue::scalar(contribution),
        values,
        adjoints,
    )
}

fn add_numeric_adjoint(
    input: &str,
    contribution: ProgramADNumericValue,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
) -> Result<(), String> {
    if contribution.values.iter().any(|value| !value.is_finite()) {
        return Err(format!("adjoint contribution for {input} must be finite"));
    }
    let Some(target) = values.get(input) else {
        return Ok(());
    };
    let reduced = reduce_to_shape(&contribution, &target.shape)?;
    let entry = adjoints.entry(input.to_owned()).or_insert_with(|| {
        ProgramADNumericValue::filled(&target.shape, 0.0)
            .expect("zero adjoint shape is already validated")
    });
    if entry.shape != reduced.shape {
        return Err(format!(
            "adjoint shape {:?} does not match contribution shape {:?}",
            entry.shape, reduced.shape
        ));
    }
    for (slot, value) in entry.values.iter_mut().zip(reduced.values.iter()) {
        *slot += value;
    }
    Ok(())
}
