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
    let reshaped = ProgramADNumericValue::new(
        copy_replay_buffer(&input.shape)?,
        copy_replay_buffer(&cotangent.values)?,
    )?;
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
    add_numeric_adjoint(&effect.inputs[0], cotangent.try_clone()?, values, adjoints)
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
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
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
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
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
    let contribution =
        ProgramADNumericValue::new(copy_replay_buffer(&input.shape)?, contribution_values)?;
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
    let mut derivative_buffer = reserve_replay_buffer(input.values.len())?;
    for (index, value) in input.values.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        derivative_buffer.push(derivative(*value));
    }
    let derivative_values = ProgramADNumericValue::new(
        copy_replay_buffer(&input.shape)?,
        derivative_buffer,
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
    for (index, value) in input.values.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        if !predicate(*value) {
            return Err(domain_error.to_owned());
        }
    }
    let mut derivative_buffer = reserve_replay_buffer(input.values.len())?;
    for (index, value) in input.values.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        derivative_buffer.push(derivative(*value));
    }
    let derivative_values = ProgramADNumericValue::new(
        copy_replay_buffer(&input.shape)?,
        derivative_buffer,
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
        ProgramADNumericValue::scalar(contribution)?,
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
    for (index, value) in contribution.values.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        if !value.is_finite() {
            return Err(format!("adjoint contribution for {input} must be finite"));
        }
    }
    let Some(target) = values.get(input) else {
        return Ok(());
    };
    let reduced = reduce_to_shape(&contribution, &target.shape)?;
    if !adjoints.contains_key(input) && adjoints.len() == adjoints.capacity() {
        return Err("Program AD adjoint map exceeds admitted target capacity".to_owned());
    }
    let entry = match adjoints.entry(copy_replay_symbol(input)?) {
        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert(ProgramADNumericValue::filled(&target.shape, 0.0)?)
        }
    };
    if entry.shape != reduced.shape {
        return Err(format!(
            "adjoint shape {:?} does not match contribution shape {:?}",
            entry.shape, reduced.shape
        ));
    }
    for (index, (slot, value)) in entry.values.iter_mut().zip(reduced.values.iter()).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        *slot += value;
    }
    Ok(())
}
