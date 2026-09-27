// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD numeric evaluation dispatch

fn evaluate_numeric_effect(
    effect: &ProgramADEffect,
    operation: &str,
    inputs: &[f64],
    input_index: &mut usize,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &ProgramADShapeMap<'_>,
) -> Result<ProgramADNumericValue, String> {
    if operation == "parameter" {
        if effect.kind != "parameter" {
            return Err(format!(
                "effect {} operation parameter must have kind parameter",
                effect.index
            ));
        }
        let shape = target_shape(effect, shapes_by_target)?;
        let size = shape_size(&shape)?;
        let end = input_index
            .checked_add(size)
            .ok_or_else(|| "Program AD parameter input index overflowed".to_owned())?;
        let Some(slice) = inputs.get(*input_index..end) else {
            return Err(format!(
                "effect {} parameter input is missing flattened values",
                effect.index
            ));
        };
        let copied = copy_replay_buffer(slice)?;
        *input_index = end;
        return ProgramADNumericValue::new(shape, copied);
    }
    if operation.starts_with("branch:") {
        return evaluate_branch_effect(effect, operation)
            .and_then(ProgramADNumericValue::scalar);
    }
    match operation {
        name if name == "sum" || name.starts_with("sum:") => {
            numeric_sum(effect, name, values, shapes_by_target)
        }
        name if name == "mean" || name.starts_with("mean:") => {
            numeric_mean(effect, name, values, shapes_by_target)
        }
        name if name == "prod" || name.starts_with("prod:") => {
            numeric_prod(effect, name, values, shapes_by_target)
        }
        name if name == "var" || name.starts_with("var:") => {
            numeric_variance(effect, name, values, shapes_by_target)
        }
        name if name == "std" || name.starts_with("std:") => {
            numeric_standard_deviation(effect, name, values, shapes_by_target)
        }
        name if is_order_statistic_operation(name) => {
            numeric_order_statistic(effect, name, values, shapes_by_target)
        }
        name if is_trapezoid_operation(name) => {
            numeric_trapezoid(effect, name, values, shapes_by_target)
        }
        name if is_cumulative_operation(name) => numeric_cumulative(effect, name, values),
        name if is_signal_operation(name) => numeric_signal(effect, name, values),
        name if is_stencil_operation(name) => numeric_stencil(effect, name, values),
        "reshape" => numeric_reshape(effect, values, shapes_by_target),
        "ravel" => numeric_ravel(effect, values, shapes_by_target),
        "broadcast_to" => numeric_broadcast_to(effect, values, shapes_by_target),
        "transpose" => numeric_transpose(effect, values, shapes_by_target),
        name if name == "concatenate" || name.starts_with("concatenate:") => {
            numeric_concatenate(effect, name, values, shapes_by_target)
        }
        name if name == "stack" || name.starts_with("stack:") => {
            numeric_stack(effect, name, values, shapes_by_target)
        }
        name if name == "index_map" || name.starts_with("index_map:") => {
            numeric_index_map(effect, name, values, shapes_by_target)
        }
        "add" => numeric_binary(effect, values, |lhs, rhs| Ok(lhs + rhs)),
        "sub" => numeric_binary(effect, values, |lhs, rhs| Ok(lhs - rhs)),
        "mul" => numeric_binary(effect, values, |lhs, rhs| Ok(lhs * rhs)),
        "div" => numeric_binary(effect, values, |lhs, rhs| {
            if rhs == 0.0 {
                Err("division denominator must be non-zero".to_owned())
            } else {
                Ok(lhs / rhs)
            }
        }),
        "pow" => numeric_binary(effect, values, |lhs, rhs| {
            let value = lhs.powf(rhs);
            if value.is_finite() {
                Ok(value)
            } else {
                Err("power result must be finite".to_owned())
            }
        }),
        "sin" => numeric_unary(effect, values, f64::sin),
        "cos" => numeric_unary(effect, values, f64::cos),
        "exp" => numeric_unary_checked(effect, values, f64::exp, "exp result must be finite"),
        "expm1" => {
            numeric_unary_checked(effect, values, f64::exp_m1, "expm1 result must be finite")
        }
        "log" => numeric_unary_domain(
            effect,
            values,
            |value| value > 0.0,
            f64::ln,
            "log input must be positive",
        ),
        "log1p" => numeric_unary_domain(
            effect,
            values,
            |value| value > -1.0,
            f64::ln_1p,
            "log1p input must be greater than -1",
        ),
        "sqrt" => numeric_unary_domain(
            effect,
            values,
            |value| value > 0.0,
            f64::sqrt,
            "sqrt input must be positive",
        ),
        "tan" => numeric_unary_domain(
            effect,
            values,
            |value| value.cos().abs() > 1.0e-15,
            f64::tan,
            "tan input must have non-zero cosine",
        ),
        "tanh" => numeric_unary(effect, values, f64::tanh),
        "arcsin" => numeric_unary_domain(
            effect,
            values,
            |value| value.abs() < 1.0,
            f64::asin,
            "arcsin input must be strictly inside (-1, 1)",
        ),
        "arccos" => numeric_unary_domain(
            effect,
            values,
            |value| value.abs() < 1.0,
            f64::acos,
            "arccos input must be strictly inside (-1, 1)",
        ),
        "reciprocal" => numeric_unary_domain(
            effect,
            values,
            |value| value != 0.0,
            |value| 1.0 / value,
            "reciprocal input must be non-zero",
        ),
        "abs" => numeric_unary(effect, values, f64::abs),
        name if is_multi_dot_operation(name) => numeric_multi_dot(effect, name, values),
        name if is_matrix_power_operation(name) => numeric_matrix_power(effect, name, values),
        name if is_eigvalsh_operation(name) => numeric_eigvalsh(effect, name, values),
        name if is_eigvals_operation(name) => numeric_eigvals(effect, name, values),
        name if is_eig_operation(name) => numeric_eig(effect, name, values),
        name if is_eigh_operation(name) => numeric_eigh(effect, name, values),
        name if is_svdvals_operation(name) => numeric_svdvals(effect, name, values),
        name if is_pinv_operation(name) => numeric_pinv(effect, name, values),
        name if is_interpolation_operation(name) => numeric_interpolation(effect, name, values),
        name if is_diag_operation(name) => numeric_diag(effect, name, values),
        name if is_diagflat_operation(name) => numeric_diagflat(effect, name, values),
        name if name.starts_with("linalg:trace:")
            || name.starts_with("linalg:det:")
            || name.starts_with("linalg:inv:")
            || name.starts_with("linalg:solve:") =>
        {
            evaluate_scalar_linalg_effect(effect, operation, values)
        }
        _ => Err(format!(
            "effect {} operation {operation} is outside bounded Rust elementwise/structural array value+gradient replay",
            effect.index
        )),
    }
}

fn evaluate_scalar_linalg_effect(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    admit_replay_table::<(String, f64)>(values.len())?;
    let mut scalar_values = HashMap::new();
    scalar_values
        .try_reserve(values.len())
        .map_err(|error| format!("Program AD scalar value-map allocation refused: {error}"))?;
    for (index, (key, value)) in values.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        let scalar = value.scalar_value()?;
        scalar_values.insert(copy_replay_symbol(key)?, scalar);
    }
    let mut input_index = 0usize;
    evaluate_effect(effect, operation, &[], &mut input_index, &scalar_values)
        .and_then(ProgramADNumericValue::scalar)
}

fn ssa_shapes_by_target(ir: &ProgramADEffectIR) -> Result<ProgramADShapeMap<'_>, String> {
    admit_replay_table::<(&str, &[usize])>(ir.ssa_values.len())?;
    let mut shapes = HashMap::new();
    shapes
        .try_reserve(ir.ssa_values.len())
        .map_err(|error| format!("Program AD shape-map allocation refused: {error}"))?;
    for (index, value) in ir.ssa_values.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        shapes.insert(value.name.as_str(), value.shape.as_slice());
    }
    Ok(shapes)
}

fn target_shape(
    effect: &ProgramADEffect,
    shapes_by_target: &ProgramADShapeMap<'_>,
) -> Result<Vec<usize>, String> {
    let source = shapes_by_target.get(effect.target.as_str()).ok_or_else(|| {
        format!(
            "effect {} target {} is missing SSA shape metadata",
            effect.index, effect.target
        )
    })?;
    copy_replay_buffer(source)
}

fn append_parameter_targets_for_effect(
    effect: &ProgramADEffect,
    value: &ProgramADNumericValue,
    targets: &mut Vec<ScalarParameterTarget>,
) -> Result<(), String> {
    let count = if value.shape.is_empty() {
        1
    } else {
        value.values.len()
    };
    if targets.capacity() - targets.len() < count {
        return Err("Program AD parameter metadata exceeds admitted target capacity".to_owned());
    }
    for flat_index in 0..count {
        if flat_index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        let label = if value.shape.is_empty() {
            copy_replay_symbol(&effect.target)?
        } else {
            let capacity = effect
                .target
                .len()
                .checked_add(2 + usize::BITS as usize)
                .ok_or_else(|| "Program AD parameter label size overflowed".to_owned())?;
            crate::program_ad_lifecycle::admit_replay_metadata(capacity)?;
            let mut label = String::new();
            label
                .try_reserve_exact(capacity)
                .map_err(|error| format!("Program AD parameter-label allocation refused: {error}"))?;
            // A usize decimal representation has fewer digits than its bit width.
            std::fmt::write(&mut label, format_args!("{}[{flat_index}]", effect.target))
                .map_err(|error| format!("Program AD parameter-label encoding failed: {error}"))?;
            label
        };
        targets.push(ScalarParameterTarget {
            label,
            source: copy_replay_symbol(&effect.target)?,
            flat_index,
        });
    }
    Ok(())
}

fn copy_replay_symbol(source: &str) -> Result<String, String> {
    crate::program_ad_lifecycle::replay_checkpoint()?;
    crate::program_ad_lifecycle::admit_replay_metadata(source.len())?;
    let mut symbol = String::new();
    symbol
        .try_reserve_exact(source.len())
        .map_err(|error| format!("Program AD replay-symbol allocation refused: {error}"))?;
    for (index, character) in source.chars().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        symbol.push(character);
    }
    crate::program_ad_lifecycle::replay_checkpoint()?;
    Ok(symbol)
}

fn shape_size(shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for (index, dimension) in shape.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        if *dimension == 0 {
            return Err("Program AD shaped values must have non-zero dimensions".to_owned());
        }
        size = size
            .checked_mul(*dimension)
            .ok_or_else(|| "Program AD shaped value size overflowed".to_owned())?;
    }
    if size > isize::MAX as usize / std::mem::size_of::<f64>() {
        return Err("Program AD shaped value bytes exceed native addressability".to_owned());
    }
    Ok(size)
}

fn numeric_operand(
    name: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    if let Some(value) = values.get(name) {
        return value.try_clone();
    }
    name.parse::<f64>()
        .map_err(|_| format!("operand {name} is neither an SSA value nor a scalar literal"))
        .and_then(ProgramADNumericValue::scalar)
}

fn numeric_operands(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<Vec<ProgramADNumericValue>, String> {
    if effect.inputs.is_empty() {
        return Err(format!(
            "effect {} requires at least one input",
            effect.index
        ));
    }
    let mut operands = Vec::new();
    operands
        .try_reserve_exact(effect.inputs.len())
        .map_err(|error| format!("Program AD operand-list allocation refused: {error}"))?;
    for (index, input) in effect.inputs.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        operands.push(numeric_operand(input, values)?);
    }
    Ok(operands)
}
