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
    shapes_by_target: &HashMap<String, Vec<usize>>,
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
        *input_index = end;
        return ProgramADNumericValue::new(shape, slice.to_vec());
    }
    if operation.starts_with("branch:") {
        return evaluate_branch_effect(effect, operation).map(ProgramADNumericValue::scalar);
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
    let scalar_values = values
        .iter()
        .map(|(key, value)| value.scalar_value().map(|scalar| (key.clone(), scalar)))
        .collect::<Result<HashMap<String, f64>, String>>()?;
    let mut input_index = 0usize;
    evaluate_effect(effect, operation, &[], &mut input_index, &scalar_values)
        .map(ProgramADNumericValue::scalar)
}

fn ssa_shapes_by_target(ir: &ProgramADEffectIR) -> HashMap<String, Vec<usize>> {
    ir.ssa_values
        .iter()
        .map(|value| (value.name.clone(), value.shape.clone()))
        .collect()
}

fn target_shape(
    effect: &ProgramADEffect,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<Vec<usize>, String> {
    shapes_by_target
        .get(&effect.target)
        .cloned()
        .ok_or_else(|| {
            format!(
                "effect {} target {} is missing SSA shape metadata",
                effect.index, effect.target
            )
        })
}

fn parameter_targets_for_effect(
    effect: &ProgramADEffect,
    value: &ProgramADNumericValue,
) -> Vec<ScalarParameterTarget> {
    if value.shape.is_empty() {
        return vec![ScalarParameterTarget {
            label: effect.target.clone(),
            source: effect.target.clone(),
            flat_index: 0,
        }];
    }
    (0..value.values.len())
        .map(|flat_index| ScalarParameterTarget {
            label: format!("{}[{flat_index}]", effect.target),
            source: effect.target.clone(),
            flat_index,
        })
        .collect()
}

fn shape_size(shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for dimension in shape {
        if *dimension == 0 {
            return Err("Program AD shaped values must have non-zero dimensions".to_owned());
        }
        size = size
            .checked_mul(*dimension)
            .ok_or_else(|| "Program AD shaped value size overflowed".to_owned())?;
    }
    Ok(size)
}

fn numeric_operand(
    name: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    if let Some(value) = values.get(name) {
        return Ok(value.clone());
    }
    name.parse::<f64>()
        .map(ProgramADNumericValue::scalar)
        .map_err(|_| format!("operand {name} is neither an SSA value nor a scalar literal"))
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
    effect
        .inputs
        .iter()
        .map(|input| numeric_operand(input, values))
        .collect()
}
