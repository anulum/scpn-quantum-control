// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD reverse dispatch

fn accumulate_reverse_effect(
    effect: &ProgramADEffect,
    operation: &str,
    cotangent: ProgramADNumericValue,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
) -> Result<(), String> {
    match operation {
        operation if operation.starts_with("branch:") => Ok(()),
        name if name == "sum" || name.starts_with("sum:") => {
            accumulate_sum(effect, name, values, adjoints, &cotangent)
        }
        name if name == "mean" || name.starts_with("mean:") => {
            accumulate_mean(effect, name, values, adjoints, &cotangent)
        }
        name if name == "prod" || name.starts_with("prod:") => {
            accumulate_prod(effect, name, values, adjoints, &cotangent)
        }
        name if name == "var" || name.starts_with("var:") => {
            accumulate_variance(effect, name, values, adjoints, &cotangent)
        }
        name if name == "std" || name.starts_with("std:") => {
            accumulate_standard_deviation(effect, name, values, adjoints, &cotangent)
        }
        name if is_order_statistic_operation(name) => {
            accumulate_order_statistic(effect, name, values, adjoints, &cotangent)
        }
        name if is_trapezoid_operation(name) => {
            accumulate_trapezoid(effect, name, values, adjoints, &cotangent)
        }
        name if is_cumulative_operation(name) => {
            accumulate_cumulative(effect, name, values, adjoints, &cotangent)
        }
        name if is_interpolation_operation(name) => {
            accumulate_interpolation(effect, name, values, adjoints, &cotangent)
        }
        name if is_signal_operation(name) => {
            accumulate_signal(effect, name, values, adjoints, &cotangent)
        }
        name if is_stencil_operation(name) => {
            accumulate_stencil(effect, name, values, adjoints, &cotangent)
        }
        "reshape" | "ravel" => accumulate_reshape_like(effect, values, adjoints, &cotangent),
        "broadcast_to" => accumulate_broadcast_to(effect, values, adjoints, &cotangent),
        "transpose" => accumulate_transpose(effect, values, adjoints, &cotangent),
        name if name == "concatenate" || name.starts_with("concatenate:") => {
            accumulate_concatenate(effect, name, values, adjoints, &cotangent)
        }
        name if name == "stack" || name.starts_with("stack:") => {
            accumulate_stack(effect, name, values, adjoints, &cotangent)
        }
        name if name == "index_map" || name.starts_with("index_map:") => {
            accumulate_index_map(effect, name, values, adjoints, &cotangent)
        }
        "add" => accumulate_add_sub(effect, values, adjoints, &cotangent, 1.0, 1.0),
        "sub" => accumulate_add_sub(effect, values, adjoints, &cotangent, 1.0, -1.0),
        "mul" => {
            let (lhs, rhs, _shape) = binary_operands(effect, values)?;
            let lhs_contribution =
                elementwise_mul(&cotangent, &broadcast_to(&rhs, &cotangent.shape)?)?;
            let rhs_contribution =
                elementwise_mul(&cotangent, &broadcast_to(&lhs, &cotangent.shape)?)?;
            add_numeric_adjoint(&effect.inputs[0], lhs_contribution, values, adjoints)?;
            add_numeric_adjoint(&effect.inputs[1], rhs_contribution, values, adjoints)
        }
        "div" => {
            let (lhs, rhs, _shape) = binary_operands(effect, values)?;
            if rhs.values.contains(&0.0) {
                return Err("division denominator must be non-zero".to_owned());
            }
            let rhs_broadcast = broadcast_to(&rhs, &cotangent.shape)?;
            let lhs_broadcast = broadcast_to(&lhs, &cotangent.shape)?;
            let lhs_contribution =
                elementwise_binary(&cotangent, &rhs_broadcast, |cot, r| Ok(cot / r))?;
            let rhs_contribution =
                elementwise_binary3(&cotangent, &lhs_broadcast, &rhs_broadcast, |cot, l, r| {
                    Ok(cot * (-l / (r * r)))
                })?;
            add_numeric_adjoint(&effect.inputs[0], lhs_contribution, values, adjoints)?;
            add_numeric_adjoint(&effect.inputs[1], rhs_contribution, values, adjoints)
        }
        "pow" => {
            let (lhs, rhs, _shape) = binary_operands(effect, values)?;
            if lhs.values.iter().any(|value| *value <= 0.0) {
                return Err("pow gradient requires a positive base".to_owned());
            }
            let lhs_broadcast = broadcast_to(&lhs, &cotangent.shape)?;
            let rhs_broadcast = broadcast_to(&rhs, &cotangent.shape)?;
            let lhs_contribution =
                elementwise_binary3(&cotangent, &lhs_broadcast, &rhs_broadcast, |cot, l, r| {
                    Ok(cot * r * l.powf(r - 1.0))
                })?;
            let rhs_contribution =
                elementwise_binary3(&cotangent, &lhs_broadcast, &rhs_broadcast, |cot, l, r| {
                    Ok(cot * l.powf(r) * l.ln())
                })?;
            add_numeric_adjoint(&effect.inputs[0], lhs_contribution, values, adjoints)?;
            add_numeric_adjoint(&effect.inputs[1], rhs_contribution, values, adjoints)
        }
        "sin" => accumulate_unary(effect, values, adjoints, &cotangent, f64::cos),
        "cos" => accumulate_unary(effect, values, adjoints, &cotangent, |value| -value.sin()),
        "exp" => accumulate_unary(effect, values, adjoints, &cotangent, f64::exp),
        "expm1" => accumulate_unary(effect, values, adjoints, &cotangent, f64::exp),
        "log" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value > 0.0,
            |value| 1.0 / value,
            "log input must be positive",
        ),
        "log1p" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value > -1.0,
            |value| 1.0 / (1.0 + value),
            "log1p input must be greater than -1",
        ),
        "sqrt" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value > 0.0,
            |value| 0.5 / value.sqrt(),
            "sqrt input must be positive",
        ),
        "tan" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value.cos().abs() > 1.0e-15,
            |value| 1.0 / (value.cos() * value.cos()),
            "tan input must have non-zero cosine",
        ),
        "tanh" => accumulate_unary(effect, values, adjoints, &cotangent, |value| {
            let tanh = value.tanh();
            1.0 - tanh * tanh
        }),
        "arcsin" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value.abs() < 1.0,
            |value| 1.0 / (1.0 - value * value).sqrt(),
            "arcsin input must be strictly inside (-1, 1)",
        ),
        "arccos" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value.abs() < 1.0,
            |value| -1.0 / (1.0 - value * value).sqrt(),
            "arccos input must be strictly inside (-1, 1)",
        ),
        "reciprocal" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value != 0.0,
            |value| -1.0 / (value * value),
            "reciprocal input must be non-zero",
        ),
        "abs" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value != 0.0,
            |value| value.signum(),
            "abs gradient is undefined at zero",
        ),
        name if is_multi_dot_operation(name) => {
            accumulate_multi_dot(effect, name, values, adjoints, &cotangent)
        }
        name if is_matrix_power_operation(name) => {
            accumulate_matrix_power(effect, name, values, adjoints, &cotangent)
        }
        name if is_diag_operation(name) => {
            accumulate_diag(effect, name, values, adjoints, &cotangent)
        }
        name if is_eigvalsh_operation(name) => {
            accumulate_eigvalsh(effect, name, values, adjoints, &cotangent)
        }
        name if is_eigvals_operation(name) => {
            accumulate_eigvals(effect, name, values, adjoints, &cotangent)
        }
        name if is_eig_operation(name) => {
            accumulate_eig(effect, name, values, adjoints, &cotangent)
        }
        name if is_eigh_operation(name) => {
            accumulate_eigh(effect, name, values, adjoints, &cotangent)
        }
        name if is_svdvals_operation(name) => {
            accumulate_svdvals(effect, name, values, adjoints, &cotangent)
        }
        name if is_pinv_operation(name) => {
            accumulate_pinv(effect, name, values, adjoints, &cotangent)
        }
        name if is_diagflat_operation(name) => {
            accumulate_diagflat(effect, name, values, adjoints, &cotangent)
        }
        name if name.starts_with("linalg:trace:") => {
            let cotangent_scalar = cotangent.scalar_value()?;
            // d(trace)/d(diagonal element) = 1 for each on-diagonal operand.
            for input in &effect.inputs {
                add_scalar_adjoint(input, cotangent_scalar, values, adjoints)?;
            }
            Ok(())
        }
        "linalg:det:2x2" => {
            let cotangent_scalar = cotangent.scalar_value()?;
            // Cofactor adjoints for det = a*d - b*c: d/da = d, d/db = -c, d/dc = -b, d/dd = a.
            if effect.inputs.len() != 4 {
                return Err(format!(
                    "effect {} linalg:det:2x2 requires four operands",
                    effect.index
                ));
            }
            let a = operand_scalar_value(&effect.inputs[0], values)?;
            let b = operand_scalar_value(&effect.inputs[1], values)?;
            let c = operand_scalar_value(&effect.inputs[2], values)?;
            let d = operand_scalar_value(&effect.inputs[3], values)?;
            add_scalar_adjoint(&effect.inputs[0], cotangent_scalar * d, values, adjoints)?;
            add_scalar_adjoint(&effect.inputs[1], cotangent_scalar * (-c), values, adjoints)?;
            add_scalar_adjoint(&effect.inputs[2], cotangent_scalar * (-b), values, adjoints)?;
            add_scalar_adjoint(&effect.inputs[3], cotangent_scalar * a, values, adjoints)?;
            Ok(())
        }
        "linalg:det:3x3" => {
            let cotangent_scalar = cotangent.scalar_value()?;
            // d(det)/dA_{ij} is the (i,j) cofactor of the row-major 3x3 matrix.
            if effect.inputs.len() != 9 {
                return Err(format!(
                    "effect {} linalg:det:3x3 requires nine operands",
                    effect.index
                ));
            }
            let [a, b, c, d, e, f, g, h, i] = read_3x3_numeric(effect, values)?;
            let cofactors = [
                e * i - f * h,
                f * g - d * i,
                d * h - e * g,
                c * h - b * i,
                a * i - c * g,
                b * g - a * h,
                b * f - c * e,
                c * d - a * f,
                a * e - b * d,
            ];
            for (input, cofactor) in effect.inputs.iter().zip(cofactors.iter()) {
                add_scalar_adjoint(input, cotangent_scalar * cofactor, values, adjoints)?;
            }
            Ok(())
        }
        name if name.starts_with("linalg:det:") => {
            let cotangent_scalar = cotangent.scalar_value()?;
            // General determinant (4x4 and up): d(det)/dA_{ij} = det * (A^{-1})_{ji}.
            let n = parse_det_dim(name).ok_or_else(|| {
                format!(
                    "effect {} {name} has no determinant dimension",
                    effect.index
                )
            })?;
            if effect.inputs.len() != n * n {
                return Err(format!(
                    "effect {} {name} requires {} operands",
                    effect.index,
                    n * n
                ));
            }
            let matrix = effect
                .inputs
                .iter()
                .map(|input| operand_scalar_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            let determinant = determinant_general(&matrix, n)?;
            let inverse = invert_square(&matrix, n)?;
            for i in 0..n {
                for j in 0..n {
                    let cofactor = determinant * inverse[j * n + i];
                    add_scalar_adjoint(
                        &effect.inputs[i * n + j],
                        cotangent_scalar * cofactor,
                        values,
                        adjoints,
                    )?;
                }
            }
            Ok(())
        }
        name if name.starts_with("linalg:inv:") => {
            let cotangent_scalar = cotangent.scalar_value()?;
            // d(A^{-1})_{ij}/dA_{kl} = -(A^{-1})_{ik} (A^{-1})_{lj}.
            let (n, row, column) = parse_inv_index(name)
                .ok_or_else(|| format!("effect {} {name} has no inverse index", effect.index))?;
            if effect.inputs.len() != n * n {
                return Err(format!(
                    "effect {} {name} requires {} operands",
                    effect.index,
                    n * n
                ));
            }
            let matrix = effect
                .inputs
                .iter()
                .map(|input| operand_scalar_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            let m = invert_square(&matrix, n)?;
            for k in 0..n {
                for l in 0..n {
                    let contribution = cotangent_scalar * (-m[row * n + k] * m[l * n + column]);
                    add_scalar_adjoint(&effect.inputs[k * n + l], contribution, values, adjoints)?;
                }
            }
            Ok(())
        }
        name if name.starts_with("linalg:solve:") => {
            let cotangent_scalar = cotangent.scalar_value()?;
            // X = A^{-1} B: dB = A^{-T}G and dA = -(A^{-T}G)X^T.
            let output = parse_solve_output(name)
                .ok_or_else(|| format!("effect {} {name} has no solution index", effect.index))?;
            let expected_inputs = output.n * output.n + output.rhs_size();
            if effect.inputs.len() != expected_inputs {
                return Err(format!(
                    "effect {} {name} requires {} operands",
                    effect.index, expected_inputs
                ));
            }
            let operands = effect
                .inputs
                .iter()
                .map(|input| operand_scalar_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            let inverse = invert_square(&operands[..output.n * output.n], output.n)?;
            let rhs = &operands[output.n * output.n..];
            let mut solution = vec![0.0; output.rhs_size()];
            for solution_row in 0..output.n {
                for solution_column in 0..output.rhs_columns {
                    solution[solution_row * output.rhs_columns + solution_column] = (0..output.n)
                        .map(|j| {
                            inverse[solution_row * output.n + j]
                                * rhs[j * output.rhs_columns + solution_column]
                        })
                        .sum();
                }
            }
            for j in 0..output.n {
                let rhs_input = output.n * output.n + j * output.rhs_columns + output.column;
                add_scalar_adjoint(
                    &effect.inputs[rhs_input],
                    cotangent_scalar * inverse[output.row * output.n + j],
                    values,
                    adjoints,
                )?;
            }
            for k in 0..output.n {
                for l in 0..output.n {
                    let contribution = cotangent_scalar
                        * (-inverse[output.row * output.n + k]
                            * solution[l * output.rhs_columns + output.column]);
                    add_scalar_adjoint(
                        &effect.inputs[k * output.n + l],
                        contribution,
                        values,
                        adjoints,
                    )?;
                }
            }
            Ok(())
        }
        _ => Err(format!(
            "effect {} operation {operation} is outside bounded Rust scalar value+gradient replay",
            effect.index
        )),
    }
}
