// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD scalar opcode evaluation

fn evaluate_effect(
    effect: &ProgramADEffect,
    operation: &str,
    inputs: &[f64],
    input_index: &mut usize,
    values: &HashMap<String, f64>,
) -> Result<f64, String> {
    if operation == "parameter" {
        if effect.kind != "parameter" {
            return Err(format!(
                "effect {} operation parameter must have kind parameter",
                effect.index
            ));
        }
        let Some(value) = inputs.get(*input_index) else {
            return Err(format!(
                "effect {} parameter input is missing",
                effect.index
            ));
        };
        *input_index += 1;
        return Ok(*value);
    }
    if operation.starts_with("branch:") {
        return evaluate_branch_effect(effect, operation);
    }
    match operation {
        "add" => binary(effect, values, |lhs, rhs| Ok(lhs + rhs)),
        "sub" => binary(effect, values, |lhs, rhs| Ok(lhs - rhs)),
        "mul" => binary(effect, values, |lhs, rhs| Ok(lhs * rhs)),
        "div" => binary(effect, values, |lhs, rhs| {
            if rhs == 0.0 {
                Err("division denominator must be non-zero".to_owned())
            } else {
                Ok(lhs / rhs)
            }
        }),
        "pow" => binary(effect, values, |lhs, rhs| {
            let value = lhs.powf(rhs);
            if value.is_finite() {
                Ok(value)
            } else {
                Err("power result must be finite".to_owned())
            }
        }),
        "sin" => unary(effect, values, f64::sin),
        "cos" => unary(effect, values, f64::cos),
        "exp" => unary_checked(effect, values, f64::exp, "exp result must be finite"),
        "expm1" => unary_checked(effect, values, f64::exp_m1, "expm1 result must be finite"),
        "log" => unary_domain(
            effect,
            values,
            |value| value > 0.0,
            f64::ln,
            "log input must be positive",
        ),
        "log1p" => unary_domain(
            effect,
            values,
            |value| value > -1.0,
            f64::ln_1p,
            "log1p input must be greater than -1",
        ),
        "sqrt" => unary_domain(
            effect,
            values,
            |value| value > 0.0,
            f64::sqrt,
            "sqrt input must be positive",
        ),
        "tan" => unary_domain(
            effect,
            values,
            |value| value.cos().abs() > 1.0e-15,
            f64::tan,
            "tan input must have non-zero cosine",
        ),
        "tanh" => unary(effect, values, f64::tanh),
        "arcsin" => unary_domain(
            effect,
            values,
            |value| value.abs() < 1.0,
            f64::asin,
            "arcsin input must be strictly inside (-1, 1)",
        ),
        "arccos" => unary_domain(
            effect,
            values,
            |value| value.abs() < 1.0,
            f64::acos,
            "arccos input must be strictly inside (-1, 1)",
        ),
        "reciprocal" => unary_domain(
            effect,
            values,
            |value| value != 0.0,
            |value| 1.0 / value,
            "reciprocal input must be non-zero",
        ),
        "abs" => unary(effect, values, f64::abs),
        name if is_cumulative_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            cumulative_output_value(effect.index, name, &input_values)
        }
        name if is_interpolation_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            interpolation_output_value(effect.index, name, &input_values)
        }
        name if is_signal_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            signal_output_value(effect.index, name, &input_values)
        }
        name if is_stencil_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            stencil_output_value(effect.index, name, &input_values)
        }
        name if is_multi_dot_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            multi_dot_output_value(effect.index, name, &input_values)
        }
        name if is_matrix_power_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            matrix_power_output_value(effect.index, name, &input_values)
        }
        name if is_eigvalsh_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            eigvalsh_output_value(effect.index, name, &input_values)
        }
        name if is_eigvals_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            eigvals_output_value(effect.index, name, &input_values)
        }
        name if is_eig_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            eig_output_value(effect.index, name, &input_values)
        }
        name if is_eigh_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            eigh_output_value(effect.index, name, &input_values)
        }
        name if is_svdvals_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            svdvals_output_value(effect.index, name, &input_values)
        }
        name if is_pinv_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            pinv_output_value(effect.index, name, &input_values)
        }
        name if is_diagflat_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            diagflat_output_value(effect.index, name, &input_values)
        }
        name if is_diag_operation(name) => {
            let input_values = scalar_replay_operands(effect, values)?;
            diag_output_value(effect.index, name, &input_values)
        }
        name if name.starts_with("linalg:trace:") => {
            // The trace opcode carries the on-diagonal element operands; its value is their sum.
            let mut total = 0.0;
            for (index, input) in effect.inputs.iter().enumerate() {
                if index % 256 == 0 {
                    crate::program_ad_lifecycle::replay_checkpoint()?;
                }
                total += operand_value(input, values)?;
            }
            Ok(total)
        }
        "linalg:det:2x2" => {
            // Row-major operands [a, b, c, d]; det = a*d - b*c.
            if effect.inputs.len() != 4 {
                return Err(format!(
                    "effect {} linalg:det:2x2 requires four operands",
                    effect.index
                ));
            }
            let a = operand_value(&effect.inputs[0], values)?;
            let b = operand_value(&effect.inputs[1], values)?;
            let c = operand_value(&effect.inputs[2], values)?;
            let d = operand_value(&effect.inputs[3], values)?;
            Ok(a * d - b * c)
        }
        "linalg:det:3x3" => {
            // Row-major operands [a,b,c, d,e,f, g,h,i]; Laplace expansion along the first row.
            if effect.inputs.len() != 9 {
                return Err(format!(
                    "effect {} linalg:det:3x3 requires nine operands",
                    effect.index
                ));
            }
            let m = read_3x3(effect, values)?;
            let [a, b, c, d, e, f, g, h, i] = m;
            Ok(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))
        }
        name if name.starts_with("linalg:det:") => {
            // General determinant (4x4 and up) via LU factorisation with partial pivoting.
            let n = parse_det_dim(name).ok_or_else(|| {
                format!(
                    "effect {} {name} has no determinant dimension",
                    effect.index
                )
            })?;
            let matrix_size = shape_size(&[n, n])?;
            if effect.inputs.len() != matrix_size {
                return Err(format!(
                    "effect {} {name} requires {} operands",
                    effect.index,
                    matrix_size
                ));
            }
            let matrix = scalar_replay_operands(effect, values)?;
            determinant_general(&matrix, n)
        }
        name if name.starts_with("linalg:inv:") => {
            // Each opcode emits one element (row, column) of the matrix inverse.
            let (n, row, column) = parse_inv_index(name)
                .ok_or_else(|| format!("effect {} {name} has no inverse index", effect.index))?;
            let matrix_size = shape_size(&[n, n])?;
            if effect.inputs.len() != matrix_size {
                return Err(format!(
                    "effect {} {name} requires {} operands",
                    effect.index,
                    matrix_size
                ));
            }
            let matrix = scalar_replay_operands(effect, values)?;
            Ok(invert_square(&matrix, n)?[row * n + column])
        }
        name if name.starts_with("linalg:solve:") => {
            // Each opcode emits one component of X = A^{-1} B.
            let output = parse_solve_output(name)
                .ok_or_else(|| format!("effect {} {name} has no solution index", effect.index))?;
            let matrix_size = output.matrix_size()?;
            let expected_inputs = output.input_size()?;
            if effect.inputs.len() != expected_inputs {
                return Err(format!(
                    "effect {} {name} requires {} operands",
                    effect.index, expected_inputs
                ));
            }
            let operands = scalar_replay_operands(effect, values)?;
            let inverse = invert_square(&operands[..matrix_size], output.n)?;
            let rhs = &operands[matrix_size..];
            solve_output_value(&inverse, rhs, output)
        }
        _ => Err(format!(
            "effect {} operation {operation} is outside the bounded Rust scalar interpreter",
            effect.index
        )),
    }
}

fn evaluate_branch_effect(effect: &ProgramADEffect, operation: &str) -> Result<f64, String> {
    if effect.kind != "control_branch" {
        return Err(format!(
            "effect {} branch operation must have kind control_branch",
            effect.index
        ));
    }
    if !effect.inputs.is_empty() {
        return Err(format!(
            "effect {} branch operation must not carry differentiable inputs",
            effect.index
        ));
    }
    Ok(if branch_operation_value(operation)? {
        1.0
    } else {
        0.0
    })
}

fn branch_operation_value(operation: &str) -> Result<bool, String> {
    if operation.ends_with(":True") {
        Ok(true)
    } else if operation.ends_with(":False") {
        Ok(false)
    } else {
        Err("branch operation must end with :True or :False".to_owned())
    }
}

fn unary(
    effect: &ProgramADEffect,
    values: &HashMap<String, f64>,
    function: fn(f64) -> f64,
) -> Result<f64, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let value = operand_value(&effect.inputs[0], values)?;
    Ok(function(value))
}

fn unary_checked(
    effect: &ProgramADEffect,
    values: &HashMap<String, f64>,
    function: fn(f64) -> f64,
    finite_error: &str,
) -> Result<f64, String> {
    let value = unary(effect, values, function)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(finite_error.to_owned())
    }
}

fn unary_domain(
    effect: &ProgramADEffect,
    values: &HashMap<String, f64>,
    predicate: fn(f64) -> bool,
    function: fn(f64) -> f64,
    domain_error: &str,
) -> Result<f64, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let value = operand_value(&effect.inputs[0], values)?;
    if !predicate(value) {
        return Err(domain_error.to_owned());
    }
    let result = function(value);
    if result.is_finite() {
        Ok(result)
    } else {
        Err(format!("effect {} result must be finite", effect.index))
    }
}

fn binary(
    effect: &ProgramADEffect,
    values: &HashMap<String, f64>,
    function: impl Fn(f64, f64) -> Result<f64, String>,
) -> Result<f64, String> {
    if effect.inputs.len() != 2 {
        return Err(format!("effect {} requires two inputs", effect.index));
    }
    let lhs = operand_value(&effect.inputs[0], values)?;
    let rhs = operand_value(&effect.inputs[1], values)?;
    let value = function(lhs, rhs)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("effect {} result must be finite", effect.index))
    }
}

fn scalar_replay_operands(
    effect: &ProgramADEffect,
    values: &HashMap<String, f64>,
) -> Result<Vec<f64>, String> {
    let mut operands = reserve_replay_buffer(effect.inputs.len())?;
    for (index, input) in effect.inputs.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        operands.push(operand_value(input, values)?);
    }
    Ok(operands)
}

fn operand_value(name: &str, values: &HashMap<String, f64>) -> Result<f64, String> {
    if let Some(value) = values.get(name) {
        return Ok(*value);
    }
    name.parse::<f64>()
        .map_err(|_| format!("operand {name} is neither an SSA value nor a scalar literal"))
}

/// Invert a row-major 2x2 matrix `[a, b; c, d]`, returning `[m00, m01, m10, m11]`.
///
/// Fails closed on a singular or non-finite determinant so a degenerate inverse is never
/// silently replayed.
fn invert_2x2(a: f64, b: f64, c: f64, d: f64) -> Result<[f64; 4], String> {
    let det = a * d - b * c;
    if det == 0.0 || !det.is_finite() {
        return Err("linalg 2x2 matrix is singular".to_owned());
    }
    Ok([d / det, -b / det, -c / det, a / det])
}

/// Read the nine row-major operands of a 3x3 linalg opcode as `[a, b, c, d, e, f, g, h, i]`.
fn read_3x3(effect: &ProgramADEffect, values: &HashMap<String, f64>) -> Result<[f64; 9], String> {
    let mut matrix = [0.0_f64; 9];
    for (slot, input) in matrix.iter_mut().zip(effect.inputs.iter()) {
        *slot = operand_value(input, values)?;
    }
    Ok(matrix)
}

/// Invert a row-major 3x3 matrix via the adjugate, returning the inverse row-major.
///
/// Fails closed on a singular or non-finite determinant.
fn invert_3x3(m: [f64; 9]) -> Result<[f64; 9], String> {
    let [a, b, c, d, e, f, g, h, i] = m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det == 0.0 || !det.is_finite() {
        return Err("linalg 3x3 matrix is singular".to_owned());
    }
    // inverse = adjugate / det = cofactor-transpose / det.
    Ok([
        (e * i - f * h) / det,
        (c * h - b * i) / det,
        (b * f - c * e) / det,
        (f * g - d * i) / det,
        (a * i - c * g) / det,
        (c * d - a * f) / det,
        (d * h - e * g) / det,
        (b * g - a * h) / det,
        (a * e - b * d) / det,
    ])
}

/// Invert an `n x n` row-major matrix for the bounded dimensions; fail closed otherwise.
fn invert_square(matrix: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let entries = shape_size(&[n, n])?;
    if matrix.len() != entries {
        return Err("Program AD inverse input length does not match square shape".to_owned());
    }
    match n {
        2 => invert_2x2(matrix[0], matrix[1], matrix[2], matrix[3])
            .and_then(|inverse| copy_replay_buffer(&inverse)),
        3 => {
            let mut m = [0.0_f64; 9];
            m.copy_from_slice(&matrix[..9]);
            invert_3x3(m).and_then(|inverse| copy_replay_buffer(&inverse))
        }
        _ => invert_general(matrix, n),
    }
}

fn parse_linalg_opcode_fields(operation: &str) -> Option<([&str; 7], usize)> {
    let mut fields = [""; 7];
    let mut count = 0usize;
    for field in operation.split(':') {
        if count == fields.len() {
            return None;
        }
        fields[count] = field;
        count += 1;
    }
    Some((fields, count))
}

/// Parse the square dimension `n` from a `linalg:det:NxN` opcode.
fn parse_det_dim(operation: &str) -> Option<usize> {
    let (parts, count) = parse_linalg_opcode_fields(operation)?;
    if count != 3 {
        return None;
    }
    parse_square_dim(parts[2])
}

/// Parse the square dimension `n` from an `NxN` opcode token.
fn parse_square_dim(token: &str) -> Option<usize> {
    let (rows, columns) = token.split_once('x')?;
    let n: usize = rows.parse().ok()?;
    (n > 0 && columns.parse::<usize>().ok()? == n).then_some(n)
}

/// Parse `(n, row, column)` from a `linalg:inv:NxN:I:J` opcode.
fn parse_inv_index(operation: &str) -> Option<(usize, usize, usize)> {
    let (parts, count) = parse_linalg_opcode_fields(operation)?;
    if count != 5 {
        return None;
    }
    let n = parse_square_dim(parts[2])?;
    let row: usize = parts[3].parse().ok()?;
    let column: usize = parts[4].parse().ok()?;
    (row < n && column < n).then_some((n, row, column))
}

/// Parse selected output metadata from a `linalg:solve:NxN:rhs:<shape>:...` opcode.
fn parse_solve_output(operation: &str) -> Option<SolveOutput> {
    let (parts, count) = parse_linalg_opcode_fields(operation)?;
    if count != 6 && count != 7 {
        return None;
    }
    if parts[0] != "linalg" || parts[1] != "solve" || parts[3] != "rhs" {
        return None;
    }
    let n = parse_square_dim(parts[2])?;
    let row: usize = parts[5].parse().ok()?;
    if row >= n {
        return None;
    }
    if count == 6 {
        let rhs_rows: usize = parts[4].parse().ok()?;
        return (rhs_rows == n).then_some(SolveOutput {
            n,
            rhs_columns: 1,
            row,
            column: 0,
        });
    }
    let (rhs_rows, rhs_columns) = parts[4].split_once('x')?;
    let rhs_rows: usize = rhs_rows.parse().ok()?;
    let rhs_columns: usize = rhs_columns.parse().ok()?;
    let column: usize = parts[6].parse().ok()?;
    (rhs_rows == n && rhs_columns > 0 && column < rhs_columns).then_some(SolveOutput {
        n,
        rhs_columns,
        row,
        column,
    })
}
