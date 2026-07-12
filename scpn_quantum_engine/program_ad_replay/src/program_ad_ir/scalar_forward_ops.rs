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
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            cumulative_output_value(effect.index, name, &input_values)
        }
        name if is_interpolation_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            interpolation_output_value(effect.index, name, &input_values)
        }
        name if is_signal_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            signal_output_value(effect.index, name, &input_values)
        }
        name if is_stencil_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            stencil_output_value(effect.index, name, &input_values)
        }
        name if is_multi_dot_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            multi_dot_output_value(effect.index, name, &input_values)
        }
        name if is_matrix_power_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            matrix_power_output_value(effect.index, name, &input_values)
        }
        name if is_eigvalsh_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            eigvalsh_output_value(effect.index, name, &input_values)
        }
        name if is_eigvals_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            eigvals_output_value(effect.index, name, &input_values)
        }
        name if is_eig_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            eig_output_value(effect.index, name, &input_values)
        }
        name if is_eigh_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            eigh_output_value(effect.index, name, &input_values)
        }
        name if is_svdvals_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            svdvals_output_value(effect.index, name, &input_values)
        }
        name if is_pinv_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            pinv_output_value(effect.index, name, &input_values)
        }
        name if is_diagflat_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            diagflat_output_value(effect.index, name, &input_values)
        }
        name if is_diag_operation(name) => {
            let input_values = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            diag_output_value(effect.index, name, &input_values)
        }
        name if name.starts_with("linalg:trace:") => {
            // The trace opcode carries the on-diagonal element operands; its value is their sum.
            let mut total = 0.0;
            for input in &effect.inputs {
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
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            determinant_general(&matrix, n)
        }
        name if name.starts_with("linalg:inv:") => {
            // Each opcode emits one element (row, column) of the matrix inverse.
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
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            Ok(invert_square(&matrix, n)?[row * n + column])
        }
        name if name.starts_with("linalg:solve:") => {
            // Each opcode emits one component of X = A^{-1} B.
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
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            let inverse = invert_square(&operands[..output.n * output.n], output.n)?;
            let rhs = &operands[output.n * output.n..];
            Ok((0..output.n)
                .map(|j| {
                    inverse[output.row * output.n + j] * rhs[j * output.rhs_columns + output.column]
                })
                .sum())
        }
        _ => Err(format!(
            "effect {} operation {operation} is outside the bounded Rust scalar interpreter",
            effect.index
        )),
    }
}

fn validate_executed_branch_metadata(ir: &ProgramADEffectIR) -> Result<(), String> {
    let mut branch_effects_by_operation: HashMap<&str, usize> = HashMap::new();
    for effect in &ir.effects {
        let Some(operation) = effect.operation.as_deref() else {
            continue;
        };
        if !operation.starts_with("branch:") {
            continue;
        }
        if effect.kind != "control_branch" {
            return Err(format!(
                "branch effect {} must have kind control_branch",
                effect.index
            ));
        }
        if !effect.inputs.is_empty() {
            return Err(format!(
                "branch effect {} must not carry differentiable inputs",
                effect.index
            ));
        }
        branch_effects_by_operation.insert(operation, effect.index);
    }

    if ir.control_regions.is_empty() && ir.phi_nodes.is_empty() {
        return Ok(());
    }
    if ir.control_regions.is_empty() || ir.phi_nodes.is_empty() {
        return Err(
            "runtime branch metadata must include both control regions and phi nodes".to_owned(),
        );
    }

    let mut runtime_region_entered_by_index: HashMap<usize, bool> = HashMap::new();
    let mut source_region_indices: HashSet<usize> = HashSet::new();
    for region in &ir.control_regions {
        if region.kind == "source_control_flow" {
            source_region_indices.insert(region.index);
            continue;
        }
        if region.kind != "runtime_branch" {
            return Err(
                "only executed runtime_branch metadata is supported by bounded Rust branch replay"
                    .to_owned(),
            );
        }
        let Some(predicate) = region.predicate.as_deref() else {
            return Err("runtime branch metadata must include a predicate".to_owned());
        };
        if !predicate.starts_with("branch:") {
            return Err("runtime branch predicate must reference a branch operation".to_owned());
        }
        if !branch_effects_by_operation.contains_key(predicate) {
            return Err("runtime branch predicate must match a control_branch effect".to_owned());
        }
        let predicate_entered = branch_operation_value(predicate)?;
        if predicate_entered != region.entered {
            return Err("runtime branch predicate and entered flag disagree".to_owned());
        }
        runtime_region_entered_by_index.insert(region.index, region.entered);
    }

    let mut phi_count_by_region: HashMap<usize, usize> = HashMap::new();
    for phi in &ir.phi_nodes {
        let Some(region_index) = phi.control_region else {
            return Err("runtime branch phi metadata must reference a control region".to_owned());
        };
        if source_region_indices.contains(&region_index) {
            continue;
        }
        let Some(entered) = runtime_region_entered_by_index.get(&region_index) else {
            return Err(
                "runtime branch phi metadata must reference a runtime_branch region".to_owned(),
            );
        };
        let Some(selected) = phi.selected.as_deref() else {
            return Err("runtime branch phi metadata must record selected path".to_owned());
        };
        let expected_selected = if *entered {
            "executed_true"
        } else {
            "executed_false"
        };
        if selected != expected_selected {
            return Err(
                "runtime branch phi selected path disagrees with executed branch".to_owned(),
            );
        }
        let has_true = phi.incoming.iter().any(|value| value == "executed_true");
        let has_false = phi.incoming.iter().any(|value| value == "executed_false");
        if !has_true || !has_false {
            return Err(
                "runtime branch phi incoming paths must include executed_true and executed_false"
                    .to_owned(),
            );
        }
        *phi_count_by_region.entry(region_index).or_insert(0) += 1;
    }
    for region_index in runtime_region_entered_by_index.keys() {
        if phi_count_by_region.get(region_index) != Some(&1) {
            return Err("each runtime branch region must have exactly one phi node".to_owned());
        }
    }
    Ok(())
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
    match n {
        2 => invert_2x2(matrix[0], matrix[1], matrix[2], matrix[3]).map(|m| m.to_vec()),
        3 => {
            let mut m = [0.0_f64; 9];
            m.copy_from_slice(&matrix[..9]);
            invert_3x3(m).map(|inv| inv.to_vec())
        }
        _ => invert_general(matrix, n),
    }
}

/// Invert an `n x n` row-major matrix by Gauss-Jordan elimination with partial pivoting.
///
/// Fails closed on a singular or non-finite system. Used for dimensions above the closed-form
/// 2x2/3x3 paths.
fn invert_general(matrix: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let width = 2 * n;
    let mut augmented = vec![0.0_f64; n * width];
    for row in 0..n {
        for column in 0..n {
            augmented[row * width + column] = matrix[row * n + column];
        }
        augmented[row * width + n + row] = 1.0;
    }
    for column in 0..n {
        let mut pivot = column;
        let mut best = augmented[column * width + column].abs();
        for row in (column + 1)..n {
            let candidate = augmented[row * width + column].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        if best == 0.0 || !best.is_finite() {
            return Err(format!("linalg {n}x{n} matrix is singular"));
        }
        if pivot != column {
            for c in 0..width {
                augmented.swap(pivot * width + c, column * width + c);
            }
        }
        let pivot_value = augmented[column * width + column];
        for c in 0..width {
            augmented[column * width + c] /= pivot_value;
        }
        for row in 0..n {
            if row != column {
                let factor = augmented[row * width + column];
                if factor != 0.0 {
                    for c in 0..width {
                        augmented[row * width + c] -= factor * augmented[column * width + c];
                    }
                }
            }
        }
    }
    let mut inverse = vec![0.0_f64; n * n];
    for row in 0..n {
        for column in 0..n {
            inverse[row * n + column] = augmented[row * width + n + column];
        }
    }
    if inverse.iter().any(|value| !value.is_finite()) {
        return Err(format!("linalg {n}x{n} inverse is non-finite"));
    }
    Ok(inverse)
}

/// Determinant of an `n x n` row-major matrix by LU factorisation with partial pivoting.
fn determinant_general(matrix: &[f64], n: usize) -> Result<f64, String> {
    let mut work = matrix.to_vec();
    let mut sign = 1.0_f64;
    for column in 0..n {
        let mut pivot = column;
        let mut best = work[column * n + column].abs();
        for row in (column + 1)..n {
            let candidate = work[row * n + column].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        if best == 0.0 {
            return Ok(0.0);
        }
        if pivot != column {
            for c in 0..n {
                work.swap(pivot * n + c, column * n + c);
            }
            sign = -sign;
        }
        let pivot_value = work[column * n + column];
        for row in (column + 1)..n {
            let factor = work[row * n + column] / pivot_value;
            for c in column..n {
                work[row * n + c] -= factor * work[column * n + c];
            }
        }
    }
    let mut determinant = sign;
    for k in 0..n {
        determinant *= work[k * n + k];
    }
    if !determinant.is_finite() {
        return Err(format!("linalg {n}x{n} determinant is non-finite"));
    }
    Ok(determinant)
}

/// Parse the square dimension `n` from a `linalg:det:NxN` opcode.
fn parse_det_dim(operation: &str) -> Option<usize> {
    let parts: Vec<&str> = operation.split(':').collect();
    if parts.len() != 3 {
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
    let parts: Vec<&str> = operation.split(':').collect();
    if parts.len() != 5 {
        return None;
    }
    let n = parse_square_dim(parts[2])?;
    let row: usize = parts[3].parse().ok()?;
    let column: usize = parts[4].parse().ok()?;
    (row < n && column < n).then_some((n, row, column))
}

/// Parse selected output metadata from a `linalg:solve:NxN:rhs:<shape>:...` opcode.
fn parse_solve_output(operation: &str) -> Option<SolveOutput> {
    let parts: Vec<&str> = operation.split(':').collect();
    if parts.len() != 6 && parts.len() != 7 {
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
    if parts.len() == 6 {
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

fn require_non_empty(value: &str, name: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("program AD IR {name} must be non-empty"));
    }
    Ok(())
}

fn require_positive_optional(value: Option<usize>, name: &str) -> Result<(), String> {
    if value == Some(0) {
        return Err(format!(
            "program AD IR {name} must be positive when present"
        ));
    }
    Ok(())
}
