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

fn numeric_scalar_operands(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<Vec<f64>, String> {
    let mut operands = reserve_replay_buffer(effect.inputs.len())?;
    for (index, input) in effect.inputs.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        operands.push(operand_scalar_value(input, values)?);
    }
    Ok(operands)
}

fn numeric_multi_dot(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = numeric_scalar_operands(effect, values)?;
    multi_dot_output_value(effect.index, operation, &input_values)
        .and_then(ProgramADNumericValue::scalar)
}

fn numeric_matrix_power(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = numeric_scalar_operands(effect, values)?;
    matrix_power_output_value(effect.index, operation, &input_values)
        .and_then(ProgramADNumericValue::scalar)
}

fn numeric_eigvalsh(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = numeric_scalar_operands(effect, values)?;
    eigvalsh_output_value(effect.index, operation, &input_values)
        .and_then(ProgramADNumericValue::scalar)
}

fn numeric_eigvals(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = numeric_scalar_operands(effect, values)?;
    eigvals_output_value(effect.index, operation, &input_values)
        .and_then(ProgramADNumericValue::scalar)
}

fn numeric_eig(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = numeric_scalar_operands(effect, values)?;
    eig_output_value(effect.index, operation, &input_values)
        .and_then(ProgramADNumericValue::scalar)
}

fn numeric_eigh(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = numeric_scalar_operands(effect, values)?;
    eigh_output_value(effect.index, operation, &input_values)
        .and_then(ProgramADNumericValue::scalar)
}

fn numeric_svdvals(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = numeric_scalar_operands(effect, values)?;
    svdvals_output_value(effect.index, operation, &input_values)
        .and_then(ProgramADNumericValue::scalar)
}

fn numeric_diagflat(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = numeric_scalar_operands(effect, values)?;
    diagflat_output_value(effect.index, operation, &input_values)
        .and_then(ProgramADNumericValue::scalar)
}

fn numeric_diag(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = numeric_scalar_operands(effect, values)?;
    diag_output_value(effect.index, operation, &input_values)
        .and_then(ProgramADNumericValue::scalar)
}

fn numeric_pinv(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = numeric_scalar_operands(effect, values)?;
    pinv_output_value(effect.index, operation, &input_values)
        .and_then(ProgramADNumericValue::scalar)
}

fn numeric_stencil(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let input_values = numeric_scalar_operands(effect, values)?;
    stencil_output_value(effect.index, operation, &input_values)
        .and_then(ProgramADNumericValue::scalar)
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
    let mut output = input.values;
    for (index, item) in output.iter_mut().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        *item = function(*item);
    }
    ProgramADNumericValue::new(input.shape, output)
}

fn numeric_unary_checked(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    function: fn(f64) -> f64,
    finite_error: &str,
) -> Result<ProgramADNumericValue, String> {
    let value = numeric_unary(effect, values, function)?;
    for (index, item) in value.values.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        if !item.is_finite() {
            return Err(finite_error.to_owned());
        }
    }
    Ok(value)
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
    for (index, value) in input.values.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        if !predicate(*value) {
            return Err(domain_error.to_owned());
        }
    }
    let mut output = input.values;
    for (index, item) in output.iter_mut().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        *item = function(*item);
    }
    ProgramADNumericValue::new(input.shape, output)
}

/// Invert an `n x n` row-major matrix by Gauss-Jordan elimination with partial pivoting.
///
/// Fails closed on a singular or non-finite system. Used for dimensions above the closed-form
/// 2x2/3x3 paths.
fn invert_general(matrix: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let entries = shape_size(&[n, n])?;
    let width = n
        .checked_mul(2)
        .ok_or_else(|| "Program AD inverse augmented width overflowed".to_owned())?;
    let augmented_entries = shape_size(&[n, width])?;
    let mut augmented = filled_replay_buffer(augmented_entries, 0.0_f64)?;
    for row in 0..n {
        if row % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        for column in 0..n {
            if column % 256 == 0 {
                crate::program_ad_lifecycle::replay_checkpoint()?;
            }
            augmented[row * width + column] = matrix[row * n + column];
        }
        augmented[row * width + n + row] = 1.0;
    }
    for column in 0..n {
        if column % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        let mut pivot = column;
        let mut best = augmented[column * width + column].abs();
        for row in (column + 1)..n {
            if row % 256 == 0 {
                crate::program_ad_lifecycle::replay_checkpoint()?;
            }
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
                if c % 256 == 0 {
                    crate::program_ad_lifecycle::replay_checkpoint()?;
                }
                augmented.swap(pivot * width + c, column * width + c);
            }
        }
        let pivot_value = augmented[column * width + column];
        for c in 0..width {
            if c % 256 == 0 {
                crate::program_ad_lifecycle::replay_checkpoint()?;
            }
            augmented[column * width + c] /= pivot_value;
        }
        for row in 0..n {
            if row % 256 == 0 {
                crate::program_ad_lifecycle::replay_checkpoint()?;
            }
            if row != column {
                let factor = augmented[row * width + column];
                if factor != 0.0 {
                    for c in 0..width {
                        if c % 256 == 0 {
                            crate::program_ad_lifecycle::replay_checkpoint()?;
                        }
                        augmented[row * width + c] -= factor * augmented[column * width + c];
                    }
                }
            }
        }
    }
    let mut inverse = filled_replay_buffer(entries, 0.0_f64)?;
    for row in 0..n {
        if row % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        for column in 0..n {
            if column % 256 == 0 {
                crate::program_ad_lifecycle::replay_checkpoint()?;
            }
            inverse[row * n + column] = augmented[row * width + n + column];
        }
    }
    for (index, value) in inverse.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        if !value.is_finite() {
            return Err(format!("linalg {n}x{n} inverse is non-finite"));
        }
    }
    Ok(inverse)
}

/// Determinant of an `n x n` row-major matrix by LU factorisation with partial pivoting.
fn determinant_general(matrix: &[f64], n: usize) -> Result<f64, String> {
    let entries = shape_size(&[n, n])?;
    if matrix.len() != entries {
        return Err("Program AD determinant input length does not match square shape".to_owned());
    }
    let mut work = copy_replay_buffer(matrix)?;
    let mut sign = 1.0_f64;
    for column in 0..n {
        if column % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        let mut pivot = column;
        let mut best = work[column * n + column].abs();
        for row in (column + 1)..n {
            if row % 256 == 0 {
                crate::program_ad_lifecycle::replay_checkpoint()?;
            }
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
                if c % 256 == 0 {
                    crate::program_ad_lifecycle::replay_checkpoint()?;
                }
                work.swap(pivot * n + c, column * n + c);
            }
            sign = -sign;
        }
        let pivot_value = work[column * n + column];
        for row in (column + 1)..n {
            if row % 256 == 0 {
                crate::program_ad_lifecycle::replay_checkpoint()?;
            }
            let factor = work[row * n + column] / pivot_value;
            for c in column..n {
                if c % 256 == 0 {
                    crate::program_ad_lifecycle::replay_checkpoint()?;
                }
                work[row * n + c] -= factor * work[column * n + c];
            }
        }
    }
    let mut determinant = sign;
    for k in 0..n {
        if k % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        determinant *= work[k * n + k];
    }
    if !determinant.is_finite() {
        return Err(format!("linalg {n}x{n} determinant is non-finite"));
    }
    Ok(determinant)
}

fn solve_output_value(
    inverse: &[f64],
    rhs: &[f64],
    output: SolveOutput,
) -> Result<f64, String> {
    let mut value = -0.0_f64;
    for j in 0..output.n {
        if j % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        value += inverse[output.row * output.n + j]
            * rhs[j * output.rhs_columns + output.column];
    }
    crate::program_ad_lifecycle::replay_checkpoint()?;
    Ok(value)
}
