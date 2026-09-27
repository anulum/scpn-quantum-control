// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD pseudoinverse matrix helpers

fn matmul(
    left: &[f64], left_rows: usize, left_cols: usize,
    right: &[f64], right_cols: usize,
) -> Result<Vec<f64>, String> {
    if left.len() != matrix_entry_count(left_rows, left_cols)? || right.len() != matrix_entry_count(left_cols, right_cols)? {
        return Err("pinv matrix product operand shapes do not match".to_owned());
    }
    let mut result = filled_replay_buffer(matrix_entry_count(left_rows, right_cols)?, 0.0_f64)?;
    for row in 0..left_rows {
        replay_checkpoint()?;
        for col in 0..right_cols {
            replay_checkpoint()?;
            let mut value = -0.0_f64;
            for inner in 0..left_cols {
                if inner % 256 == 0 { replay_checkpoint()?; }
                value += left[row * left_cols + inner] * right[inner * right_cols + col];
            }
            result[row * right_cols + col] = value;
        }
    }
    replay_checkpoint()?;
    Ok(result)
}

fn transpose(matrix: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>, String> {
    let size = matrix_entry_count(rows, cols)?;
    if matrix.len() != size { return Err("pinv transpose shape does not match values".to_owned()); }
    let mut result = filled_replay_buffer(size, 0.0_f64)?;
    for row in 0..rows {
        replay_checkpoint()?;
        for col in 0..cols {
            if col % 256 == 0 { replay_checkpoint()?; }
            result[col * rows + row] = matrix[row * cols + col];
        }
    }
    Ok(result)
}

fn identity(size: usize) -> Result<Vec<f64>, String> {
    let mut result = filled_replay_buffer(matrix_entry_count(size, size)?, 0.0_f64)?;
    for index in 0..size {
        if index % 256 == 0 { replay_checkpoint()?; }
        result[index * size + index] = 1.0;
    }
    Ok(result)
}

fn subtract(left: &[f64], right: &[f64]) -> Result<Vec<f64>, String> {
    if left.len() != right.len() { return Err("pinv subtraction operand lengths do not match".to_owned()); }
    let mut result = reserve_replay_buffer(left.len())?;
    for (index, (left_value, right_value)) in left.iter().zip(right).enumerate() {
        if index % 256 == 0 { replay_checkpoint()?; }
        result.push(left_value - right_value);
    }
    Ok(result)
}

fn matrix_entry_count(rows: usize, cols: usize) -> Result<usize, String> {
    rows.checked_mul(cols)
        .filter(|entries| *entries <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| "pinv matrix bytes exceed native addressability".to_owned())
}

fn validate_finite_values(effect_index: usize, role: &str, values: &[f64]) -> Result<(), String> {
    for (index, value) in values.iter().enumerate() {
        if index % 256 == 0 { replay_checkpoint()?; }
        if !value.is_finite() { return Err(format!("effect {effect_index} pinv {role} must be finite")); }
    }
    replay_checkpoint()?;
    Ok(())
}
