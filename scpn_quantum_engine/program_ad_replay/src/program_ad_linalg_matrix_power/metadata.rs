// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Matrix-power metadata admission

fn parse_matrix_power_metadata(
    effect_index: usize,
    operation: &str,
    input_count: usize,
) -> Result<MatrixPowerMetadata, String> {
    replay_checkpoint()?;
    for _ in operation.as_bytes().chunks(256) { replay_checkpoint()?; }
    let mut fields = operation.split(':');
    let mut parts = [""; 7];
    for part in &mut parts {
        *part = fields.next().ok_or_else(|| {
            format!("effect {effect_index} matrix_power operation metadata is malformed")
        })?;
    }
    if fields.next().is_some()
        || parts[0] != "linalg"
        || parts[1] != "matrix_power"
        || parts[3] != "power"
    {
        return Err(format!(
            "effect {effect_index} matrix_power operation metadata is malformed"
        ));
    }
    let size = parse_square_shape(effect_index, parts[2])?;
    let exponent = parts[4].parse::<i64>().map_err(|_| {
        format!("effect {effect_index} matrix_power exponent metadata is malformed")
    })?;
    let output_row = parts[5].parse::<usize>().map_err(|_| {
        format!("effect {effect_index} matrix_power output-row metadata is malformed")
    })?;
    let output_col = parts[6].parse::<usize>().map_err(|_| {
        format!("effect {effect_index} matrix_power output-column metadata is malformed")
    })?;
    if output_row >= size || output_col >= size {
        return Err(format!(
            "effect {effect_index} matrix_power output index is outside matrix shape"
        ));
    }
    let expected = size
        .checked_mul(size)
        .ok_or_else(|| format!("effect {effect_index} matrix_power shape size overflows"))?;
    if input_count != expected {
        return Err(format!(
            "effect {effect_index} matrix_power requires {expected} flattened matrix operands"
        ));
    }
    Ok(MatrixPowerMetadata {
        size,
        exponent,
        output_row,
        output_col,
    })
}

fn parse_square_shape(effect_index: usize, label: &str) -> Result<usize, String> {
    let Some((rows_label, cols_label)) = label.split_once('x') else {
        return Err(format!(
            "effect {effect_index} matrix_power shape metadata is malformed"
        ));
    };
    if cols_label.contains('x') {
        return Err(format!(
            "effect {effect_index} matrix_power shape metadata is malformed"
        ));
    }
    let rows = rows_label
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} matrix_power row metadata is malformed"))?;
    let cols = cols_label
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} matrix_power column metadata is malformed"))?;
    if rows == 0 || rows != cols {
        return Err(format!(
            "effect {effect_index} matrix_power requires non-empty square matrix metadata"
        ));
    }
    Ok(rows)
}
