// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD pseudoinverse metadata

fn parse_pinv_metadata(
    effect_index: usize,
    operation: &str,
) -> Result<(usize, usize, f64, usize, usize), String> {
    replay_checkpoint()?;
    for _ in operation.as_bytes().chunks(256) { replay_checkpoint()?; }
    let mut fields = operation.split(':');
    let mut parts = [""; 6];
    for part in &mut parts {
        *part = fields.next().ok_or_else(|| format!("effect {effect_index} pinv operation metadata is malformed"))?;
    }
    if fields.next().is_some() || parts[0] != "linalg" || parts[1] != "pinv" {
        return Err(format!(
            "effect {effect_index} pinv operation metadata is malformed"
        ));
    }
    let mut dimensions = parts[2].split('x');
    let shape = [dimensions.next().unwrap_or(""), dimensions.next().unwrap_or("")];
    if dimensions.next().is_some() || shape.iter().any(|field| field.is_empty()) {
        return Err(format!(
            "effect {effect_index} pinv shape metadata is malformed"
        ));
    }
    let rows = shape[0]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} pinv row metadata is malformed"))?;
    let cols = shape[1]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} pinv column metadata is malformed"))?;
    if rows == 0 || cols == 0 {
        return Err(format!(
            "effect {effect_index} pinv shape metadata must be positive"
        ));
    }
    let rcond = parts[3]
        .parse::<f64>()
        .map_err(|_| format!("effect {effect_index} pinv cutoff metadata is malformed"))?;
    if !rcond.is_finite() || rcond < 0.0 {
        return Err(format!(
            "effect {effect_index} pinv cutoff metadata must be finite and non-negative"
        ));
    }
    let output_row = parts[4]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} pinv output-row metadata is malformed"))?;
    let output_col = parts[5]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} pinv output-column metadata is malformed"))?;
    Ok((rows, cols, rcond, output_row, output_col))
}


fn validate_pinv_layout(
    effect_index: usize,
    operation: &str,
    input_count: usize,
    requires_adjoint: bool,
) -> Result<(usize, usize, f64, usize, usize), String> {
    let metadata = parse_pinv_metadata(effect_index, operation)?;
    let (rows, cols, _, output_row, output_col) = metadata;
    if !is_bounded_pinv_shape(rows, cols) {
        return Err(format!(
            "effect {effect_index} pinv Rust replay supports only rank-1, Nx2, and 2xN matrices"
        ));
    }
    let expected_size = matrix_entry_count(rows, cols)?;
    if requires_adjoint {
        matrix_entry_count(rows, rows)?;
        matrix_entry_count(cols, cols)?;
    }
    if input_count != expected_size {
        return Err(format!(
            "effect {effect_index} pinv requires {expected_size} flattened matrix operands"
        ));
    }
    if output_row >= cols || output_col >= rows {
        return Err(format!(
            "effect {effect_index} pinv output index is outside pseudoinverse shape"
        ));
    }
    Ok(metadata)
}
