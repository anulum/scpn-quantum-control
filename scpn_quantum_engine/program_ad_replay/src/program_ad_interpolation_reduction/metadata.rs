// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Interpolation metadata without grid materialization

struct InterpolationLayout<'a> {
    sample_count: usize,
    grid_label: &'a str,
    grid_count: usize,
    left: InterpolationBoundary,
    right: InterpolationBoundary,
    output_index: usize,
}

fn parse_interpolation_layout(
    effect_index: usize,
    operation: &str,
    source_length: usize,
) -> Result<InterpolationLayout<'_>, String> {
    replay_checkpoint()?;
    for _ in operation.as_bytes().chunks(256) {
        replay_checkpoint()?;
    }
    let mut fields = operation.split(':');
    let mut parts = [""; 12];
    for part in &mut parts {
        *part = fields.next().ok_or_else(|| {
            format!("effect {effect_index} interpolation operation metadata is malformed")
        })?;
    }
    if fields.next().is_some()
        || parts[0] != "interpolation"
        || parts[1] != "interp"
        || parts[2] != "samples"
        || parts[4] != "grid"
        || parts[6] != "left"
        || parts[8] != "right"
        || parts[10] != "out"
    {
        return Err(format!(
            "effect {effect_index} interpolation operation metadata is malformed"
        ));
    }
    let sample_count = parse_positive_usize(effect_index, "sample count", parts[3])?;
    let grid_count = validate_grid_metadata(effect_index, parts[5], sample_count, source_length)?;
    let left = parse_boundary(effect_index, "left", parts[7])?;
    let right = parse_boundary(effect_index, "right", parts[9])?;
    let output_index = parts[11].parse::<usize>().map_err(|_| {
        format!("effect {effect_index} interpolation output index must be non-negative")
    })?;
    if output_index >= sample_count {
        return Err(format!(
            "effect {effect_index} interpolation output index {output_index} is outside sample count {sample_count}"
        ));
    }
    Ok(InterpolationLayout {
        sample_count,
        grid_label: parts[5],
        grid_count,
        left,
        right,
        output_index,
    })
}

fn parse_positive_usize(effect_index: usize, field: &str, label: &str) -> Result<usize, String> {
    let value = label.parse::<usize>().map_err(|_| {
        format!("effect {effect_index} interpolation {field} must be a positive integer")
    })?;
    if value == 0 {
        return Err(format!(
            "effect {effect_index} interpolation {field} must be positive"
        ));
    }
    Ok(value)
}

fn validate_grid_metadata(
    effect_index: usize,
    label: &str,
    sample_count: usize,
    source_length: usize,
) -> Result<usize, String> {
    if label.is_empty() {
        return Err(format!(
            "effect {effect_index} interpolation grid metadata must not be empty"
        ));
    }
    let mut count = 1usize;
    for chunk in label.as_bytes().chunks(256) {
        replay_checkpoint()?;
        count = count.checked_add(chunk.iter().filter(|byte| **byte == b',').count())
            .ok_or_else(|| format!("effect {effect_index} interpolation grid length overflows"))?;
    }
    if count < 2 {
        return Err(format!(
            "effect {effect_index} interpolation grid requires at least two points"
        ));
    }
    let expected_size = interpolation_input_size(effect_index, sample_count, count)?;
    if source_length != expected_size {
        return Err(format!(
            "effect {effect_index} interpolation expects {expected_size} inputs, got {source_length}"
        ));
    }
    let mut previous: Option<f64> = None;
    for item in label.split(',') {
        replay_checkpoint()?;
        let value = item.parse::<f64>().map_err(|_| {
            format!("effect {effect_index} interpolation grid values must be finite floats")
        })?;
        if !value.is_finite() {
            return Err(format!(
                "effect {effect_index} interpolation grid values must be finite"
            ));
        }
        if previous.is_some_and(|previous| value <= previous) {
            return Err(format!(
                "effect {effect_index} interpolation grid must be strictly increasing"
            ));
        }
        previous = Some(value);
    }
    replay_checkpoint()?;
    Ok(count)
}

fn parse_boundary(
    effect_index: usize,
    role: &str,
    label: &str,
) -> Result<InterpolationBoundary, String> {
    if label == "none" {
        return Ok(InterpolationBoundary::Endpoint);
    }
    let value = label.parse::<f64>().map_err(|_| {
        format!("effect {effect_index} interpolation {role} boundary must be a finite float")
    })?;
    if !value.is_finite() {
        return Err(format!(
            "effect {effect_index} interpolation {role} boundary must be finite"
        ));
    }
    Ok(InterpolationBoundary::Static(value))
}

fn interpolation_input_size(
    effect_index: usize,
    sample_count: usize,
    grid_count: usize,
) -> Result<usize, String> {
    sample_count.checked_add(grid_count)
        .filter(|size| *size <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| format!("effect {effect_index} interpolation input bytes exceed native addressability"))
}
