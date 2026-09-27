// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD moment metadata and index admission

/// Parse static axis and correction metadata for `var` or `std` opcodes.
pub(crate) fn parse_moment_reduction_metadata(
    operation: &str,
    prefix: &str,
    rank: usize,
) -> Result<MomentReductionMetadata, String> {
    if operation == prefix {
        return Ok(MomentReductionMetadata {
            axis: None,
            correction: 0.0,
        });
    }
    replay_checkpoint()?;
    let Some(raw_metadata) = operation.strip_prefix(prefix).and_then(|rest| rest.strip_prefix(':')) else {
        return Err(format!(
            "{prefix} operation requires static metadata {prefix}[:axis:<int>][:ddof:<nonnegative>] or {prefix}[:axis:<int>][:correction:<nonnegative>]"
        ));
    };
    let mut fields = raw_metadata.split(':');
    let mut axis: Option<usize> = None;
    let mut correction: Option<f64> = None;
    while let Some(field) = fields.next() {
        replay_checkpoint()?;
        let Some(raw_value) = fields.next() else {
            return Err(format!(
                "{prefix} metadata field {field:?} must include a value"
            ));
        };
        match field {
            "axis" => {
                if axis.is_some() {
                    return Err(format!("{prefix} axis metadata must appear only once"));
                }
                let parsed_axis = raw_value
                    .parse::<isize>()
                    .map_err(|_| format!("{prefix} axis metadata must be an integer"))?;
                axis =
                    Some(normalise_static_axis(parsed_axis, rank).map_err(|reason| {
                        format!("{prefix} axis metadata is invalid: {reason}")
                    })?);
            }
            "ddof" | "correction" => {
                if correction.is_some() {
                    return Err(format!(
                        "{prefix} correction metadata must appear only once"
                    ));
                }
                let parsed_correction = raw_value.parse::<f64>().map_err(|_| {
                    format!("{prefix} correction metadata must be a finite non-negative scalar")
                })?;
                validate_correction_scalar(prefix, parsed_correction)?;
                correction = Some(parsed_correction);
            }
            "" => return Err(format!("{prefix} metadata field must be non-empty")),
            _ => {
                return Err(format!(
                    "{prefix} metadata field {field:?} is unsupported; expected axis, ddof, or correction"
                ));
            }
        }
    }
    Ok(MomentReductionMetadata {
        axis,
        correction: correction.unwrap_or(0.0),
    })
}

fn axis_reduction_shape(
    reduction: MomentReduction,
    source_shape: &[usize],
    axis: usize,
) -> Result<Vec<usize>, String> {
    if axis >= source_shape.len() {
        return Err(format!(
            "{} axis {axis} is outside rank {}",
            reduction.label(),
            source_shape.len()
        ));
    }
    index_without_axis(source_shape, axis)
}

fn normalise_static_axis(axis: isize, rank: usize) -> Result<usize, String> {
    if rank == 0 {
        return Err("rank must be positive".to_owned());
    }
    let rank_isize =
        isize::try_from(rank).map_err(|_| "rank exceeds axis metadata range".to_owned())?;
    let normalised = if axis < 0 { rank_isize + axis } else { axis };
    if normalised < 0 || normalised >= rank_isize {
        return Err(format!("axis {axis} is outside rank {rank}"));
    }
    usize::try_from(normalised).map_err(|_| "axis normalisation overflowed".to_owned())
}

fn shape_size(reduction: MomentReduction, shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for (index, dimension) in shape.iter().enumerate() {
        if index % 256 == 0 { replay_checkpoint()?; }
        if *dimension == 0 {
            return Err(format!(
                "{} shaped values must have non-zero dimensions",
                reduction.label()
            ));
        }
        size = size
            .checked_mul(*dimension)
            .ok_or_else(|| format!("{} shaped value size overflowed", reduction.label()))?;
    }
    Ok(size)
}

fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Result<Vec<usize>, String> {
    let mut index = filled_replay_buffer(shape.len(), 0usize)?;
    for (axis, dimension) in shape.iter().enumerate().rev() {
        if axis % 256 == 0 { replay_checkpoint()?; }
        if *dimension == 0 {
            return Err("moment index shape dimensions must be positive".to_owned());
        }
        index[axis] = flat_index % dimension;
        flat_index /= dimension;
    }
    Ok(index)
}

fn index_without_axis(index: &[usize], axis: usize) -> Result<Vec<usize>, String> {
    let mut result = reserve_replay_buffer(index.len() - usize::from(axis < index.len()))?;
    for (entry_axis, entry) in index.iter().enumerate() {
        if entry_axis % 256 == 0 { replay_checkpoint()?; }
        if entry_axis != axis { result.push(*entry); }
    }
    Ok(result)
}

fn ravel_index(
    reduction: MomentReduction,
    index: &[usize],
    shape: &[usize],
) -> Result<usize, String> {
    if index.len() != shape.len() {
        return Err(format!(
            "{} index rank {} does not match shape rank {}",
            reduction.label(),
            index.len(),
            shape.len()
        ));
    }
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (axis, (coordinate, dimension)) in index.iter().zip(shape.iter()).rev().enumerate() {
        if axis % 256 == 0 { replay_checkpoint()?; }
        if coordinate >= dimension {
            return Err(format!(
                "{} coordinate {coordinate} is outside dimension {dimension}",
                reduction.label()
            ));
        }
        flat = coordinate.checked_mul(stride)
            .and_then(|offset| flat.checked_add(offset))
            .ok_or_else(|| format!("{} ravel offset overflowed", reduction.label()))?;
        stride = stride
            .checked_mul(*dimension)
            .ok_or_else(|| format!("{} ravel stride overflowed", reduction.label()))?;
    }
    Ok(flat)
}


/// Validate a static group count/correction using the actual moment denominator rule.
pub(crate) fn validate_moment_group_size(effect_index: usize, label: &str, size: usize, correction: f64) -> Result<f64, String> {
    if size == 0 { return Err(format!("effect {effect_index} {label} requires non-empty values")); }
    validate_correction_scalar(label, correction)?;
    let count = size as f64;
    if correction >= count {
        return Err(format!("effect {effect_index} {label} correction must be less than reduction group size {count}, got {correction}"));
    }
    Ok(count - correction)
}
