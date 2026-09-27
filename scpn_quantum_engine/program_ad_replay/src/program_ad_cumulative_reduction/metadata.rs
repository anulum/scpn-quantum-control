// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD cumulative replay

// Borrowed metadata is shared by admission and actual kernel preparation.
struct CumulativeLayout<'a> {
    kind: CumulativeKind,
    shape_label: &'a str,
    rank: usize,
    source_size: usize,
    axis: CumulativeAxis,
    order: usize,
    output_index: usize,
    index_count: usize,
}

fn parse_cumulative_layout<'a>(
    effect_index: usize,
    operation: &'a str,
    input_count: usize,
) -> Result<CumulativeLayout<'a>, String> {
    for (index, _) in operation.bytes().enumerate() {
        if index.is_multiple_of(256) { replay_checkpoint()?; }
    }
    let mut fields = operation.split(':');
    let kind = CumulativeKind::from_label(fields.next().unwrap_or_default())
        .ok_or_else(|| format!("effect {effect_index} operation {operation} is not cumulative"))?;
    let mut source_shape = None;
    let mut axis = None;
    let mut order = None;
    let mut output_index = None;
    while let Some(field) = fields.next() {
        replay_checkpoint()?;
        let Some(raw_value) = fields.next() else {
            return Err(format!(
                "effect {effect_index} cumulative metadata field {field:?} must include a value"
            ));
        };
        match field {
            "shape" => {
                if source_shape.is_some() {
                    return Err(format!(
                        "effect {effect_index} cumulative shape metadata must appear once"
                    ));
                }
                source_shape = Some((raw_value, scan_cumulative_shape(effect_index, kind, raw_value)?));
            }
            "axis" => {
                if axis.is_some() {
                    return Err(format!(
                        "effect {effect_index} cumulative axis metadata must appear once"
                    ));
                }
                axis = Some(parse_axis_label(effect_index, raw_value)?);
            }
            "n" => {
                if kind != CumulativeKind::Diff {
                    return Err(format!(
                        "effect {effect_index} cumulative n metadata is valid only for diff"
                    ));
                }
                if order.is_some() {
                    return Err(format!(
                        "effect {effect_index} diff order metadata must appear once"
                    ));
                }
                order = Some(raw_value.parse::<usize>().map_err(|_| {
                    format!("effect {effect_index} diff order metadata must be non-negative")
                })?);
            }
            "out" => {
                if output_index.is_some() {
                    return Err(format!(
                        "effect {effect_index} cumulative output metadata must appear once"
                    ));
                }
                output_index = Some(raw_value.parse::<usize>().map_err(|_| {
                    format!("effect {effect_index} cumulative output index must be non-negative")
                })?);
            }
            "" => {
                return Err(format!(
                    "effect {effect_index} cumulative metadata field must be non-empty"
                ));
            }
            other => {
                return Err(format!(
                    "effect {effect_index} cumulative metadata field {other:?} is unsupported"
                ));
            }
        }
    }
    let (shape_label, (rank, source_size)) = source_shape.ok_or_else(|| {
        format!(
            "effect {effect_index} {} requires source shape metadata",
            kind.label()
        )
    })?;
    let axis = axis.ok_or_else(|| {
        format!(
            "effect {effect_index} {} requires axis metadata",
            kind.label()
        )
    })?;
    let axis = match (kind, axis) {
        (CumulativeKind::Diff, CumulativeAxis::Flat) => {
            return Err(format!(
                "effect {effect_index} diff requires a ranked static axis"
            ));
        }
        (_, CumulativeAxis::Axis(raw_axis)) => CumulativeAxis::Axis(normalise_axis(
            effect_index,
            kind,
            raw_axis,
            rank,
        )?),
        (_, CumulativeAxis::Flat) => CumulativeAxis::Flat,
    };
    let order = match kind {
        CumulativeKind::Diff => order
            .ok_or_else(|| format!("effect {effect_index} diff requires order metadata n:<int>"))?,
        CumulativeKind::Cumsum | CumulativeKind::Cumprod => {
            if order.is_some() {
                return Err(format!(
                    "effect {effect_index} {} does not accept order metadata",
                    kind.label()
                ));
            }
            0
        }
    };
    let output_index = output_index.ok_or_else(|| {
        format!("effect {effect_index} {} requires output index metadata", kind.label())
    })?;
    let (axis_size, stride) = match axis {
        CumulativeAxis::Flat => (source_size, 1),
        CumulativeAxis::Axis(axis) => cumulative_axis_extent(shape_label, axis)?,
    };
    let output_size = if kind == CumulativeKind::Diff {
        if order > axis_size {
            return Err(format!("effect {effect_index} diff order {order} exceeds axis length {axis_size}"));
        }
        (source_size / axis_size).checked_mul(axis_size - order)
            .ok_or_else(|| "cumulative shaped value size overflowed".to_owned())?
    } else { source_size };
    if output_index >= output_size {
        return Err(format!("effect {effect_index} {} output index {output_index} is outside output size {output_size}", kind.label()));
    }
    if source_size != input_count {
        return Err(format!("effect {effect_index} {} source shape {shape_label} expects {source_size} values, got {input_count}", kind.label()));
    }
    let index_count = if kind == CumulativeKind::Diff { order } else {
        match axis {
            CumulativeAxis::Flat => output_index,
            CumulativeAxis::Axis(_) => (output_index / stride) % axis_size,
        }
    }.checked_add(1).ok_or_else(|| "cumulative index count overflowed".to_owned())?;
    Ok(CumulativeLayout { kind, shape_label, rank, source_size, axis, order, output_index, index_count })
}

fn parse_axis_label(effect_index: usize, label: &str) -> Result<CumulativeAxis, String> {
    if label == "flat" {
        return Ok(CumulativeAxis::Flat);
    }
    let raw_axis = label.parse::<usize>().map_err(|_| {
        format!("effect {effect_index} cumulative axis metadata must be flat or non-negative")
    })?;
    Ok(CumulativeAxis::Axis(raw_axis))
}

fn normalise_axis(
    effect_index: usize,
    kind: CumulativeKind,
    axis: usize,
    rank: usize,
) -> Result<usize, String> {
    if axis < rank {
        Ok(axis)
    } else {
        Err(format!(
            "effect {effect_index} {} axis {axis} is outside rank {rank}",
            kind.label()
        ))
    }
}

fn scan_cumulative_shape(effect_index: usize, kind: CumulativeKind, label: &str) -> Result<(usize, usize), String> {
    if label.is_empty() {
        return Err(format!("effect {effect_index} {} source shape metadata must be non-empty", kind.label()));
    }
    let mut rank = 0usize;
    let mut size = 1usize;
    for entry in label.split('x') {
        replay_checkpoint()?;
        let dimension = entry.parse::<usize>().map_err(|_| {
            format!("effect {effect_index} {} source shape dimension must be non-negative", kind.label())
        })?;
        if dimension == 0 {
            return Err(format!("effect {effect_index} {} source shape dimensions must be positive", kind.label()));
        }
        size = size.checked_mul(dimension)
            .ok_or_else(|| "cumulative shaped value size overflowed".to_owned())?;
        rank = rank.checked_add(1)
            .ok_or_else(|| "cumulative shape rank overflowed".to_owned())?;
    }
    size.checked_mul(std::mem::size_of::<f64>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "cumulative source exceeds native addressable memory".to_owned())?;
    rank.checked_mul(std::mem::size_of::<usize>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "cumulative shape exceeds native addressable memory".to_owned())?;
    Ok((rank, size))
}

fn cumulative_axis_extent(label: &str, axis: usize) -> Result<(usize, usize), String> {
    let mut axis_size = 1usize;
    let mut stride = 1usize;
    for (index, entry) in label.split('x').enumerate() {
        replay_checkpoint()?;
        let dimension = entry.parse::<usize>().map_err(|_| "cumulative validated shape dimension is invalid".to_owned())?;
        if index == axis { axis_size = dimension; }
        if index > axis {
            stride = stride.checked_mul(dimension)
                .ok_or_else(|| "cumulative stride overflowed".to_owned())?;
        }
    }
    Ok((axis_size, stride))
}
