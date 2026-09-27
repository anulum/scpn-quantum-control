// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD stencil metadata

#[derive(Clone, Copy)]
enum StencilSpacingLabel<'a> { Scalar(f64), Coordinates(&'a str) }

struct StencilLayout<'a> {
    shape_label: &'a str,
    rank: usize,
    source_size: usize,
    axis_size: usize,
    axis: usize,
    edge_order: usize,
    spacing: StencilSpacingLabel<'a>,
    output_index: usize,
}

fn parse_stencil_operation(effect_index: usize, operation: &str, source_length: usize) -> Result<StencilSpec, String> {
    let layout = parse_stencil_layout(effect_index, operation, source_length)?;
    let source_shape = parse_shape_label(effect_index, layout.shape_label)?;
    let spacing = match layout.spacing {
        StencilSpacingLabel::Scalar(value) => StencilSpacing::Scalar(value),
        StencilSpacingLabel::Coordinates(raw) => {
            let mut coordinates = reserve_replay_buffer(layout.axis_size)?;
            for value in raw.split(',') {
                replay_checkpoint()?;
                coordinates.push(parse_stencil_coordinate(effect_index, value)?);
            }
            StencilSpacing::Coordinates(coordinates)
        },
    };
    Ok(StencilSpec { source_shape, axis: layout.axis, edge_order: layout.edge_order,
        spacing, output_index: layout.output_index })
}

fn parse_stencil_layout<'a>(effect_index: usize, operation: &'a str, source_length: usize) -> Result<StencilLayout<'a>, String> {
    replay_checkpoint()?;
    for _ in operation.as_bytes().chunks(256) { replay_checkpoint()?; }
    let mut fields = operation.split(':');
    let mut parts = [""; 12];
    for part in &mut parts {
        *part = fields.next().ok_or_else(|| {
            format!("effect {effect_index} stencil gradient operation metadata is malformed")
        })?;
    }
    if fields.next().is_some()
        || parts[0] != "stencil"
        || parts[1] != "gradient"
        || parts[2] != "shape"
        || parts[4] != "axis"
        || parts[6] != "edge"
        || parts[8] != "spacing"
        || parts[10] != "out"
    {
        return Err(format!(
            "effect {effect_index} stencil gradient operation metadata is malformed"
        ));
    }
    let (rank, source_size) = scan_stencil_shape(effect_index, parts[3])?;
    let axis = parts[5]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} stencil gradient axis must be non-negative"))?;
    let edge_order = parts[7]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} stencil gradient edge order must be 1 or 2"))?;
    if edge_order != 1 && edge_order != 2 {
        return Err(format!(
            "effect {effect_index} stencil gradient edge order must be 1 or 2"
        ));
    }
    if source_size != source_length {
        return Err(format!("effect {effect_index} stencil gradient expects {source_size} inputs, got {source_length}"));
    }
    let axis_size = stencil_axis_extent(effect_index, parts[3], axis, rank)?;
    if axis_size < edge_order + 1 {
        return Err(format!("effect {effect_index} stencil gradient edge order {edge_order} requires at least {} samples", edge_order + 1));
    }
    let spacing = parse_spacing_layout(effect_index, parts[9], axis_size)?;
    let output_index = parts[11].parse::<usize>().map_err(|_| {
        format!("effect {effect_index} stencil gradient output index must be non-negative")
    })?;
    if output_index >= source_size {
        return Err(format!("effect {effect_index} stencil gradient output index is outside source shape"));
    }
    Ok(StencilLayout {
        shape_label: parts[3], rank, source_size, axis_size,
        axis,
        edge_order,
        spacing,
        output_index,
    })
}

fn parse_shape_label(effect_index: usize, label: &str) -> Result<Vec<usize>, String> {
    if label.is_empty() {
        return Err(format!(
            "effect {effect_index} stencil gradient shape metadata must not be empty"
        ));
    }
    let count = metadata_entry_count(effect_index, label, b'x')?;
    let mut shape = reserve_replay_buffer(count)?;
    for part in label.split('x') {
        replay_checkpoint()?;
        let dimension = parse_stencil_dimension(effect_index, part)?;
        shape.push(dimension);
    }
    Ok(shape)
}

fn parse_spacing_layout<'a>(effect_index: usize, label: &'a str, axis_size: usize) -> Result<StencilSpacingLabel<'a>, String> {
    if let Some(raw) = label.strip_prefix("scalar=") {
        let value = raw.parse::<f64>().map_err(|_| {
            format!("effect {effect_index} stencil gradient scalar spacing must be finite")
        })?;
        if !value.is_finite() || value == 0.0 {
            return Err(format!(
                "effect {effect_index} stencil gradient scalar spacing must be finite and non-zero"
            ));
        }
        return Ok(StencilSpacingLabel::Scalar(value));
    }
    let Some(raw) = label.strip_prefix("coordinates=") else {
        return Err(format!(
            "effect {effect_index} stencil gradient spacing metadata is malformed"
        ));
    };
    if raw.is_empty() {
        return Err(format!(
            "effect {effect_index} stencil gradient coordinates must not be empty"
        ));
    }
    let count = metadata_entry_count(effect_index, raw, b',')?;
    if count != axis_size {
        return Err(format!("effect {effect_index} stencil gradient coordinates must match axis size"));
    }
    let mut previous: Option<f64> = None;
    let mut direction = None;
    for part in raw.split(',') {
        replay_checkpoint()?;
        let value = parse_stencil_coordinate(effect_index, part)?;
        if let Some(previous) = previous {
            let increasing = value > previous;
            if value == previous || direction.is_some_and(|prior| prior != increasing) {
                return Err(format!("effect {effect_index} stencil gradient coordinates must be strictly monotonic"));
            }
            direction = Some(increasing);
        }
        previous = Some(value);
    }
    replay_checkpoint()?;
    Ok(StencilSpacingLabel::Coordinates(raw))
}


fn parse_stencil_coordinate(effect_index: usize, raw: &str) -> Result<f64, String> {
    let value = raw.parse::<f64>().map_err(|_| format!("effect {effect_index} stencil gradient coordinates must be finite"))?;
    if !value.is_finite() { return Err(format!("effect {effect_index} stencil gradient coordinates must be finite with at least two samples")); }
    Ok(value)
}

fn parse_stencil_dimension(effect_index: usize, raw: &str) -> Result<usize, String> {
    let dimension = raw.parse::<usize>().map_err(|_| format!("effect {effect_index} stencil gradient shape must be positive integers"))?;
    if dimension == 0 { return Err(format!("effect {effect_index} stencil gradient shape dimensions must be positive")); }
    Ok(dimension)
}

fn scan_stencil_shape(effect_index: usize, label: &str) -> Result<(usize, usize), String> {
    if label.is_empty() { return Err(format!("effect {effect_index} stencil gradient shape metadata must not be empty")); }
    let rank = metadata_entry_count(effect_index, label, b'x')?;
    rank.checked_mul(std::mem::size_of::<usize>()).filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "stencil shape exceeds native addressable memory".to_owned())?;
    let mut size = 1usize;
    for part in label.split('x') {
        replay_checkpoint()?;
        size = size.checked_mul(parse_stencil_dimension(effect_index, part)?)
            .filter(|count| *count <= isize::MAX as usize / std::mem::size_of::<f64>())
            .ok_or_else(|| format!("effect {effect_index} stencil gradient shape size overflowed"))?;
    }
    Ok((rank, size))
}

fn stencil_axis_extent(effect_index: usize, label: &str, axis: usize, rank: usize) -> Result<usize, String> {
    if axis >= rank { return Err(format!("effect {effect_index} stencil gradient axis is outside source rank")); }
    for (index, value) in label.split('x').enumerate() {
        replay_checkpoint()?;
        if index == axis { return parse_stencil_dimension(effect_index, value); }
    }
    Err(format!("effect {effect_index} stencil gradient axis is outside source rank"))
}

fn metadata_entry_count(effect_index: usize, label: &str, separator: u8) -> Result<usize, String> {
    let mut count = 1usize;
    for chunk in label.as_bytes().chunks(256) {
        replay_checkpoint()?;
        count = count.checked_add(chunk.iter().filter(|byte| **byte == separator).count())
            .ok_or_else(|| format!("effect {effect_index} stencil metadata length overflows"))?;
    }
    Ok(count)
}
