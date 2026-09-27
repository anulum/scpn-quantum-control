// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD trapezoid metadata

#[derive(Clone, Copy)]
enum TrapezoidGridLabel<'a> {
    ConstantDx(f64),
    AxisGrid(&'a str),
    FullGrid(&'a str),
}

struct TrapezoidLayout<'a> {
    axis: usize,
    grid: TrapezoidGridLabel<'a>,
    grid_count: usize,
}

fn parse_trapezoid_operation(effect_index: usize, operation: &str, source_shape: &[usize]) -> Result<TrapezoidSpec, String> {
    let layout = parse_trapezoid_layout(effect_index, operation, source_shape)?;
    let grid = match layout.grid {
        TrapezoidGridLabel::ConstantDx(dx) => TrapezoidGrid::ConstantDx(dx),
        TrapezoidGridLabel::AxisGrid(value) => TrapezoidGrid::AxisGrid(parse_grid_values(effect_index, "x", value, layout.grid_count)?),
        TrapezoidGridLabel::FullGrid(value) => TrapezoidGrid::FullGrid(parse_grid_values(effect_index, "xfull", value, layout.grid_count)?),
    };
    Ok(TrapezoidSpec { axis: layout.axis, grid })
}

fn parse_trapezoid_layout<'a>(
    effect_index: usize,
    operation: &'a str,
    source_shape: &[usize],
) -> Result<TrapezoidLayout<'a>, String> {
    replay_checkpoint()?;
    if source_shape.is_empty() {
        return Err(format!("effect {effect_index} trapezoid requires ranked source values"));
    }
    let source_size = shape_size(source_shape)?;
    for _ in operation.as_bytes().chunks(256) { replay_checkpoint()?; }
    let mut fields = operation.split(':');
    if fields.next() != Some("trapezoid") {
        return Err(format!("effect {effect_index} operation {operation} is not a trapezoid reduction"));
    }
    let mut axis = None;
    let mut raw_grid = None;
    while let Some(field) = fields.next() {
        replay_checkpoint()?;
        let raw_value = fields.next().ok_or_else(|| {
            format!("effect {effect_index} trapezoid metadata field {field:?} must include a value")
        })?;
        match field {
            "axis" => {
                if axis.is_some() {
                    return Err(format!("effect {effect_index} trapezoid axis metadata must appear only once"));
                }
                let parsed_axis = raw_value.parse::<isize>().map_err(|_| {
                    format!("effect {effect_index} trapezoid axis metadata must be an integer")
                })?;
                axis = Some(normalise_static_axis(parsed_axis, source_shape.len()).map_err(|reason| {
                    format!("effect {effect_index} trapezoid axis metadata is invalid: {reason}")
                })?);
            }
            "dx" | "x" | "xfull" => {
                if raw_grid.is_some() {
                    return Err(format!("effect {effect_index} trapezoid metadata accepts only one of dx, x, or xfull"));
                }
                raw_grid = Some((field, raw_value));
            }
            "" => return Err(format!("effect {effect_index} trapezoid metadata field must be non-empty")),
            other => return Err(format!("effect {effect_index} trapezoid metadata field {other:?} is unsupported; expected axis, dx, x, or xfull")),
        }
    }
    let axis = match axis {
        Some(value) => value,
        None => source_shape.len().checked_sub(1).ok_or_else(|| {
            format!("effect {effect_index} trapezoid requires ranked source values")
        })?,
    };
    let grid = match raw_grid {
        None => TrapezoidGridLabel::ConstantDx(1.0),
        Some(("dx", value)) => TrapezoidGridLabel::ConstantDx(parse_finite_scalar(effect_index, "dx", value)?),
        Some(("x", value)) => TrapezoidGridLabel::AxisGrid(value),
        Some(("xfull", value)) => TrapezoidGridLabel::FullGrid(value),
        Some(_) => return Err(format!("effect {effect_index} trapezoid grid metadata is unsupported")),
    };
    let axis_size = source_shape[axis];
    if axis_size < 2 {
        return Err(format!("effect {effect_index} trapezoid integration axis size must be at least 2"));
    }
    let grid_count = match grid {
        TrapezoidGridLabel::ConstantDx(_) => 0,
        TrapezoidGridLabel::AxisGrid(value) => validate_grid_values(effect_index, "x", value, axis_size)?,
        TrapezoidGridLabel::FullGrid(value) => validate_grid_values(effect_index, "xfull", value, source_size)?,
    };
    Ok(TrapezoidLayout { axis, grid, grid_count })
}

fn parse_finite_scalar(effect_index: usize, field: &str, raw_value: &str) -> Result<f64, String> {
    let value = raw_value.parse::<f64>().map_err(|_| {
        format!("effect {effect_index} trapezoid {field} metadata must be a finite float")
    })?;
    if !value.is_finite() {
        return Err(format!("effect {effect_index} trapezoid {field} metadata must be finite"));
    }
    Ok(value)
}

fn validate_grid_values(effect_index: usize, field: &str, raw_value: &str, expected: usize) -> Result<usize, String> {
    if raw_value.is_empty() {
        return Err(format!("effect {effect_index} trapezoid {field} metadata must contain comma-separated floats"));
    }
    let mut count = 1usize;
    for chunk in raw_value.as_bytes().chunks(256) {
        replay_checkpoint()?;
        count = count.checked_add(chunk.iter().filter(|byte| **byte == b',').count())
            .ok_or_else(|| format!("effect {effect_index} trapezoid grid length overflows"))?;
    }
    // Validate literals before the length error, preserving dynamic-grid diagnostics.
    for item in raw_value.split(',') {
        replay_checkpoint()?;
        parse_finite_scalar(effect_index, field, item)?;
    }
    if count != expected {
        let role = if field == "x" { "integration axis size" } else { "source value count" };
        return Err(format!("effect {effect_index} trapezoid {field} metadata length must match {role} {expected}, got {count}"));
    }
    count.checked_mul(std::mem::size_of::<f64>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "trapezoid grid exceeds native addressable memory".to_owned())?;
    Ok(count)
}

fn parse_grid_values(effect_index: usize, field: &str, raw_value: &str, expected: usize) -> Result<Vec<f64>, String> {
    let count = validate_grid_values(effect_index, field, raw_value, expected)?;
    let mut values = reserve_replay_buffer(count)?;
    for item in raw_value.split(',') {
        replay_checkpoint()?;
        values.push(parse_finite_scalar(effect_index, field, item)?);
    }
    replay_checkpoint()?;
    Ok(values)
}
