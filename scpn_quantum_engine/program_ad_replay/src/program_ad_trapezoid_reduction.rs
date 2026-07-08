// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD trapezoid reduction replay

//! Static-grid trapezoidal integration replay for bounded Program AD IR.
//!
//! The replay accepts compact `trapezoid` opcodes with static `axis`, `dx`,
//! one-dimensional `x`, or full-shape `xfull` metadata. Reverse replay
//! propagates cotangents only into the integrated samples; grid metadata is
//! treated as nondifferentiable static metadata and is validated fail-closed.

#[derive(Clone, Debug, PartialEq)]
enum TrapezoidGrid {
    ConstantDx(f64),
    AxisGrid(Vec<f64>),
    FullGrid(Vec<f64>),
}

#[derive(Clone, Debug, PartialEq)]
struct TrapezoidSpec {
    axis: usize,
    grid: TrapezoidGrid,
}

impl TrapezoidSpec {
    fn width(
        &self,
        segment: usize,
        left_flat: usize,
        right_flat: usize,
        effect_index: usize,
    ) -> Result<f64, String> {
        let width = match &self.grid {
            TrapezoidGrid::ConstantDx(dx) => *dx,
            TrapezoidGrid::AxisGrid(grid) => grid[segment + 1] - grid[segment],
            TrapezoidGrid::FullGrid(grid) => grid[right_flat] - grid[left_flat],
        };
        if width.is_finite() {
            Ok(width)
        } else {
            Err(format!(
                "effect {effect_index} trapezoid segment width must be finite"
            ))
        }
    }
}

/// Return whether an operation string names a compact trapezoid reduction.
pub(crate) fn is_trapezoid_operation(operation: &str) -> bool {
    operation == "trapezoid" || operation.starts_with("trapezoid:")
}

/// Evaluate static-grid trapezoidal integration over one source axis.
pub(crate) fn trapezoid_values(
    effect_index: usize,
    operation: &str,
    source_shape: &[usize],
    target_shape: &[usize],
    source_values: &[f64],
) -> Result<Vec<f64>, String> {
    validate_source(effect_index, source_shape, source_values)?;
    let spec = parse_trapezoid_operation(effect_index, operation, source_shape)?;
    validate_grid(effect_index, &spec, source_shape, source_values.len())?;
    validate_target_shape(effect_index, source_shape, spec.axis, target_shape)?;

    let target_size = shape_size(target_shape)?;
    let axis_size = source_shape[spec.axis];
    let mut output = vec![0.0_f64; target_size];
    for (target_flat, output_value) in output.iter_mut().enumerate() {
        let target_index = unravel_index(target_flat, target_shape);
        for segment in 0..(axis_size - 1) {
            let left_flat =
                source_flat_from_reduced_index(&target_index, source_shape, spec.axis, segment)?;
            let right_flat = source_flat_from_reduced_index(
                &target_index,
                source_shape,
                spec.axis,
                segment + 1,
            )?;
            let width = spec.width(segment, left_flat, right_flat, effect_index)?;
            *output_value += 0.5 * width * (source_values[left_flat] + source_values[right_flat]);
        }
    }
    if output.iter().all(|value| value.is_finite()) {
        Ok(output)
    } else {
        Err(format!(
            "effect {effect_index} trapezoid value must be finite"
        ))
    }
}

/// Build the source-shaped cotangent for static-grid trapezoidal integration.
pub(crate) fn trapezoid_cotangent(
    effect_index: usize,
    operation: &str,
    source_shape: &[usize],
    cotangent_values: &[f64],
    source_values: &[f64],
) -> Result<Vec<f64>, String> {
    validate_source(effect_index, source_shape, source_values)?;
    if cotangent_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} trapezoid cotangent values must be finite"
        ));
    }
    let spec = parse_trapezoid_operation(effect_index, operation, source_shape)?;
    validate_grid(effect_index, &spec, source_shape, source_values.len())?;

    let target_shape = axis_reduction_shape(source_shape, spec.axis);
    let target_size = shape_size(&target_shape)?;
    if target_size != cotangent_values.len() {
        return Err(format!(
            "effect {effect_index} trapezoid axis cotangent shape must be {:?}",
            target_shape
        ));
    }

    let axis_size = source_shape[spec.axis];
    let mut contribution = vec![0.0_f64; source_values.len()];
    for (target_flat, cotangent) in cotangent_values.iter().enumerate() {
        let target_index = unravel_index(target_flat, &target_shape);
        for segment in 0..(axis_size - 1) {
            let left_flat =
                source_flat_from_reduced_index(&target_index, source_shape, spec.axis, segment)?;
            let right_flat = source_flat_from_reduced_index(
                &target_index,
                source_shape,
                spec.axis,
                segment + 1,
            )?;
            let contribution_value =
                0.5 * spec.width(segment, left_flat, right_flat, effect_index)? * cotangent;
            contribution[left_flat] += contribution_value;
            contribution[right_flat] += contribution_value;
        }
    }
    if contribution.iter().all(|value| value.is_finite()) {
        Ok(contribution)
    } else {
        Err(format!(
            "effect {effect_index} trapezoid adjoint contribution must be finite"
        ))
    }
}

fn parse_trapezoid_operation(
    effect_index: usize,
    operation: &str,
    source_shape: &[usize],
) -> Result<TrapezoidSpec, String> {
    let fields = operation.split(':').collect::<Vec<&str>>();
    if fields.first().copied() != Some("trapezoid") {
        return Err(format!(
            "effect {effect_index} operation {operation} is not a trapezoid reduction"
        ));
    }
    let mut axis = None;
    let mut grid = None;
    let mut index = 1usize;
    while index < fields.len() {
        let field = fields[index];
        let Some(raw_value) = fields.get(index + 1).copied() else {
            return Err(format!(
                "effect {effect_index} trapezoid metadata field {field:?} must include a value"
            ));
        };
        match field {
            "axis" => {
                if axis.is_some() {
                    return Err(format!(
                        "effect {effect_index} trapezoid axis metadata must appear only once"
                    ));
                }
                let parsed_axis = raw_value.parse::<isize>().map_err(|_| {
                    format!("effect {effect_index} trapezoid axis metadata must be an integer")
                })?;
                axis = Some(
                    normalise_static_axis(parsed_axis, source_shape.len()).map_err(|reason| {
                        format!(
                            "effect {effect_index} trapezoid axis metadata is invalid: {reason}"
                        )
                    })?,
                );
            }
            "dx" => {
                ensure_single_grid_metadata(effect_index, &grid)?;
                let dx = parse_finite_scalar(effect_index, "dx", raw_value)?;
                grid = Some(TrapezoidGrid::ConstantDx(dx));
            }
            "x" => {
                ensure_single_grid_metadata(effect_index, &grid)?;
                grid = Some(TrapezoidGrid::AxisGrid(parse_grid_values(
                    effect_index,
                    "x",
                    raw_value,
                )?));
            }
            "xfull" => {
                ensure_single_grid_metadata(effect_index, &grid)?;
                grid = Some(TrapezoidGrid::FullGrid(parse_grid_values(
                    effect_index,
                    "xfull",
                    raw_value,
                )?));
            }
            "" => {
                return Err(format!(
                    "effect {effect_index} trapezoid metadata field must be non-empty"
                ));
            }
            other => {
                return Err(format!(
                    "effect {effect_index} trapezoid metadata field {other:?} is unsupported; expected axis, dx, x, or xfull"
                ));
            }
        }
        index += 2;
    }
    let axis = match axis {
        Some(value) => value,
        None => source_shape.len().checked_sub(1).ok_or_else(|| {
            format!("effect {effect_index} trapezoid requires ranked source values")
        })?,
    };
    Ok(TrapezoidSpec {
        axis,
        grid: grid.unwrap_or(TrapezoidGrid::ConstantDx(1.0)),
    })
}

fn ensure_single_grid_metadata(
    effect_index: usize,
    grid: &Option<TrapezoidGrid>,
) -> Result<(), String> {
    if grid.is_some() {
        return Err(format!(
            "effect {effect_index} trapezoid metadata accepts only one of dx, x, or xfull"
        ));
    }
    Ok(())
}

fn parse_finite_scalar(effect_index: usize, field: &str, raw_value: &str) -> Result<f64, String> {
    let value = raw_value.parse::<f64>().map_err(|_| {
        format!("effect {effect_index} trapezoid {field} metadata must be a finite float")
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "effect {effect_index} trapezoid {field} metadata must be finite"
        ))
    }
}

fn parse_grid_values(
    effect_index: usize,
    field: &str,
    raw_value: &str,
) -> Result<Vec<f64>, String> {
    if raw_value.is_empty() {
        return Err(format!(
            "effect {effect_index} trapezoid {field} metadata must contain comma-separated floats"
        ));
    }
    let values = raw_value
        .split(',')
        .map(|item| parse_finite_scalar(effect_index, field, item))
        .collect::<Result<Vec<f64>, String>>()?;
    if values.is_empty() {
        return Err(format!(
            "effect {effect_index} trapezoid {field} metadata must contain at least one value"
        ));
    }
    Ok(values)
}

fn validate_source(
    effect_index: usize,
    source_shape: &[usize],
    source_values: &[f64],
) -> Result<(), String> {
    if source_shape.is_empty() {
        return Err(format!(
            "effect {effect_index} trapezoid requires ranked source values"
        ));
    }
    let expected = shape_size(source_shape)?;
    if expected != source_values.len() {
        return Err(format!(
            "effect {effect_index} trapezoid source shape {:?} expects {expected} values, got {}",
            source_shape,
            source_values.len()
        ));
    }
    if source_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} trapezoid source values must be finite"
        ));
    }
    Ok(())
}

fn validate_grid(
    effect_index: usize,
    spec: &TrapezoidSpec,
    source_shape: &[usize],
    source_size: usize,
) -> Result<(), String> {
    let axis_size = source_shape[spec.axis];
    if axis_size < 2 {
        return Err(format!(
            "effect {effect_index} trapezoid integration axis size must be at least 2"
        ));
    }
    match &spec.grid {
        TrapezoidGrid::ConstantDx(dx) => {
            if !dx.is_finite() {
                return Err(format!(
                    "effect {effect_index} trapezoid dx metadata must be finite"
                ));
            }
        }
        TrapezoidGrid::AxisGrid(grid) => {
            if grid.len() != axis_size {
                return Err(format!(
                    "effect {effect_index} trapezoid x metadata length must match integration axis size {axis_size}, got {}",
                    grid.len()
                ));
            }
        }
        TrapezoidGrid::FullGrid(grid) => {
            if grid.len() != source_size {
                return Err(format!(
                    "effect {effect_index} trapezoid xfull metadata length must match source value count {source_size}, got {}",
                    grid.len()
                ));
            }
        }
    }
    Ok(())
}

fn validate_target_shape(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    target_shape: &[usize],
) -> Result<(), String> {
    let expected = axis_reduction_shape(source_shape, axis);
    if expected == target_shape {
        Ok(())
    } else {
        Err(format!(
            "effect {effect_index} trapezoid target shape must be {:?}, got {:?}",
            expected, target_shape
        ))
    }
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

fn axis_reduction_shape(source_shape: &[usize], axis: usize) -> Vec<usize> {
    source_shape
        .iter()
        .enumerate()
        .filter_map(|(index, dimension)| (index != axis).then_some(*dimension))
        .collect()
}

fn source_flat_from_reduced_index(
    reduced_index: &[usize],
    source_shape: &[usize],
    axis: usize,
    axis_coordinate: usize,
) -> Result<usize, String> {
    let mut source_index = Vec::with_capacity(source_shape.len());
    let mut reduced_axis = 0usize;
    for source_axis in 0..source_shape.len() {
        if source_axis == axis {
            source_index.push(axis_coordinate);
        } else {
            source_index.push(reduced_index[reduced_axis]);
            reduced_axis += 1;
        }
    }
    ravel_index(&source_index, source_shape)
}

fn shape_size(shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for dimension in shape {
        if *dimension == 0 {
            return Err("trapezoid shaped values must have non-zero dimensions".to_owned());
        }
        size = size
            .checked_mul(*dimension)
            .ok_or_else(|| "trapezoid shaped value size overflowed".to_owned())?;
    }
    Ok(size)
}

fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut index = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dimension = shape[axis];
        index[axis] = flat_index % dimension;
        flat_index /= dimension;
    }
    index
}

fn ravel_index(index: &[usize], shape: &[usize]) -> Result<usize, String> {
    if index.len() != shape.len() {
        return Err("trapezoid index rank does not match shape rank".to_owned());
    }
    let mut flat = 0usize;
    for (coordinate, dimension) in index.iter().zip(shape.iter()) {
        if coordinate >= dimension {
            return Err("trapezoid index is outside shape bounds".to_owned());
        }
        flat = flat
            .checked_mul(*dimension)
            .and_then(|value| value.checked_add(*coordinate))
            .ok_or_else(|| "trapezoid flat index overflowed".to_owned())?;
    }
    Ok(flat)
}
