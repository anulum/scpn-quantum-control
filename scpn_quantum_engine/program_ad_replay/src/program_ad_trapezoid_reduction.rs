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

use crate::program_ad_ir::{filled_replay_buffer, reserve_replay_buffer};
use crate::program_ad_lifecycle::replay_checkpoint;

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
    let mut output = filled_replay_buffer(target_size, 0.0_f64)?;
    for (target_flat, output_value) in output.iter_mut().enumerate() {
        let target_index = unravel_index(target_flat, target_shape)?;
        for segment in 0..(axis_size - 1) {
            replay_checkpoint()?;
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
    validate_finite_values(effect_index, "value", &output)?;
    Ok(output)
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
    validate_finite_values(effect_index, "cotangent values", cotangent_values)?;
    let spec = parse_trapezoid_operation(effect_index, operation, source_shape)?;
    validate_grid(effect_index, &spec, source_shape, source_values.len())?;

    let target_shape = axis_reduction_shape(source_shape, spec.axis)?;
    let target_size = shape_size(&target_shape)?;
    if target_size != cotangent_values.len() {
        return Err(format!(
            "effect {effect_index} trapezoid axis cotangent shape must be {:?}",
            target_shape
        ));
    }

    let axis_size = source_shape[spec.axis];
    let mut contribution = filled_replay_buffer(source_values.len(), 0.0_f64)?;
    for (target_flat, cotangent) in cotangent_values.iter().enumerate() {
        let target_index = unravel_index(target_flat, &target_shape)?;
        for segment in 0..(axis_size - 1) {
            replay_checkpoint()?;
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
    validate_finite_values(effect_index, "adjoint contribution", &contribution)?;
    Ok(contribution)
}

include!("program_ad_trapezoid_reduction/metadata.rs");
include!("program_ad_trapezoid_reduction/workspace.rs");

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
    validate_finite_values(effect_index, "source values", source_values)?;
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
    let expected = axis_reduction_shape(source_shape, axis)?;
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

fn axis_reduction_shape(source_shape: &[usize], axis: usize) -> Result<Vec<usize>, String> {
    let mut shape = reserve_replay_buffer(source_shape.len().saturating_sub(1))?;
    for (index, dimension) in source_shape.iter().enumerate() {
        replay_checkpoint()?;
        if index != axis { shape.push(*dimension); }
    }
    Ok(shape)
}

fn source_flat_from_reduced_index(
    reduced_index: &[usize],
    source_shape: &[usize],
    axis: usize,
    axis_coordinate: usize,
) -> Result<usize, String> {
    if axis >= source_shape.len() || reduced_index.len() != source_shape.len() - 1 {
        return Err("trapezoid index rank does not match shape rank".to_owned());
    }
    let mut reduced_axis = 0usize;
    let mut flat = 0usize;
    for (source_axis, dimension) in source_shape.iter().enumerate() {
        replay_checkpoint()?;
        let coordinate = if source_axis == axis {
            axis_coordinate
        } else {
            let coordinate = reduced_index[reduced_axis];
            reduced_axis += 1;
            coordinate
        };
        if coordinate >= *dimension {
            return Err("trapezoid index is outside shape bounds".to_owned());
        }
        flat = flat.checked_mul(*dimension)
            .and_then(|value| value.checked_add(coordinate))
            .ok_or_else(|| "trapezoid flat index overflowed".to_owned())?;
    }
    Ok(flat)
}

fn shape_size(shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for dimension in shape {
        replay_checkpoint()?;
        if *dimension == 0 {
            return Err("trapezoid shaped values must have non-zero dimensions".to_owned());
        }
        size = size
            .checked_mul(*dimension)
            .filter(|size| *size <= isize::MAX as usize / std::mem::size_of::<f64>())
            .ok_or_else(|| "trapezoid shaped value size overflowed".to_owned())?;
    }
    Ok(size)
}

fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Result<Vec<usize>, String> {
    if shape.is_empty() {
        return Ok(Vec::new());
    }
    let mut index = filled_replay_buffer(shape.len(), 0usize)?;
    for axis in (0..shape.len()).rev() {
        replay_checkpoint()?;
        let dimension = shape[axis];
        index[axis] = flat_index % dimension;
        flat_index /= dimension;
    }
    Ok(index)
}

fn validate_finite_values(effect_index: usize, role: &str, values: &[f64]) -> Result<(), String> {
    for (index, value) in values.iter().enumerate() {
        if index % 256 == 0 { replay_checkpoint()?; }
        if !value.is_finite() {
            return Err(format!("effect {effect_index} trapezoid {role} must be finite"));
        }
    }
    replay_checkpoint()?;
    Ok(())
}
