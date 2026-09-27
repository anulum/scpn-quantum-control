// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD interpolation replay

//! Compact interpolation replay for bounded Program AD IR.
//!
//! The replay accepts Python-emitted scalar output opcodes for static-grid
//! `np.interp` operations. The differentiable source vector is flattened as
//! samples followed by interpolation `fp` values; grid and boundary policy stay
//! nondifferentiable static metadata.

use crate::program_ad_ir::{filled_replay_buffer, reserve_replay_buffer};
use crate::program_ad_lifecycle::replay_checkpoint;

#[derive(Clone, Copy, Debug, PartialEq)]
enum InterpolationBoundary {
    Endpoint,
    Static(f64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InterpolationRegion {
    Left,
    Right,
    Interior,
}

#[derive(Clone, Debug, PartialEq)]
struct InterpolationSpec {
    sample_count: usize,
    grid: Vec<f64>,
    left: InterpolationBoundary,
    right: InterpolationBoundary,
    output_index: usize,
}

/// Return whether an operation string names a compact interpolation primitive.
pub(crate) fn is_interpolation_operation(operation: &str) -> bool {
    operation.starts_with("interpolation:interp:")
}

/// Evaluate one compact interpolation output element.
pub(crate) fn interpolation_output_value(
    effect_index: usize,
    operation: &str,
    source_values: &[f64],
) -> Result<f64, String> {
    let spec = parse_interpolation_operation(effect_index, operation, source_values.len())?;
    validate_source(effect_index, &spec, source_values)?;
    let value = interpolation_value_for_output(effect_index, &spec, source_values)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "effect {effect_index} interpolation compact value must be finite"
        ))
    }
}

/// Build flattened sample/fp cotangent contribution for one compact output.
pub(crate) fn interpolation_output_cotangent(
    effect_index: usize,
    operation: &str,
    source_values: &[f64],
    cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} interpolation cotangent must be finite"
        ));
    }
    let spec = parse_interpolation_operation(effect_index, operation, source_values.len())?;
    validate_source(effect_index, &spec, source_values)?;
    interpolation_cotangent_for_output(effect_index, &spec, source_values, cotangent)
}

fn parse_interpolation_operation(
    effect_index: usize,
    operation: &str,
    source_length: usize,
) -> Result<InterpolationSpec, String> {
    let layout = parse_interpolation_layout(effect_index, operation, source_length)?;
    let mut grid = reserve_replay_buffer(layout.grid_count)?;
    for item in layout.grid_label.split(',') {
        replay_checkpoint()?;
        grid.push(item.parse::<f64>().map_err(|_| {
            format!("effect {effect_index} interpolation grid values must be finite floats")
        })?);
    }
    Ok(InterpolationSpec {
        sample_count: layout.sample_count,
        grid,
        left: layout.left,
        right: layout.right,
        output_index: layout.output_index,
    })
}

include!("program_ad_interpolation_reduction/metadata.rs");
include!("program_ad_interpolation_reduction/workspace.rs");

fn validate_source(
    effect_index: usize,
    spec: &InterpolationSpec,
    source_values: &[f64],
) -> Result<(), String> {
    replay_checkpoint()?;
    let expected_size = interpolation_input_size(effect_index, spec.sample_count, spec.grid.len())?;
    if source_values.len() != expected_size {
        return Err(format!(
            "effect {effect_index} interpolation expects {expected_size} inputs, got {}",
            source_values.len()
        ));
    }
    if spec.output_index >= spec.sample_count {
        return Err(format!(
            "effect {effect_index} interpolation output index {} is outside sample count {}",
            spec.output_index, spec.sample_count
        ));
    }
    for (index, value) in source_values.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        if !value.is_finite() {
            return Err(format!(
                "effect {effect_index} interpolation inputs must be finite"
            ));
        }
    }
    replay_checkpoint()?;
    Ok(())
}

fn interpolation_segment(
    effect_index: usize,
    sample: f64,
    grid: &[f64],
) -> Result<(InterpolationRegion, usize, f64), String> {
    if !sample.is_finite() {
        return Err(format!(
            "effect {effect_index} interpolation sample must be finite"
        ));
    }
    let mut lower = 0usize;
    let mut upper = grid.len();
    while lower < upper {
        replay_checkpoint()?;
        let middle = lower + (upper - lower) / 2;
        if grid[middle] == sample {
            return Err(format!(
                "effect {effect_index} interpolation samples must avoid grid knots"
            ));
        }
        if grid[middle] < sample {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    replay_checkpoint()?;
    if lower == 0 {
        return Ok((InterpolationRegion::Left, 0, 0.0));
    }
    if lower == grid.len() {
        return Ok((InterpolationRegion::Right, grid.len() - 1, 0.0));
    }
    let upper = lower;
    let segment = upper.saturating_sub(1);
    let lower_value = grid[segment];
    let upper_value = grid[segment + 1];
    let weight = (sample - lower_value) / (upper_value - lower_value);
    Ok((InterpolationRegion::Interior, segment, weight))
}

fn interpolation_value_for_output(
    effect_index: usize,
    spec: &InterpolationSpec,
    source_values: &[f64],
) -> Result<f64, String> {
    let sample = source_values[spec.output_index];
    let fp_values = &source_values[spec.sample_count..];
    let (region, segment, weight) = interpolation_segment(effect_index, sample, &spec.grid)?;
    match region {
        InterpolationRegion::Left => match spec.left {
            InterpolationBoundary::Endpoint => Ok(fp_values[0]),
            InterpolationBoundary::Static(value) => Ok(value),
        },
        InterpolationRegion::Right => match spec.right {
            InterpolationBoundary::Endpoint => Ok(fp_values[fp_values.len() - 1]),
            InterpolationBoundary::Static(value) => Ok(value),
        },
        InterpolationRegion::Interior => {
            Ok((1.0 - weight) * fp_values[segment] + weight * fp_values[segment + 1])
        }
    }
}

fn interpolation_cotangent_for_output(
    effect_index: usize,
    spec: &InterpolationSpec,
    source_values: &[f64],
    cotangent: f64,
) -> Result<Vec<f64>, String> {
    let sample = source_values[spec.output_index];
    let fp_values = &source_values[spec.sample_count..];
    let (region, segment, weight) = interpolation_segment(effect_index, sample, &spec.grid)?;
    let mut contribution = filled_replay_buffer(source_values.len(), 0.0_f64)?;
    match region {
        InterpolationRegion::Left => {
            if spec.left == InterpolationBoundary::Endpoint {
                contribution[spec.sample_count] += cotangent;
            }
        }
        InterpolationRegion::Right => {
            if spec.right == InterpolationBoundary::Endpoint {
                contribution[spec.sample_count + fp_values.len() - 1] += cotangent;
            }
        }
        InterpolationRegion::Interior => {
            let width = spec.grid[segment + 1] - spec.grid[segment];
            let slope = (fp_values[segment + 1] - fp_values[segment]) / width;
            contribution[spec.output_index] += cotangent * slope;
            contribution[spec.sample_count + segment] += cotangent * (1.0 - weight);
            contribution[spec.sample_count + segment + 1] += cotangent * weight;
        }
    }
    replay_checkpoint()?;
    for (index, value) in contribution.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        if !value.is_finite() {
            return Err(format!(
                "effect {effect_index} interpolation cotangent entries must be finite"
            ));
        }
    }
    replay_checkpoint()?;
    Ok(contribution)
}
