// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD stencil replay

//! Compact static `np.gradient` replay for bounded Program AD IR.
//!
//! The replay accepts Python-emitted scalar output opcodes for one static
//! gradient axis. Reverse replay returns the flattened source cotangent
//! contribution for one compact output element and treats shape, axis, edge
//! order, and spacing metadata as nondifferentiable static metadata.

use crate::program_ad_ir::{filled_replay_buffer, reserve_replay_buffer};
use crate::program_ad_lifecycle::replay_checkpoint;

#[derive(Clone, Debug, PartialEq)]
enum StencilSpacing {
    Scalar(f64),
    Coordinates(Vec<f64>),
}

#[derive(Clone, Debug, PartialEq)]
struct StencilSpec {
    source_shape: Vec<usize>,
    axis: usize,
    edge_order: usize,
    spacing: StencilSpacing,
    output_index: usize,
}

/// Return whether an operation string names a compact stencil primitive.
pub(crate) fn is_stencil_operation(operation: &str) -> bool {
    operation.starts_with("stencil:gradient:")
}

/// Evaluate one compact static-gradient output element.
pub(crate) fn stencil_output_value(
    effect_index: usize,
    operation: &str,
    source_values: &[f64],
) -> Result<f64, String> {
    let spec = parse_stencil_operation(effect_index, operation, source_values.len())?;
    validate_source(effect_index, &spec, source_values)?;
    let mut target_index = unravel_index(effect_index, &spec, spec.output_index)?;
    let position = target_index[spec.axis];
    let coefficients = gradient_coefficients(effect_index, &spec, position)?;
    let mut value = -0.0_f64;
    for &(axis_index, coefficient) in coefficients.as_slice() {
        replay_checkpoint()?;
        target_index[spec.axis] = axis_index;
        let source_index = ravel_index(&spec.source_shape, &target_index)?;
        value += coefficient * source_values[source_index];
    }
    replay_checkpoint()?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "effect {effect_index} stencil gradient compact value must be finite"
        ))
    }
}

/// Build flattened source cotangent contribution for one static-gradient output.
pub(crate) fn stencil_output_cotangent(
    effect_index: usize,
    operation: &str,
    source_values: &[f64],
    cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} stencil gradient cotangent must be finite"
        ));
    }
    let spec = parse_stencil_operation(effect_index, operation, source_values.len())?;
    validate_source(effect_index, &spec, source_values)?;
    let mut target_index = unravel_index(effect_index, &spec, spec.output_index)?;
    let position = target_index[spec.axis];
    let mut contribution = filled_replay_buffer(source_values.len(), 0.0_f64)?;
    let coefficients = gradient_coefficients(effect_index, &spec, position)?;
    for &(axis_index, coefficient) in coefficients.as_slice() {
        replay_checkpoint()?;
        target_index[spec.axis] = axis_index;
        contribution[ravel_index(&spec.source_shape, &target_index)?] += cotangent * coefficient;
    }
    for (index, value) in contribution.iter().enumerate() {
        if index % 256 == 0 { replay_checkpoint()?; }
        if !value.is_finite() {
            return Err(format!(
                "effect {effect_index} stencil gradient compact adjoint contribution must be finite"
            ));
        }
    }
    replay_checkpoint()?;
    Ok(contribution)
}

include!("program_ad_stencil_reduction/metadata.rs");
include!("program_ad_stencil_reduction/workspace.rs");

fn validate_source(
    effect_index: usize,
    spec: &StencilSpec,
    source_values: &[f64],
) -> Result<(), String> {
    let source_size = shape_size(effect_index, &spec.source_shape)?;
    if source_values.len() != source_size {
        return Err(format!(
            "effect {effect_index} stencil gradient expects {source_size} inputs, got {}",
            source_values.len()
        ));
    }
    for (index, value) in source_values.iter().enumerate() {
        if index % 256 == 0 { replay_checkpoint()?; }
        if !value.is_finite() {
            return Err(format!("effect {effect_index} stencil gradient inputs must be finite"));
        }
    }
    replay_checkpoint()?;
    if spec.axis >= spec.source_shape.len() {
        return Err(format!(
            "effect {effect_index} stencil gradient axis is outside source rank"
        ));
    }
    let axis_size = spec.source_shape[spec.axis];
    if axis_size < spec.edge_order + 1 {
        return Err(format!(
            "effect {effect_index} stencil gradient edge order {} requires at least {} samples",
            spec.edge_order,
            spec.edge_order + 1
        ));
    }
    match &spec.spacing {
        StencilSpacing::Scalar(_) => {}
        StencilSpacing::Coordinates(coordinates) => {
            if coordinates.len() != axis_size {
                return Err(format!(
                    "effect {effect_index} stencil gradient coordinates must match axis size"
                ));
            }
        }
    }
    if spec.output_index >= source_size {
        return Err(format!(
            "effect {effect_index} stencil gradient output index is outside source shape"
        ));
    }
    Ok(())
}

fn shape_size(effect_index: usize, shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for dimension in shape {
        replay_checkpoint()?;
        size = size.checked_mul(*dimension)
            .filter(|size| *size <= isize::MAX as usize / std::mem::size_of::<f64>())
            .ok_or_else(|| {
            format!("effect {effect_index} stencil gradient shape size overflowed")
        })?;
    }
    Ok(size)
}

fn unravel_index(
    effect_index: usize,
    spec: &StencilSpec,
    flat_index: usize,
) -> Result<Vec<usize>, String> {
    let source_size = shape_size(effect_index, &spec.source_shape)?;
    if flat_index >= source_size {
        return Err(format!(
            "effect {effect_index} stencil gradient output index is outside source shape"
        ));
    }
    let mut remainder = flat_index;
    let mut index = filled_replay_buffer(spec.source_shape.len(), 0usize)?;
    for axis in (0..spec.source_shape.len()).rev() {
        replay_checkpoint()?;
        let dimension = spec.source_shape[axis];
        index[axis] = remainder % dimension;
        remainder /= dimension;
    }
    Ok(index)
}

fn ravel_index(shape: &[usize], index: &[usize]) -> Result<usize, String> {
    if shape.len() != index.len() {
        return Err("stencil source index rank does not match shape".to_owned());
    }
    let mut flat_index = 0usize;
    for (axis, dimension) in shape.iter().enumerate() {
        replay_checkpoint()?;
        if index[axis] >= *dimension {
            return Err("stencil source index is outside shape".to_owned());
        }
        flat_index = flat_index.checked_mul(*dimension)
            .and_then(|offset| offset.checked_add(index[axis]))
            .ok_or_else(|| "stencil source index overflows".to_owned())?;
    }
    Ok(flat_index)
}

enum StencilCoefficients {
    Pair([(usize, f64); 2]),
    Triple([(usize, f64); 3]),
}

impl StencilCoefficients {
    fn as_slice(&self) -> &[(usize, f64)] {
        match self {
            Self::Pair(terms) => terms,
            Self::Triple(terms) => terms,
        }
    }
}

fn gradient_coefficients(
    effect_index: usize,
    spec: &StencilSpec,
    position: usize,
) -> Result<StencilCoefficients, String> {
    replay_checkpoint()?;
    let axis_size = spec.source_shape[spec.axis];
    let coefficients = match &spec.spacing {
        StencilSpacing::Scalar(dx) => {
            scalar_gradient_coefficients(position, axis_size, *dx, spec.edge_order)
        }
        StencilSpacing::Coordinates(coordinates) => {
            coordinate_gradient_coefficients(position, axis_size, coordinates, spec.edge_order)
        }
    };
    if coefficients.as_slice().iter().all(|(_, value)| value.is_finite()) {
        Ok(coefficients)
    } else {
        Err(format!(
            "effect {effect_index} stencil gradient coefficients must be finite"
        ))
    }
}

fn scalar_gradient_coefficients(
    position: usize,
    axis_size: usize,
    dx: f64,
    edge_order: usize,
) -> StencilCoefficients {
    if position == 0 {
        if edge_order == 1 {
            return StencilCoefficients::Pair([(0, -1.0 / dx), (1, 1.0 / dx)]);
        }
        return StencilCoefficients::Triple([(0, -1.5 / dx), (1, 2.0 / dx), (2, -0.5 / dx)]);
    }
    if position == axis_size - 1 {
        if edge_order == 1 {
            return StencilCoefficients::Pair([(axis_size - 2, -1.0 / dx), (axis_size - 1, 1.0 / dx)]);
        }
        return StencilCoefficients::Triple([
            (axis_size - 3, 0.5 / dx),
            (axis_size - 2, -2.0 / dx),
            (axis_size - 1, 1.5 / dx),
        ]);
    }
    StencilCoefficients::Pair([(position - 1, -0.5 / dx), (position + 1, 0.5 / dx)])
}

fn coordinate_gradient_coefficients(
    position: usize,
    axis_size: usize,
    coordinates: &[f64],
    edge_order: usize,
) -> StencilCoefficients {
    if position == 0 {
        let dx1 = coordinates[1] - coordinates[0];
        if edge_order == 1 {
            return StencilCoefficients::Pair([(0, -1.0 / dx1), (1, 1.0 / dx1)]);
        }
        let dx2 = coordinates[2] - coordinates[1];
        return StencilCoefficients::Triple([
            (0, -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))),
            (1, (dx1 + dx2) / (dx1 * dx2)),
            (2, -dx1 / (dx2 * (dx1 + dx2))),
        ]);
    }
    if position == axis_size - 1 {
        let dx2 = coordinates[axis_size - 1] - coordinates[axis_size - 2];
        if edge_order == 1 {
            return StencilCoefficients::Pair([(axis_size - 2, -1.0 / dx2), (axis_size - 1, 1.0 / dx2)]);
        }
        let dx1 = coordinates[axis_size - 2] - coordinates[axis_size - 3];
        return StencilCoefficients::Triple([
            (axis_size - 3, dx2 / (dx1 * (dx1 + dx2))),
            (axis_size - 2, -(dx1 + dx2) / (dx1 * dx2)),
            (axis_size - 1, (dx1 + 2.0 * dx2) / (dx2 * (dx1 + dx2))),
        ]);
    }
    let dx1 = coordinates[position] - coordinates[position - 1];
    let dx2 = coordinates[position + 1] - coordinates[position];
    StencilCoefficients::Triple([
        (position - 1, -dx2 / (dx1 * (dx1 + dx2))),
        (position, (dx2 - dx1) / (dx1 * dx2)),
        (position + 1, dx1 / (dx2 * (dx1 + dx2))),
    ])
}
