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
    let spec = parse_stencil_operation(effect_index, operation)?;
    validate_source(effect_index, &spec, source_values)?;
    let target_index = unravel_index(effect_index, &spec, spec.output_index)?;
    let position = target_index[spec.axis];
    let value = gradient_coefficients(effect_index, &spec, position)?
        .iter()
        .map(|(axis_index, coefficient)| {
            let mut source_index = target_index.clone();
            source_index[spec.axis] = *axis_index;
            coefficient * source_values[ravel_index(&spec.source_shape, &source_index)]
        })
        .sum::<f64>();
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
    let spec = parse_stencil_operation(effect_index, operation)?;
    validate_source(effect_index, &spec, source_values)?;
    let target_index = unravel_index(effect_index, &spec, spec.output_index)?;
    let position = target_index[spec.axis];
    let mut contribution = vec![0.0_f64; source_values.len()];
    for (axis_index, coefficient) in gradient_coefficients(effect_index, &spec, position)? {
        let mut source_index = target_index.clone();
        source_index[spec.axis] = axis_index;
        contribution[ravel_index(&spec.source_shape, &source_index)] += cotangent * coefficient;
    }
    if contribution.iter().all(|value| value.is_finite()) {
        Ok(contribution)
    } else {
        Err(format!(
            "effect {effect_index} stencil gradient compact adjoint contribution must be finite"
        ))
    }
}

fn parse_stencil_operation(effect_index: usize, operation: &str) -> Result<StencilSpec, String> {
    let parts = operation.split(':').collect::<Vec<&str>>();
    if parts.len() != 12
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
    let source_shape = parse_shape_label(effect_index, parts[3])?;
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
    let spacing = parse_spacing_label(effect_index, parts[9])?;
    let output_index = parts[11].parse::<usize>().map_err(|_| {
        format!("effect {effect_index} stencil gradient output index must be non-negative")
    })?;
    Ok(StencilSpec {
        source_shape,
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
    let shape = label
        .split('x')
        .map(|part| {
            part.parse::<usize>().map_err(|_| {
                format!("effect {effect_index} stencil gradient shape must be positive integers")
            })
        })
        .collect::<Result<Vec<usize>, String>>()?;
    if shape.is_empty() || shape.contains(&0) {
        return Err(format!(
            "effect {effect_index} stencil gradient shape dimensions must be positive"
        ));
    }
    Ok(shape)
}

fn parse_spacing_label(effect_index: usize, label: &str) -> Result<StencilSpacing, String> {
    if let Some(raw) = label.strip_prefix("scalar=") {
        let value = raw.parse::<f64>().map_err(|_| {
            format!("effect {effect_index} stencil gradient scalar spacing must be finite")
        })?;
        if !value.is_finite() || value == 0.0 {
            return Err(format!(
                "effect {effect_index} stencil gradient scalar spacing must be finite and non-zero"
            ));
        }
        return Ok(StencilSpacing::Scalar(value));
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
    let coordinates = raw
        .split(',')
        .map(|part| {
            part.parse::<f64>().map_err(|_| {
                format!("effect {effect_index} stencil gradient coordinates must be finite")
            })
        })
        .collect::<Result<Vec<f64>, String>>()?;
    if coordinates.len() < 2 || coordinates.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} stencil gradient coordinates must be finite with at least two samples"
        ));
    }
    let increasing = coordinates.windows(2).all(|pair| pair[1] > pair[0]);
    let decreasing = coordinates.windows(2).all(|pair| pair[1] < pair[0]);
    if !increasing && !decreasing {
        return Err(format!(
            "effect {effect_index} stencil gradient coordinates must be strictly monotonic"
        ));
    }
    Ok(StencilSpacing::Coordinates(coordinates))
}

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
    if source_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} stencil gradient inputs must be finite"
        ));
    }
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
        size = size.checked_mul(*dimension).ok_or_else(|| {
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
    let mut index = vec![0usize; spec.source_shape.len()];
    for axis in (0..spec.source_shape.len()).rev() {
        let dimension = spec.source_shape[axis];
        index[axis] = remainder % dimension;
        remainder /= dimension;
    }
    Ok(index)
}

fn ravel_index(shape: &[usize], index: &[usize]) -> usize {
    let mut flat_index = 0usize;
    for (axis, dimension) in shape.iter().enumerate() {
        flat_index = flat_index * *dimension + index[axis];
    }
    flat_index
}

fn gradient_coefficients(
    effect_index: usize,
    spec: &StencilSpec,
    position: usize,
) -> Result<Vec<(usize, f64)>, String> {
    let axis_size = spec.source_shape[spec.axis];
    let coefficients = match &spec.spacing {
        StencilSpacing::Scalar(dx) => {
            scalar_gradient_coefficients(position, axis_size, *dx, spec.edge_order)
        }
        StencilSpacing::Coordinates(coordinates) => {
            coordinate_gradient_coefficients(position, axis_size, coordinates, spec.edge_order)
        }
    };
    if coefficients.iter().all(|(_, value)| value.is_finite()) {
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
) -> Vec<(usize, f64)> {
    if position == 0 {
        if edge_order == 1 {
            return vec![(0, -1.0 / dx), (1, 1.0 / dx)];
        }
        return vec![(0, -1.5 / dx), (1, 2.0 / dx), (2, -0.5 / dx)];
    }
    if position == axis_size - 1 {
        if edge_order == 1 {
            return vec![(axis_size - 2, -1.0 / dx), (axis_size - 1, 1.0 / dx)];
        }
        return vec![
            (axis_size - 3, 0.5 / dx),
            (axis_size - 2, -2.0 / dx),
            (axis_size - 1, 1.5 / dx),
        ];
    }
    vec![(position - 1, -0.5 / dx), (position + 1, 0.5 / dx)]
}

fn coordinate_gradient_coefficients(
    position: usize,
    axis_size: usize,
    coordinates: &[f64],
    edge_order: usize,
) -> Vec<(usize, f64)> {
    if position == 0 {
        let dx1 = coordinates[1] - coordinates[0];
        if edge_order == 1 {
            return vec![(0, -1.0 / dx1), (1, 1.0 / dx1)];
        }
        let dx2 = coordinates[2] - coordinates[1];
        return vec![
            (0, -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))),
            (1, (dx1 + dx2) / (dx1 * dx2)),
            (2, -dx1 / (dx2 * (dx1 + dx2))),
        ];
    }
    if position == axis_size - 1 {
        let dx1 = coordinates[axis_size - 2] - coordinates[axis_size - 3];
        let dx2 = coordinates[axis_size - 1] - coordinates[axis_size - 2];
        if edge_order == 1 {
            return vec![(axis_size - 2, -1.0 / dx2), (axis_size - 1, 1.0 / dx2)];
        }
        return vec![
            (axis_size - 3, dx2 / (dx1 * (dx1 + dx2))),
            (axis_size - 2, -(dx1 + dx2) / (dx1 * dx2)),
            (axis_size - 1, (dx1 + 2.0 * dx2) / (dx2 * (dx1 + dx2))),
        ];
    }
    let dx1 = coordinates[position] - coordinates[position - 1];
    let dx2 = coordinates[position + 1] - coordinates[position];
    vec![
        (position - 1, -dx2 / (dx1 * (dx1 + dx2))),
        (position, (dx2 - dx1) / (dx1 * dx2)),
        (position + 1, dx1 / (dx2 * (dx1 + dx2))),
    ]
}
