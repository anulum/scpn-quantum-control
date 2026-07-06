// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD cumulative replay

//! Compact cumulative scan and finite-difference replay for bounded Program AD IR.
//!
//! The replay accepts Python-emitted scalar output opcodes for static-shape
//! `cumsum`, `cumprod`, and `diff`. Reverse replay returns the source-shaped
//! cotangent contribution for one compact output element and treats axis/order
//! metadata as nondifferentiable static metadata.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CumulativeKind {
    Cumsum,
    Cumprod,
    Diff,
}

impl CumulativeKind {
    fn from_label(label: &str) -> Option<Self> {
        match label {
            "cumsum" => Some(Self::Cumsum),
            "cumprod" => Some(Self::Cumprod),
            "diff" => Some(Self::Diff),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cumsum => "cumsum",
            Self::Cumprod => "cumprod",
            Self::Diff => "diff",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CumulativeAxis {
    Flat,
    Axis(usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CumulativeSpec {
    kind: CumulativeKind,
    source_shape: Vec<usize>,
    axis: CumulativeAxis,
    order: usize,
    output_index: usize,
}

/// Return whether an operation string names a compact cumulative primitive.
pub(crate) fn is_cumulative_operation(operation: &str) -> bool {
    CumulativeKind::from_label(operation.split(':').next().unwrap_or_default()).is_some()
}

/// Evaluate one compact cumulative output element.
pub(crate) fn cumulative_output_value(
    effect_index: usize,
    operation: &str,
    source_values: &[f64],
) -> Result<f64, String> {
    let spec = parse_cumulative_operation(effect_index, operation)?;
    validate_source(effect_index, &spec, source_values)?;
    let value: f64 = match spec.kind {
        CumulativeKind::Cumsum => prefix_indices(effect_index, &spec)?
            .iter()
            .map(|source_index| source_values[*source_index])
            .sum(),
        CumulativeKind::Cumprod => prefix_indices(effect_index, &spec)?
            .iter()
            .map(|source_index| source_values[*source_index])
            .product(),
        CumulativeKind::Diff => diff_terms(effect_index, &spec)?
            .iter()
            .map(|(source_index, coefficient)| coefficient * source_values[*source_index])
            .sum(),
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "effect {effect_index} {} compact value must be finite",
            spec.kind.label()
        ))
    }
}

/// Build source-shaped cotangent contribution for one compact cumulative output element.
pub(crate) fn cumulative_output_cotangent(
    effect_index: usize,
    operation: &str,
    source_values: &[f64],
    cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} cumulative cotangent must be finite"
        ));
    }
    let spec = parse_cumulative_operation(effect_index, operation)?;
    validate_source(effect_index, &spec, source_values)?;
    let mut contribution = vec![0.0_f64; source_values.len()];
    match spec.kind {
        CumulativeKind::Cumsum => {
            for source_index in prefix_indices(effect_index, &spec)? {
                contribution[source_index] += cotangent;
            }
        }
        CumulativeKind::Cumprod => {
            let prefix = prefix_indices(effect_index, &spec)?;
            for differentiated_index in &prefix {
                let product = prefix
                    .iter()
                    .filter(|source_index| *source_index != differentiated_index)
                    .map(|source_index| source_values[*source_index])
                    .product::<f64>();
                contribution[*differentiated_index] += cotangent * product;
            }
        }
        CumulativeKind::Diff => {
            for (source_index, coefficient) in diff_terms(effect_index, &spec)? {
                contribution[source_index] += cotangent * coefficient;
            }
        }
    }
    if contribution.iter().all(|value| value.is_finite()) {
        Ok(contribution)
    } else {
        Err(format!(
            "effect {effect_index} {} compact adjoint contribution must be finite",
            spec.kind.label()
        ))
    }
}

fn parse_cumulative_operation(
    effect_index: usize,
    operation: &str,
) -> Result<CumulativeSpec, String> {
    let fields = operation.split(':').collect::<Vec<&str>>();
    let kind = CumulativeKind::from_label(fields.first().copied().unwrap_or_default())
        .ok_or_else(|| format!("effect {effect_index} operation {operation} is not cumulative"))?;
    let mut source_shape = None;
    let mut axis = None;
    let mut order = None;
    let mut output_index = None;
    let mut index = 1usize;
    while index < fields.len() {
        let field = fields[index];
        let Some(raw_value) = fields.get(index + 1).copied() else {
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
                source_shape = Some(parse_shape_label(effect_index, kind, raw_value)?);
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
        index += 2;
    }
    let source_shape = source_shape.ok_or_else(|| {
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
            source_shape.len(),
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
    Ok(CumulativeSpec {
        kind,
        source_shape,
        axis,
        order,
        output_index: output_index.ok_or_else(|| {
            format!(
                "effect {effect_index} {} requires output index metadata",
                kind.label()
            )
        })?,
    })
}

fn parse_shape_label(
    effect_index: usize,
    kind: CumulativeKind,
    label: &str,
) -> Result<Vec<usize>, String> {
    if label.is_empty() {
        return Err(format!(
            "effect {effect_index} {} source shape metadata must be non-empty",
            kind.label()
        ));
    }
    let shape = label
        .split('x')
        .map(|entry| {
            entry.parse::<usize>().map_err(|_| {
                format!(
                    "effect {effect_index} {} source shape dimension must be non-negative",
                    kind.label()
                )
            })
        })
        .collect::<Result<Vec<usize>, String>>()?;
    if shape.is_empty() || shape.contains(&0) {
        return Err(format!(
            "effect {effect_index} {} source shape dimensions must be positive",
            kind.label()
        ));
    }
    Ok(shape)
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

fn validate_source(
    effect_index: usize,
    spec: &CumulativeSpec,
    source_values: &[f64],
) -> Result<(), String> {
    let source_size = shape_size(&spec.source_shape)?;
    if source_size != source_values.len() {
        return Err(format!(
            "effect {effect_index} {} source shape {:?} expects {source_size} values, got {}",
            spec.kind.label(),
            spec.source_shape,
            source_values.len()
        ));
    }
    if source_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} {} source values must be finite",
            spec.kind.label()
        ));
    }
    let output_size = shape_size(&output_shape(effect_index, spec)?)?;
    if spec.output_index >= output_size {
        return Err(format!(
            "effect {effect_index} {} output index {} is outside output size {output_size}",
            spec.kind.label(),
            spec.output_index
        ));
    }
    Ok(())
}

fn output_shape(effect_index: usize, spec: &CumulativeSpec) -> Result<Vec<usize>, String> {
    match spec.kind {
        CumulativeKind::Cumsum | CumulativeKind::Cumprod => match spec.axis {
            CumulativeAxis::Flat => Ok(vec![shape_size(&spec.source_shape)?]),
            CumulativeAxis::Axis(_) => Ok(spec.source_shape.clone()),
        },
        CumulativeKind::Diff => {
            let CumulativeAxis::Axis(axis) = spec.axis else {
                return Err(format!(
                    "effect {effect_index} diff requires a ranked static axis"
                ));
            };
            let axis_size = spec.source_shape[axis];
            if spec.order > axis_size {
                return Err(format!(
                    "effect {effect_index} diff order {} exceeds axis length {axis_size}",
                    spec.order
                ));
            }
            let mut shape = spec.source_shape.clone();
            shape[axis] = axis_size - spec.order;
            Ok(shape)
        }
    }
}

fn prefix_indices(effect_index: usize, spec: &CumulativeSpec) -> Result<Vec<usize>, String> {
    match spec.axis {
        CumulativeAxis::Flat => Ok((0..=spec.output_index).collect()),
        CumulativeAxis::Axis(axis) => {
            let target_index = unravel_index(spec.output_index, &spec.source_shape);
            let mut indices = Vec::with_capacity(target_index[axis] + 1);
            for axis_index in 0..=target_index[axis] {
                let mut source_index = target_index.clone();
                source_index[axis] = axis_index;
                indices.push(
                    ravel_index(&source_index, &spec.source_shape).map_err(|reason| {
                        format!(
                            "effect {effect_index} {} prefix index is invalid: {reason}",
                            spec.kind.label()
                        )
                    })?,
                );
            }
            Ok(indices)
        }
    }
}

fn diff_terms(effect_index: usize, spec: &CumulativeSpec) -> Result<Vec<(usize, f64)>, String> {
    let CumulativeAxis::Axis(axis) = spec.axis else {
        return Err(format!(
            "effect {effect_index} diff requires a ranked static axis"
        ));
    };
    let shape = output_shape(effect_index, spec)?;
    let output_index = unravel_index(spec.output_index, &shape);
    let mut terms = Vec::with_capacity(spec.order + 1);
    for offset in 0..=spec.order {
        let mut source_index = output_index.clone();
        source_index[axis] += offset;
        let coefficient = binomial(spec.order, offset) as f64
            * if (spec.order - offset).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
        terms.push((
            ravel_index(&source_index, &spec.source_shape).map_err(|reason| {
                format!("effect {effect_index} diff source index is invalid: {reason}")
            })?,
            coefficient,
        ));
    }
    Ok(terms)
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (0..k).fold(1usize, |accumulator, index| {
        accumulator * (n - index) / (index + 1)
    })
}

fn shape_size(shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for dimension in shape {
        size = size
            .checked_mul(*dimension)
            .ok_or_else(|| "cumulative shaped value size overflowed".to_owned())?;
    }
    Ok(size)
}

fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0usize; shape.len()];
    for (axis, dimension) in shape.iter().enumerate().rev() {
        index[axis] = flat_index % dimension;
        flat_index /= dimension;
    }
    index
}

fn ravel_index(index: &[usize], shape: &[usize]) -> Result<usize, String> {
    if index.len() != shape.len() {
        return Err(format!(
            "cumulative index rank {} does not match shape rank {}",
            index.len(),
            shape.len()
        ));
    }
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (coordinate, dimension) in index.iter().zip(shape.iter()).rev() {
        if coordinate >= dimension {
            return Err(format!(
                "cumulative coordinate {coordinate} is outside dimension {dimension}"
            ));
        }
        flat += coordinate * stride;
        stride = stride
            .checked_mul(*dimension)
            .ok_or_else(|| "cumulative stride overflowed".to_owned())?;
    }
    Ok(flat)
}
