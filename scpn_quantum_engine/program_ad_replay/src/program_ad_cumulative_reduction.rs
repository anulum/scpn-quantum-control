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

use crate::program_ad_lifecycle::replay_checkpoint;

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
    replay_checkpoint()?;
    let spec = parse_cumulative_operation(effect_index, operation, source_values.len())?;
    validate_source(effect_index, &spec, source_values)?;
    let mut value = if spec.kind == CumulativeKind::Cumprod {
        1.0
    } else {
        -0.0
    };
    match spec.kind {
        CumulativeKind::Cumsum | CumulativeKind::Cumprod => {
            for source_index in prefix_indices(effect_index, &spec)? {
                replay_checkpoint()?;
                if spec.kind == CumulativeKind::Cumprod {
                    value *= source_values[source_index];
                } else {
                    value += source_values[source_index];
                }
            }
        }
        CumulativeKind::Diff => {
            for (source_index, coefficient) in diff_terms(effect_index, &spec)? {
                replay_checkpoint()?;
                value += coefficient * source_values[source_index];
            }
        }
    }
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
    replay_checkpoint()?;
    let spec = parse_cumulative_operation(effect_index, operation, source_values.len())?;
    validate_source(effect_index, &spec, source_values)?;
    let mut contribution = cumulative_buffer(source_values.len())?;
    for index in 0..source_values.len() {
        if index.is_multiple_of(256) {
            replay_checkpoint()?;
        }
        contribution.push(0.0_f64);
    }
    match spec.kind {
        CumulativeKind::Cumsum => {
            for source_index in prefix_indices(effect_index, &spec)? {
                replay_checkpoint()?;
                contribution[source_index] += cotangent;
            }
        }
        CumulativeKind::Cumprod => {
            let prefix = prefix_indices(effect_index, &spec)?;
            for differentiated_index in &prefix {
                replay_checkpoint()?;
                let mut product = 1.0;
                for source_index in &prefix {
                    replay_checkpoint()?;
                    if source_index != differentiated_index {
                        product *= source_values[*source_index];
                    }
                }
                contribution[*differentiated_index] += cotangent * product;
            }
        }
        CumulativeKind::Diff => {
            for (source_index, coefficient) in diff_terms(effect_index, &spec)? {
                replay_checkpoint()?;
                contribution[source_index] += cotangent * coefficient;
            }
        }
    }
    for (index, value) in contribution.iter().enumerate() {
        if index.is_multiple_of(256) {
            replay_checkpoint()?;
        }
        if !value.is_finite() {
            return Err(format!(
                "effect {effect_index} {} compact adjoint contribution must be finite",
                spec.kind.label()
            ));
        }
    }
    Ok(contribution)
}

include!("program_ad_cumulative_reduction/metadata.rs");
include!("program_ad_cumulative_reduction/indices.rs");
include!("program_ad_cumulative_reduction/workspace.rs");

fn parse_cumulative_operation(
    effect_index: usize,
    operation: &str,
    input_count: usize,
) -> Result<CumulativeSpec, String> {
    let layout = parse_cumulative_layout(effect_index, operation, input_count)?;
    let mut source_shape = cumulative_buffer(layout.rank)?;
    for entry in layout.shape_label.split('x') {
        replay_checkpoint()?;
        source_shape.push(
            entry
                .parse::<usize>()
                .map_err(|_| "cumulative validated shape dimension is invalid".to_owned())?,
        );
    }
    Ok(CumulativeSpec {
        kind: layout.kind,
        source_shape,
        axis: layout.axis,
        order: layout.order,
        output_index: layout.output_index,
    })
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
    for (index, value) in source_values.iter().enumerate() {
        if index.is_multiple_of(256) {
            replay_checkpoint()?;
        }
        if !value.is_finite() {
            return Err(format!(
                "effect {effect_index} {} source values must be finite",
                spec.kind.label()
            ));
        }
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

fn cumulative_buffer<T>(count: usize) -> Result<Vec<T>, String> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "cumulative buffer exceeds native addressable memory".to_owned())?;
    replay_checkpoint()?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(count)
        .map_err(|error| format!("cumulative buffer allocation refused: {error}"))?;
    Ok(buffer)
}

fn copy_cumulative_buffer<T: Copy>(source: &[T]) -> Result<Vec<T>, String> {
    let mut buffer = cumulative_buffer(source.len())?;
    buffer.extend_from_slice(source);
    replay_checkpoint()?;
    Ok(buffer)
}
