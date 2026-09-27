// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD order-statistic reduction replay

//! Strict-order selector and order-statistic reductions for bounded Program AD.
//!
//! The replay supports finite all-axis and static-axis `max`, `min`, `median`,
//! scalar-`q` `quantile`, and scalar-`q` `percentile` operations. Reverse replay
//! routes linear-interpolation cotangents to the selected source entries and
//! fails closed when equal source values make the selector nondifferentiable.

use crate::program_ad_ir::{filled_replay_buffer, reserve_replay_buffer};
use crate::program_ad_lifecycle::replay_checkpoint;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OrderStatisticReduction {
    Maximum,
    Minimum,
    Median,
    Quantile,
    Percentile,
}

impl OrderStatisticReduction {
    fn from_label(label: &str) -> Option<Self> {
        match label {
            "max" => Some(Self::Maximum),
            "min" => Some(Self::Minimum),
            "median" => Some(Self::Median),
            "quantile" => Some(Self::Quantile),
            "percentile" => Some(Self::Percentile),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Maximum => "max",
            Self::Minimum => "min",
            Self::Median => "median",
            Self::Quantile => "quantile",
            Self::Percentile => "percentile",
        }
    }

    fn fixed_q(self) -> Option<f64> {
        match self {
            Self::Maximum => Some(1.0),
            Self::Minimum => Some(0.0),
            Self::Median => Some(0.5),
            Self::Quantile | Self::Percentile => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct OrderStatisticSpec {
    reduction: OrderStatisticReduction,
    axis: Option<usize>,
    q: f64,
}

type InterpolationSelection = ((usize, f64), Option<(usize, f64)>);

/// Return whether an operation string names a selector or order-statistic reduction.
pub(crate) fn is_order_statistic_operation(operation: &str) -> bool {
    let label = operation.split(':').next().unwrap_or_default();
    OrderStatisticReduction::from_label(label).is_some()
}

/// Evaluate all-axis or static-axis selector/order-statistic reductions.
pub(crate) fn order_statistic_values(
    effect_index: usize,
    operation: &str,
    source_shape: &[usize],
    target_shape: &[usize],
    source_values: &[f64],
) -> Result<Vec<f64>, String> {
    validate_source(effect_index, source_shape, source_values)?;
    let spec = parse_order_statistic_operation(effect_index, operation, source_shape.len())?;
    match spec.axis {
        None => {
            if !target_shape.is_empty() {
                return Err(format!(
                    "effect {effect_index} {} non-scalar target requires static axis metadata {}:axis:<int>",
                    spec.reduction.label(),
                    spec.reduction.label()
                ));
            }
            let value = group_value(effect_index, spec, &indexed_source_values(source_values)?)?;
            let mut output = reserve_replay_buffer(1)?;
            output.push(value);
            Ok(output)
        }
        Some(axis) => {
            let expected_shape = axis_reduction_shape(spec.reduction, source_shape, axis)?;
            if expected_shape != target_shape {
                return Err(format!(
                    "effect {effect_index} {} axis reduction target shape must be {:?}, got {:?}",
                    spec.reduction.label(),
                    expected_shape,
                    target_shape
                ));
            }
            let groups = axis_groups(effect_index, spec.reduction, source_shape, axis, target_shape, source_values)?;
            let mut output = reserve_replay_buffer(groups.len())?;
            for group in &groups {
                replay_checkpoint()?;
                output.push(group_value(effect_index, spec, group)?);
            }
            Ok(output)
        }
    }
}

/// Build the source-shaped cotangent for selector/order-statistic reductions.
pub(crate) fn order_statistic_cotangent(
    effect_index: usize,
    operation: &str,
    source_shape: &[usize],
    cotangent_values: &[f64],
    source_values: &[f64],
) -> Result<Vec<f64>, String> {
    validate_source(effect_index, source_shape, source_values)?;
    let spec = parse_order_statistic_operation(effect_index, operation, source_shape.len())?;
    match spec.axis {
        None => {
            if cotangent_values.len() != 1 {
                return Err(format!(
                    "effect {effect_index} {} all-axis cotangent must be scalar",
                    spec.reduction.label()
                ));
            }
            let mut contribution = filled_replay_buffer(source_values.len(), 0.0_f64)?;
            for (source_index, value) in group_cotangent(
                effect_index,
                spec,
                &indexed_source_values(source_values)?,
                cotangent_values[0],
            )? {
                contribution[source_index] += value;
            }
            Ok(contribution)
        }
        Some(axis) => {
            let target_shape = axis_reduction_shape(spec.reduction, source_shape, axis)?;
            if shape_size(spec.reduction, &target_shape)? != cotangent_values.len() {
                return Err(format!(
                    "effect {effect_index} {} axis cotangent shape must be {:?}",
                    spec.reduction.label(),
                    target_shape
                ));
            }
            let groups = axis_groups(
                effect_index,
                spec.reduction,
                source_shape,
                axis,
                &target_shape,
                source_values,
            )?;
            let mut contribution = filled_replay_buffer(source_values.len(), 0.0_f64)?;
            for (group, cotangent) in groups.iter().zip(cotangent_values.iter()) {
                for (source_index, value) in group_cotangent(effect_index, spec, group, *cotangent)?
                {
                    contribution[source_index] += value;
                }
            }
            Ok(contribution)
        }
    }
}

include!("program_ad_order_statistic_reduction/metadata.rs");
include!("program_ad_order_statistic_reduction/ordering.rs");

fn group_value(
    effect_index: usize,
    spec: OrderStatisticSpec,
    group: &[(usize, f64)],
) -> Result<f64, String> {
    let ((lower_index, lower_weight), upper) = interpolation_weights(effect_index, spec, group)?;
    let lower_value = group[lower_index].1;
    let mut value = lower_value * lower_weight;
    if let Some((upper_index, upper_weight)) = upper {
        value += group[upper_index].1 * upper_weight;
    }
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "effect {effect_index} {} result must be finite",
            spec.reduction.label()
        ))
    }
}

fn group_cotangent(
    effect_index: usize,
    spec: OrderStatisticSpec,
    group: &[(usize, f64)],
    cotangent: f64,
) -> Result<Vec<(usize, f64)>, String> {
    if !cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} {} cotangent must be finite",
            spec.reduction.label()
        ));
    }
    let ((lower_index, lower_weight), upper) = interpolation_weights(effect_index, spec, group)?;
    let mut contribution = reserve_replay_buffer(2)?;
    contribution.push((group[lower_index].0, cotangent * lower_weight));
    if let Some((upper_index, upper_weight)) = upper {
        contribution.push((group[upper_index].0, cotangent * upper_weight));
    }
    Ok(contribution)
}

fn interpolation_weights(
    effect_index: usize,
    spec: OrderStatisticSpec,
    group: &[(usize, f64)],
) -> Result<InterpolationSelection, String> {
    validate_group(effect_index, spec.reduction, group)?;
    let mut order = reserve_replay_buffer(group.len())?;
    for index in 0..group.len() { replay_checkpoint()?; order.push(index); }
    checked_order(&mut order, |left, right| group[*left].1 < group[*right].1)?;
    let position = spec.q * ((group.len() - 1) as f64);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let upper_weight = position - lower as f64;
    let lower_weight = 1.0 - upper_weight;
    let lower_index = order.get(lower).copied().ok_or_else(|| {
        format!("effect {effect_index} order-statistic lower selection is outside group bounds")
    })?;
    let upper_index = order.get(upper).copied().ok_or_else(|| {
        format!("effect {effect_index} order-statistic upper selection is outside group bounds")
    })?;
    let upper_entry = (lower_index != upper_index).then_some((upper_index, upper_weight));
    Ok(((lower_index, lower_weight), upper_entry))
}

fn validate_group(
    effect_index: usize,
    reduction: OrderStatisticReduction,
    group: &[(usize, f64)],
) -> Result<(), String> {
    if group.is_empty() {
        return Err(format!(
            "effect {effect_index} {} requires at least one value per reduction group",
            reduction.label()
        ));
    }
    let mut sorted_values = reserve_replay_buffer(group.len())?;
    for (_, value) in group {
        replay_checkpoint()?;
        if !value.is_finite() {
            return Err(format!("effect {effect_index} {} source values must be finite", reduction.label()));
        }
        sorted_values.push(*value);
    }
    checked_order(&mut sorted_values, |left, right| left < right)?;
    for window in sorted_values.windows(2) {
        replay_checkpoint()?;
        if window[0] == window[1] {
            return Err(format!("effect {effect_index} {} gradient requires strictly ordered values per reduction group", reduction.label()));
        }
    }
    Ok(())
}

fn validate_source(
    effect_index: usize,
    source_shape: &[usize],
    source_values: &[f64],
) -> Result<(), String> {
    let expected = shape_size(OrderStatisticReduction::Median, source_shape)?;
    if expected != source_values.len() {
        return Err(format!(
            "effect {effect_index} order-statistic source shape {:?} expects {expected} values, got {}",
            source_shape,
            source_values.len()
        ));
    }
    if source_values.is_empty() {
        return Err(format!(
            "effect {effect_index} order-statistic reductions require at least one source value"
        ));
    }
    Ok(())
}

fn axis_groups(
    effect_index: usize,
    reduction: OrderStatisticReduction,
    source_shape: &[usize],
    axis: usize,
    target_shape: &[usize],
    source_values: &[f64],
) -> Result<Vec<Vec<(usize, f64)>>, String> {
    let count = shape_size(reduction, target_shape)?;
    let mut groups = reserve_replay_buffer(count)?;
    for _ in 0..count { groups.push(reserve_replay_buffer::<(usize, f64)>(source_shape[axis])?); }
    for (flat_index, value) in source_values.iter().copied().enumerate() {
        replay_checkpoint()?;
        let source_index = unravel_index(flat_index, source_shape)?;
        let target_index = index_without_axis(&source_index, axis)?;
        let target_flat = ravel_index(reduction, &target_index, target_shape)?;
        groups[target_flat].push((flat_index, value));
    }
    for group in &groups {
        replay_checkpoint()?;
        if group.is_empty() {
            return Err(format!(
                "effect {effect_index} {} axis reduction produced an empty group",
                reduction.label()
            ));
        }
    }
    Ok(groups)
}

fn indexed_source_values(source: &[f64]) -> Result<Vec<(usize, f64)>, String> {
    let mut values = reserve_replay_buffer(source.len())?;
    for (index, value) in source.iter().enumerate() {
        if index % 256 == 0 { replay_checkpoint()?; }
        values.push((index, *value));
    }
    Ok(values)
}
