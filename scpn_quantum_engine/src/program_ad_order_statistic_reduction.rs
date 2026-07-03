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
            Ok(vec![group_value(
                effect_index,
                spec,
                &source_values
                    .iter()
                    .copied()
                    .enumerate()
                    .collect::<Vec<(usize, f64)>>(),
            )?])
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
            axis_groups(
                effect_index,
                spec.reduction,
                source_shape,
                axis,
                target_shape,
                source_values,
            )?
            .iter()
            .map(|group| group_value(effect_index, spec, group))
            .collect()
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
            let mut contribution = vec![0.0_f64; source_values.len()];
            for (source_index, value) in group_cotangent(
                effect_index,
                spec,
                &source_values
                    .iter()
                    .copied()
                    .enumerate()
                    .collect::<Vec<(usize, f64)>>(),
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
            let mut contribution = vec![0.0_f64; source_values.len()];
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

fn parse_order_statistic_operation(
    effect_index: usize,
    operation: &str,
    rank: usize,
) -> Result<OrderStatisticSpec, String> {
    let tokens = operation.split(':').collect::<Vec<&str>>();
    let reduction = OrderStatisticReduction::from_label(tokens[0]).ok_or_else(|| {
        format!("effect {effect_index} operation {operation} is not an order-statistic reduction")
    })?;
    let mut axis = None;
    let mut q = reduction.fixed_q();
    let mut index = 1usize;
    while index < tokens.len() {
        match tokens[index] {
            "axis" => {
                if axis.is_some() {
                    return Err(format!(
                        "effect {effect_index} {} operation has duplicate axis metadata",
                        reduction.label()
                    ));
                }
                let raw_axis = tokens.get(index + 1).ok_or_else(|| {
                    format!(
                        "effect {effect_index} {} operation requires static axis metadata {}:axis:<int>",
                        reduction.label(),
                        reduction.label()
                    )
                })?;
                let parsed_axis = raw_axis.parse::<isize>().map_err(|_| {
                    format!(
                        "effect {effect_index} {} axis metadata must be an integer",
                        reduction.label()
                    )
                })?;
                axis = Some(normalise_static_axis(parsed_axis, rank).map_err(|reason| {
                    format!(
                        "effect {effect_index} {} axis metadata is invalid: {reason}",
                        reduction.label()
                    )
                })?);
                index += 2;
            }
            "q" => {
                if reduction.fixed_q().is_some() {
                    return Err(format!(
                        "effect {effect_index} {} operation does not accept q metadata",
                        reduction.label()
                    ));
                }
                if q.is_some() {
                    return Err(format!(
                        "effect {effect_index} {} operation has duplicate q metadata",
                        reduction.label()
                    ));
                }
                let raw_q = tokens.get(index + 1).ok_or_else(|| {
                    format!(
                        "effect {effect_index} {} operation requires static scalar q metadata {}:q:<float>",
                        reduction.label(),
                        reduction.label()
                    )
                })?;
                let parsed_q = raw_q.parse::<f64>().map_err(|_| {
                    format!(
                        "effect {effect_index} {} q metadata must be a finite float",
                        reduction.label()
                    )
                })?;
                q = Some(normalise_q(effect_index, reduction, parsed_q)?);
                index += 2;
            }
            other => {
                return Err(format!(
                    "effect {effect_index} {} operation metadata field {other} is unsupported",
                    reduction.label()
                ));
            }
        }
    }
    let Some(q) = q else {
        return Err(format!(
            "effect {effect_index} {} operation requires static scalar q metadata {}:q:<float>",
            reduction.label(),
            reduction.label()
        ));
    };
    Ok(OrderStatisticSpec { reduction, axis, q })
}

fn normalise_q(
    effect_index: usize,
    reduction: OrderStatisticReduction,
    q: f64,
) -> Result<f64, String> {
    if !q.is_finite() {
        return Err(format!(
            "effect {effect_index} {} q metadata must be finite",
            reduction.label()
        ));
    }
    match reduction {
        OrderStatisticReduction::Quantile => {
            if !(0.0..=1.0).contains(&q) {
                return Err(format!(
                    "effect {effect_index} quantile q metadata must be in [0, 1]"
                ));
            }
            Ok(q)
        }
        OrderStatisticReduction::Percentile => {
            if !(0.0..=100.0).contains(&q) {
                return Err(format!(
                    "effect {effect_index} percentile q metadata must be in [0, 100]"
                ));
            }
            Ok(q / 100.0)
        }
        OrderStatisticReduction::Maximum
        | OrderStatisticReduction::Minimum
        | OrderStatisticReduction::Median => Ok(q),
    }
}

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
    let mut contribution = vec![(group[lower_index].0, cotangent * lower_weight)];
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
    let mut order = (0..group.len()).collect::<Vec<usize>>();
    order.sort_by(|left, right| group[*left].1.partial_cmp(&group[*right].1).unwrap());
    let position = spec.q * ((group.len() - 1) as f64);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let upper_weight = position - lower as f64;
    let lower_weight = 1.0 - upper_weight;
    let lower_index = order[lower];
    let upper_index = order[upper];
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
    if group.iter().any(|(_, value)| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} {} source values must be finite",
            reduction.label()
        ));
    }
    let mut sorted_values = group.iter().map(|(_, value)| *value).collect::<Vec<f64>>();
    sorted_values.sort_by(|left, right| left.partial_cmp(right).unwrap());
    if sorted_values
        .windows(2)
        .any(|window| window[0] == window[1])
    {
        return Err(format!(
            "effect {effect_index} {} gradient requires strictly ordered values per reduction group",
            reduction.label()
        ));
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
    let mut groups = vec![Vec::<(usize, f64)>::new(); shape_size(reduction, target_shape)?];
    for (flat_index, value) in source_values.iter().copied().enumerate() {
        let source_index = unravel_index(flat_index, source_shape);
        let target_index = index_without_axis(&source_index, axis);
        let target_flat = ravel_index(reduction, &target_index, target_shape)?;
        groups[target_flat].push((flat_index, value));
    }
    if groups.iter().any(Vec::is_empty) {
        return Err(format!(
            "effect {effect_index} {} axis reduction produced an empty group",
            reduction.label()
        ));
    }
    Ok(groups)
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

fn axis_reduction_shape(
    reduction: OrderStatisticReduction,
    source_shape: &[usize],
    axis: usize,
) -> Result<Vec<usize>, String> {
    if axis >= source_shape.len() {
        return Err(format!(
            "{} axis {axis} is outside rank {}",
            reduction.label(),
            source_shape.len()
        ));
    }
    Ok(source_shape
        .iter()
        .enumerate()
        .filter_map(|(index, dimension)| (index != axis).then_some(*dimension))
        .collect())
}

fn shape_size(reduction: OrderStatisticReduction, shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for dimension in shape {
        if *dimension == 0 {
            return Err(format!(
                "{} shaped values must have non-zero dimensions",
                reduction.label()
            ));
        }
        size = size
            .checked_mul(*dimension)
            .ok_or_else(|| format!("{} shaped value size overflowed", reduction.label()))?;
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

fn index_without_axis(index: &[usize], axis: usize) -> Vec<usize> {
    index
        .iter()
        .enumerate()
        .filter_map(|(entry_axis, entry)| (entry_axis != axis).then_some(*entry))
        .collect()
}

fn ravel_index(
    reduction: OrderStatisticReduction,
    index: &[usize],
    shape: &[usize],
) -> Result<usize, String> {
    if index.len() != shape.len() {
        return Err(format!(
            "{} index rank {} does not match shape rank {}",
            reduction.label(),
            index.len(),
            shape.len()
        ));
    }
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (coordinate, dimension) in index.iter().zip(shape.iter()).rev() {
        if coordinate >= dimension {
            return Err(format!(
                "{} coordinate {coordinate} is outside dimension {dimension}",
                reduction.label()
            ));
        }
        flat += coordinate * stride;
        stride = stride
            .checked_mul(*dimension)
            .ok_or_else(|| format!("{} ravel stride overflowed", reduction.label()))?;
    }
    Ok(flat)
}
