// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD order-statistic metadata and ordering

fn parse_order_statistic_operation(
    effect_index: usize,
    operation: &str,
    rank: usize,
) -> Result<OrderStatisticSpec, String> {
    replay_checkpoint()?;
    let mut tokens = operation.split(':');
    let reduction = OrderStatisticReduction::from_label(tokens.next().unwrap_or_default()).ok_or_else(|| {
        format!("effect {effect_index} operation {operation} is not an order-statistic reduction")
    })?;
    let mut axis = None;
    let mut q = reduction.fixed_q();
    while let Some(field) = tokens.next() {
        replay_checkpoint()?;
        match field {
            "axis" => {
                if axis.is_some() {
                    return Err(format!(
                        "effect {effect_index} {} operation has duplicate axis metadata",
                        reduction.label()
                    ));
                }
                let raw_axis = tokens.next().ok_or_else(|| {
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
                let raw_q = tokens.next().ok_or_else(|| {
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
    index_without_axis(source_shape, axis)
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

fn shape_size(reduction: OrderStatisticReduction, shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for (index, dimension) in shape.iter().enumerate() {
        if index % 256 == 0 { replay_checkpoint()?; }
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

fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Result<Vec<usize>, String> {
    let mut index = filled_replay_buffer(shape.len(), 0usize)?;
    for (axis, dimension) in shape.iter().enumerate().rev() {
        if axis % 256 == 0 { replay_checkpoint()?; }
        if *dimension == 0 {
            return Err("order-statistic index shape dimensions must be positive".to_owned());
        }
        index[axis] = flat_index % dimension;
        flat_index /= dimension;
    }
    Ok(index)
}

fn index_without_axis(index: &[usize], axis: usize) -> Result<Vec<usize>, String> {
    let mut result = reserve_replay_buffer(index.len() - usize::from(axis < index.len()))?;
    for (entry_axis, entry) in index.iter().enumerate() {
        if entry_axis % 256 == 0 { replay_checkpoint()?; }
        if entry_axis != axis { result.push(*entry); }
    }
    Ok(result)
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
    for (axis, (coordinate, dimension)) in index.iter().zip(shape.iter()).rev().enumerate() {
        if axis % 256 == 0 { replay_checkpoint()?; }
        if coordinate >= dimension {
            return Err(format!(
                "{} coordinate {coordinate} is outside dimension {dimension}",
                reduction.label()
            ));
        }
        flat = coordinate.checked_mul(stride)
            .and_then(|offset| flat.checked_add(offset))
            .ok_or_else(|| format!("{} ravel offset overflowed", reduction.label()))?;
        stride = stride
            .checked_mul(*dimension)
            .ok_or_else(|| format!("{} ravel stride overflowed", reduction.label()))?;
    }
    Ok(flat)
}


/// Validate the canonical selector metadata and return its optional reduction axis.
pub(crate) fn order_statistic_reduction_axis(effect_index: usize, operation: &str, rank: usize) -> Result<Option<usize>, String> {
    Ok(parse_order_statistic_operation(effect_index, operation, rank)?.axis)
}
