// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD variance and standard-deviation replay

//! Population variance and standard-deviation reduction replay for Program AD.
//!
//! The forward pass supports all-axis and static-axis population moments. Reverse
//! replay uses the exact centered cotangent rules and fails closed for standard
//! deviation groups with zero variance, where the derivative is singular.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MomentReduction {
    Variance,
    StandardDeviation,
}

impl MomentReduction {
    fn label(self) -> &'static str {
        match self {
            Self::Variance => "var",
            Self::StandardDeviation => "std",
        }
    }

    fn group_value(self, effect_index: usize, values: &[f64]) -> Result<f64, String> {
        let (mean, variance) = population_moments(effect_index, self.label(), values)?;
        let value = match self {
            Self::Variance => variance,
            Self::StandardDeviation => variance.sqrt(),
        };
        validate_finite_moment(effect_index, self.label(), value)?;
        if !mean.is_finite() {
            return Err(format!(
                "effect {effect_index} {} mean must be finite",
                self.label()
            ));
        }
        Ok(value)
    }

    fn group_cotangent(
        self,
        effect_index: usize,
        values: &[f64],
        cotangent: f64,
    ) -> Result<Vec<f64>, String> {
        let (mean, variance) = population_moments(effect_index, self.label(), values)?;
        let count = values.len() as f64;
        let contributions = match self {
            Self::Variance => values
                .iter()
                .map(|value| cotangent * 2.0 * (value - mean) / count)
                .collect::<Vec<f64>>(),
            Self::StandardDeviation => {
                if variance <= 0.0 {
                    return Err(format!(
                        "effect {effect_index} std gradient requires positive variance per reduction group"
                    ));
                }
                let standard_deviation = variance.sqrt();
                values
                    .iter()
                    .map(|value| cotangent * (value - mean) / (count * standard_deviation))
                    .collect::<Vec<f64>>()
            }
        };
        if contributions.iter().all(|value| value.is_finite()) {
            Ok(contributions)
        } else {
            Err(format!(
                "effect {effect_index} {} adjoint contribution must be finite",
                self.label()
            ))
        }
    }
}

/// Evaluate the population variance over every flattened source value.
pub(crate) fn variance_all_value(
    effect_index: usize,
    source_values: &[f64],
) -> Result<f64, String> {
    MomentReduction::Variance.group_value(effect_index, source_values)
}

/// Evaluate the population standard deviation over every flattened source value.
pub(crate) fn std_all_value(effect_index: usize, source_values: &[f64]) -> Result<f64, String> {
    MomentReduction::StandardDeviation.group_value(effect_index, source_values)
}

/// Evaluate a static-axis population variance reduction.
pub(crate) fn variance_axis_values(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    target_shape: &[usize],
    source_values: &[f64],
) -> Result<Vec<f64>, String> {
    moment_axis_values(
        effect_index,
        source_shape,
        axis,
        target_shape,
        source_values,
        MomentReduction::Variance,
    )
}

/// Evaluate a static-axis population standard-deviation reduction.
pub(crate) fn std_axis_values(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    target_shape: &[usize],
    source_values: &[f64],
) -> Result<Vec<f64>, String> {
    moment_axis_values(
        effect_index,
        source_shape,
        axis,
        target_shape,
        source_values,
        MomentReduction::StandardDeviation,
    )
}

/// Build the all-axis population variance adjoint contribution.
pub(crate) fn variance_all_cotangent(
    effect_index: usize,
    source_values: &[f64],
    scalar_cotangent: f64,
) -> Result<Vec<f64>, String> {
    MomentReduction::Variance.group_cotangent(effect_index, source_values, scalar_cotangent)
}

/// Build the all-axis population standard-deviation adjoint contribution.
pub(crate) fn std_all_cotangent(
    effect_index: usize,
    source_values: &[f64],
    scalar_cotangent: f64,
) -> Result<Vec<f64>, String> {
    MomentReduction::StandardDeviation.group_cotangent(
        effect_index,
        source_values,
        scalar_cotangent,
    )
}

/// Build the source-shaped adjoint contribution for a static-axis variance.
pub(crate) fn variance_axis_cotangent(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    cotangent_values: &[f64],
    source_values: &[f64],
) -> Result<Vec<f64>, String> {
    moment_axis_cotangent(
        effect_index,
        source_shape,
        axis,
        cotangent_values,
        source_values,
        MomentReduction::Variance,
    )
}

/// Build the source-shaped adjoint contribution for a static-axis standard deviation.
pub(crate) fn std_axis_cotangent(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    cotangent_values: &[f64],
    source_values: &[f64],
) -> Result<Vec<f64>, String> {
    moment_axis_cotangent(
        effect_index,
        source_shape,
        axis,
        cotangent_values,
        source_values,
        MomentReduction::StandardDeviation,
    )
}

fn moment_axis_values(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    target_shape: &[usize],
    source_values: &[f64],
    reduction: MomentReduction,
) -> Result<Vec<f64>, String> {
    validate_source_size(reduction, source_shape, source_values)?;
    validate_axis_target_shape(effect_index, reduction, source_shape, axis, target_shape)?;
    axis_groups(
        effect_index,
        reduction,
        source_shape,
        axis,
        target_shape,
        source_values,
    )?
    .iter()
    .map(|group| reduction.group_value(effect_index, group))
    .collect()
}

fn moment_axis_cotangent(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    cotangent_values: &[f64],
    source_values: &[f64],
    reduction: MomentReduction,
) -> Result<Vec<f64>, String> {
    validate_source_size(reduction, source_shape, source_values)?;
    let target_shape = axis_reduction_shape(reduction, source_shape, axis)?;
    if shape_size(reduction, &target_shape)? != cotangent_values.len() {
        return Err(format!(
            "effect {effect_index} {} axis cotangent shape must be {:?}",
            reduction.label(),
            target_shape
        ));
    }
    let mut groups = vec![Vec::<(usize, f64)>::new(); cotangent_values.len()];
    for (flat_index, value) in source_values.iter().copied().enumerate() {
        let source_index = unravel_index(flat_index, source_shape);
        let target_index = index_without_axis(&source_index, axis);
        let target_flat = ravel_index(reduction, &target_index, &target_shape)?;
        groups[target_flat].push((flat_index, value));
    }
    let mut contribution = vec![0.0_f64; source_values.len()];
    for (group, cotangent) in groups.iter().zip(cotangent_values.iter()) {
        let group_values = group.iter().map(|(_, value)| *value).collect::<Vec<f64>>();
        let group_contribution =
            reduction.group_cotangent(effect_index, &group_values, *cotangent)?;
        for ((source_index, _), value) in group.iter().zip(group_contribution.iter()) {
            contribution[*source_index] = *value;
        }
    }
    Ok(contribution)
}

fn axis_groups(
    effect_index: usize,
    reduction: MomentReduction,
    source_shape: &[usize],
    axis: usize,
    target_shape: &[usize],
    source_values: &[f64],
) -> Result<Vec<Vec<f64>>, String> {
    let mut groups = vec![Vec::<f64>::new(); shape_size(reduction, target_shape)?];
    for (flat_index, value) in source_values.iter().copied().enumerate() {
        let source_index = unravel_index(flat_index, source_shape);
        let target_index = index_without_axis(&source_index, axis);
        let target_flat = ravel_index(reduction, &target_index, target_shape)?;
        groups[target_flat].push(value);
    }
    if groups.iter().any(Vec::is_empty) {
        return Err(format!(
            "effect {effect_index} {} axis reduction produced an empty group",
            reduction.label()
        ));
    }
    Ok(groups)
}

fn population_moments(
    effect_index: usize,
    label: &str,
    values: &[f64],
) -> Result<(f64, f64), String> {
    if values.is_empty() {
        return Err(format!(
            "effect {effect_index} {label} requires non-empty values"
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} {label} source values must be finite"
        ));
    }
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / count;
    if variance.is_finite() && variance >= 0.0 {
        Ok((mean, variance))
    } else {
        Err(format!(
            "effect {effect_index} {label} population variance must be finite and non-negative"
        ))
    }
}

fn validate_source_size(
    reduction: MomentReduction,
    source_shape: &[usize],
    source_values: &[f64],
) -> Result<(), String> {
    let expected = shape_size(reduction, source_shape)?;
    if expected != source_values.len() {
        return Err(format!(
            "{} source shape {:?} expects {expected} values, got {}",
            reduction.label(),
            source_shape,
            source_values.len()
        ));
    }
    Ok(())
}

fn validate_axis_target_shape(
    effect_index: usize,
    reduction: MomentReduction,
    source_shape: &[usize],
    axis: usize,
    target_shape: &[usize],
) -> Result<(), String> {
    let expected_shape = axis_reduction_shape(reduction, source_shape, axis)?;
    if expected_shape != target_shape {
        return Err(format!(
            "effect {effect_index} {} axis reduction target shape must be {:?}, got {:?}",
            reduction.label(),
            expected_shape,
            target_shape
        ));
    }
    Ok(())
}

fn validate_finite_moment(effect_index: usize, label: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!(
            "effect {effect_index} {label} result must be finite"
        ))
    }
}

fn axis_reduction_shape(
    reduction: MomentReduction,
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

fn shape_size(reduction: MomentReduction, shape: &[usize]) -> Result<usize, String> {
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
    reduction: MomentReduction,
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
