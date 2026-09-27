// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD variance and standard-deviation replay

//! Corrected variance and standard-deviation reduction replay for Program AD.
//!
//! The forward pass supports all-axis and static-axis moments with static
//! `ddof`/`correction` metadata. Reverse replay uses the exact centered
//! cotangent rules and fails closed for invalid correction denominators and
//! standard-deviation groups with zero variance, where the derivative is
//! singular.

use crate::program_ad_ir::{filled_replay_buffer, reserve_replay_buffer};
use crate::program_ad_lifecycle::replay_checkpoint;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MomentReduction {
    Variance,
    StandardDeviation,
}

/// Static metadata attached to a Program AD moment-reduction opcode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MomentReductionMetadata {
    /// Optional static axis for shaped reductions; `None` means all-axis.
    pub(crate) axis: Option<usize>,
    /// Static delta-degrees-of-freedom correction in the variance denominator.
    pub(crate) correction: f64,
}

impl MomentReduction {
    fn label(self) -> &'static str {
        match self {
            Self::Variance => "var",
            Self::StandardDeviation => "std",
        }
    }

    fn group_value(
        self,
        effect_index: usize,
        values: &[f64],
        correction: f64,
    ) -> Result<f64, String> {
        let (mean, variance, _denominator) =
            corrected_moments(effect_index, self.label(), values, correction)?;
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
        correction: f64,
    ) -> Result<Vec<f64>, String> {
        let (mean, variance, denominator) =
            corrected_moments(effect_index, self.label(), values, correction)?;
        if self == Self::StandardDeviation && variance <= 0.0 {
            return Err(format!(
                "effect {effect_index} std gradient requires positive variance per reduction group"
            ));
        }
        let mut contributions = reserve_replay_buffer(values.len())?;
        let standard_deviation = variance.sqrt();
        for (index, value) in values.iter().enumerate() {
            if index % 256 == 0 {
                replay_checkpoint()?;
            }
            let contribution = match self {
                Self::Variance => cotangent * 2.0 * (value - mean) / denominator,
                Self::StandardDeviation => {
                    cotangent * (value - mean) / (denominator * standard_deviation)
                }
            };
            if !contribution.is_finite() {
                return Err(format!(
                    "effect {effect_index} {} adjoint contribution must be finite",
                    self.label()
                ));
            }
            contributions.push(contribution);
        }
        replay_checkpoint()?;
        Ok(contributions)
    }
}

include!("program_ad_variance_reduction/metadata.rs");

/// Evaluate the corrected variance over every flattened source value.
pub(crate) fn variance_all_value(
    effect_index: usize,
    source_values: &[f64],
    correction: f64,
) -> Result<f64, String> {
    MomentReduction::Variance.group_value(effect_index, source_values, correction)
}

/// Evaluate the corrected standard deviation over every flattened source value.
pub(crate) fn std_all_value(
    effect_index: usize,
    source_values: &[f64],
    correction: f64,
) -> Result<f64, String> {
    MomentReduction::StandardDeviation.group_value(effect_index, source_values, correction)
}

/// Evaluate a static-axis corrected variance reduction.
pub(crate) fn variance_axis_values(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    target_shape: &[usize],
    source_values: &[f64],
    correction: f64,
) -> Result<Vec<f64>, String> {
    moment_axis_values(
        effect_index,
        source_shape,
        axis,
        target_shape,
        source_values,
        MomentReduction::Variance,
        correction,
    )
}

/// Evaluate a static-axis corrected standard-deviation reduction.
pub(crate) fn std_axis_values(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    target_shape: &[usize],
    source_values: &[f64],
    correction: f64,
) -> Result<Vec<f64>, String> {
    moment_axis_values(
        effect_index,
        source_shape,
        axis,
        target_shape,
        source_values,
        MomentReduction::StandardDeviation,
        correction,
    )
}

/// Build the all-axis corrected variance adjoint contribution.
pub(crate) fn variance_all_cotangent(
    effect_index: usize,
    source_values: &[f64],
    scalar_cotangent: f64,
    correction: f64,
) -> Result<Vec<f64>, String> {
    MomentReduction::Variance.group_cotangent(
        effect_index,
        source_values,
        scalar_cotangent,
        correction,
    )
}

/// Build the all-axis corrected standard-deviation adjoint contribution.
pub(crate) fn std_all_cotangent(
    effect_index: usize,
    source_values: &[f64],
    scalar_cotangent: f64,
    correction: f64,
) -> Result<Vec<f64>, String> {
    MomentReduction::StandardDeviation.group_cotangent(
        effect_index,
        source_values,
        scalar_cotangent,
        correction,
    )
}

/// Build the source-shaped adjoint contribution for a static-axis corrected variance.
pub(crate) fn variance_axis_cotangent(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    cotangent_values: &[f64],
    source_values: &[f64],
    correction: f64,
) -> Result<Vec<f64>, String> {
    moment_axis_cotangent(
        effect_index,
        source_shape,
        axis,
        cotangent_values,
        source_values,
        MomentReduction::Variance,
        correction,
    )
}

/// Build the source-shaped adjoint contribution for a static-axis corrected standard deviation.
pub(crate) fn std_axis_cotangent(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    cotangent_values: &[f64],
    source_values: &[f64],
    correction: f64,
) -> Result<Vec<f64>, String> {
    moment_axis_cotangent(
        effect_index,
        source_shape,
        axis,
        cotangent_values,
        source_values,
        MomentReduction::StandardDeviation,
        correction,
    )
}

fn moment_axis_values(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    target_shape: &[usize],
    source_values: &[f64],
    reduction: MomentReduction,
    correction: f64,
) -> Result<Vec<f64>, String> {
    validate_source_size(reduction, source_shape, source_values)?;
    validate_axis_target_shape(effect_index, reduction, source_shape, axis, target_shape)?;
    let groups = axis_groups(
        effect_index,
        reduction,
        source_shape,
        axis,
        target_shape,
        source_values,
    )?;
    let mut output = reserve_replay_buffer(groups.len())?;
    for (index, group) in groups.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        output.push(reduction.group_value(effect_index, group, correction)?);
    }
    Ok(output)
}

fn moment_axis_cotangent(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    cotangent_values: &[f64],
    source_values: &[f64],
    reduction: MomentReduction,
    correction: f64,
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
    let mut groups = reserve_replay_buffer(cotangent_values.len())?;
    for _ in 0..cotangent_values.len() {
        groups.push(reserve_replay_buffer::<(usize, f64)>(source_shape[axis])?);
    }
    for (flat_index, value) in source_values.iter().copied().enumerate() {
        if flat_index % 256 == 0 {
            replay_checkpoint()?;
        }
        let source_index = unravel_index(flat_index, source_shape)?;
        let target_index = index_without_axis(&source_index, axis)?;
        let target_flat = ravel_index(reduction, &target_index, &target_shape)?;
        groups[target_flat].push((flat_index, value));
    }
    let mut contribution = filled_replay_buffer(source_values.len(), 0.0_f64)?;
    for (group_index, (group, cotangent)) in groups.iter().zip(cotangent_values.iter()).enumerate()
    {
        if group_index % 256 == 0 {
            replay_checkpoint()?;
        }
        let mut group_values = reserve_replay_buffer(group.len())?;
        for (index, (_, value)) in group.iter().enumerate() {
            if index % 256 == 0 {
                replay_checkpoint()?;
            }
            group_values.push(*value);
        }
        let group_contribution =
            reduction.group_cotangent(effect_index, &group_values, *cotangent, correction)?;
        for (index, ((source_index, _), value)) in
            group.iter().zip(group_contribution.iter()).enumerate()
        {
            if index % 256 == 0 {
                replay_checkpoint()?;
            }
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
    let group_count = shape_size(reduction, target_shape)?;
    let mut groups = reserve_replay_buffer(group_count)?;
    for _ in 0..group_count {
        groups.push(reserve_replay_buffer::<f64>(source_shape[axis])?);
    }
    for (flat_index, value) in source_values.iter().copied().enumerate() {
        if flat_index % 256 == 0 {
            replay_checkpoint()?;
        }
        let source_index = unravel_index(flat_index, source_shape)?;
        let target_index = index_without_axis(&source_index, axis)?;
        let target_flat = ravel_index(reduction, &target_index, target_shape)?;
        groups[target_flat].push(value);
    }
    for (index, group) in groups.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        if group.is_empty() {
            return Err(format!(
                "effect {effect_index} {} axis reduction produced an empty group",
                reduction.label()
            ));
        }
    }
    Ok(groups)
}

fn corrected_moments(
    effect_index: usize,
    label: &str,
    values: &[f64],
    correction: f64,
) -> Result<(f64, f64, f64), String> {
    if values.is_empty() {
        return Err(format!(
            "effect {effect_index} {label} requires non-empty values"
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        if !value.is_finite() {
            return Err(format!(
                "effect {effect_index} {label} source values must be finite"
            ));
        }
    }
    let denominator = validate_moment_group_size(effect_index, label, values.len(), correction)?;
    let count = values.len() as f64;
    let mut total = -0.0_f64;
    for (index, value) in values.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        total += value;
    }
    let mean = total / count;
    let mut centered_sum = -0.0_f64;
    for (index, value) in values.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        let delta = value - mean;
        centered_sum += delta * delta;
    }
    let variance = centered_sum / denominator;
    replay_checkpoint()?;
    if variance.is_finite() && variance >= 0.0 {
        Ok((mean, variance, denominator))
    } else {
        Err(format!(
            "effect {effect_index} {label} corrected variance must be finite and non-negative"
        ))
    }
}

fn validate_correction_scalar(label: &str, correction: f64) -> Result<(), String> {
    if correction.is_finite() && correction >= 0.0 {
        Ok(())
    } else {
        Err(format!(
            "{label} correction metadata must be a finite non-negative scalar"
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
