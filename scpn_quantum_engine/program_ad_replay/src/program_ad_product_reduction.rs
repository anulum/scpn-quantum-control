// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD product reduction replay

//! Product-reduction replay for bounded Program AD IR.
//!
//! The forward pass accepts finite all-axis and static-axis products. Reverse
//! replay implements the exact smooth product derivative, including the
//! single-zero case, and fails closed for groups with two or more zeros.

use crate::program_ad_lifecycle::replay_checkpoint;

/// Evaluate the product over every flattened source value.
pub(crate) fn product_all_value(effect_index: usize, source_values: &[f64]) -> Result<f64, String> {
    let mut product = 1.0_f64;
    for (index, value) in source_values.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        product *= value;
    }
    replay_checkpoint()?;
    validate_finite_product(effect_index, product)?;
    Ok(product)
}

/// Evaluate a static-axis product reduction.
pub(crate) fn product_axis_values(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    target_shape: &[usize],
    source_values: &[f64],
) -> Result<Vec<f64>, String> {
    validate_source_size(source_shape, source_values)?;
    validate_axis_target_shape(effect_index, source_shape, axis, target_shape)?;
    let mut output = filled_product_buffer(shape_size(target_shape)?, 1.0_f64)?;
    for (flat_index, value) in source_values.iter().enumerate() {
        if flat_index % 256 == 0 {
            replay_checkpoint()?;
        }
        let source_index = unravel_index(flat_index, source_shape)?;
        let target_index = index_without_axis(&source_index, axis)?;
        let target_flat = ravel_index(&target_index, target_shape)?;
        output[target_flat] *= value;
        validate_finite_product(effect_index, output[target_flat])?;
    }
    Ok(output)
}

/// Build the all-axis product adjoint contribution.
pub(crate) fn product_all_cotangent(
    effect_index: usize,
    source_values: &[f64],
    scalar_cotangent: f64,
) -> Result<Vec<f64>, String> {
    product_group_cotangent(effect_index, source_values, scalar_cotangent)
}

/// Build the source-shaped adjoint contribution for a static-axis product.
pub(crate) fn product_axis_cotangent(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    cotangent_values: &[f64],
    source_values: &[f64],
) -> Result<Vec<f64>, String> {
    validate_source_size(source_shape, source_values)?;
    let target_shape = axis_reduction_shape(source_shape, axis)?;
    if shape_size(&target_shape)? != cotangent_values.len() {
        return Err(format!(
            "effect {effect_index} prod axis cotangent shape must be {:?}",
            target_shape
        ));
    }
    let mut groups = reserve_product_buffer(cotangent_values.len())?;
    for index in 0..cotangent_values.len() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        groups.push(reserve_product_buffer::<(usize, f64)>(source_shape[axis])?);
    }
    for (flat_index, value) in source_values.iter().copied().enumerate() {
        if flat_index % 256 == 0 {
            replay_checkpoint()?;
        }
        let source_index = unravel_index(flat_index, source_shape)?;
        let target_index = index_without_axis(&source_index, axis)?;
        let target_flat = ravel_index(&target_index, &target_shape)?;
        groups[target_flat].push((flat_index, value));
    }
    let mut contribution = filled_product_buffer(source_values.len(), 0.0_f64)?;
    for (group_index, (group, cotangent)) in groups.iter().zip(cotangent_values.iter()).enumerate()
    {
        if group_index % 256 == 0 {
            replay_checkpoint()?;
        }
        let mut group_values = reserve_product_buffer(group.len())?;
        for (index, (_, value)) in group.iter().enumerate() {
            if index % 256 == 0 {
                replay_checkpoint()?;
            }
            group_values.push(*value);
        }
        let group_contribution = product_group_cotangent(effect_index, &group_values, *cotangent)?;
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

fn product_group_cotangent(
    effect_index: usize,
    source_values: &[f64],
    cotangent: f64,
) -> Result<Vec<f64>, String> {
    let mut zero_count = 0usize;
    for (index, value) in source_values.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        if *value == 0.0 {
            zero_count += 1;
            if zero_count > 1 {
                return Err(format!(
                    "effect {effect_index} prod gradient supports at most one zero input per reduction group"
                ));
            }
        }
    }
    let mut contribution = reserve_product_buffer(source_values.len())?;
    if zero_count == 1 {
        let mut non_zero_product = 1.0_f64;
        for (index, value) in source_values.iter().enumerate() {
            if index % 256 == 0 {
                replay_checkpoint()?;
            }
            if *value != 0.0 {
                non_zero_product *= value;
            }
        }
        validate_finite_product(effect_index, non_zero_product)?;
        for (index, value) in source_values.iter().enumerate() {
            if index % 256 == 0 {
                replay_checkpoint()?;
            }
            contribution.push(if *value == 0.0 {
                cotangent * non_zero_product
            } else {
                0.0
            });
        }
    } else {
        let product = product_all_value(effect_index, source_values)?;
        for (index, value) in source_values.iter().enumerate() {
            if index % 256 == 0 {
                replay_checkpoint()?;
            }
            contribution.push(cotangent * product / value);
        }
    }
    replay_checkpoint()?;
    Ok(contribution)
}

fn validate_source_size(source_shape: &[usize], source_values: &[f64]) -> Result<(), String> {
    let expected = shape_size(source_shape)?;
    if expected != source_values.len() {
        return Err(format!(
            "prod source shape {:?} expects {expected} values, got {}",
            source_shape,
            source_values.len()
        ));
    }
    Ok(())
}

fn validate_axis_target_shape(
    effect_index: usize,
    source_shape: &[usize],
    axis: usize,
    target_shape: &[usize],
) -> Result<(), String> {
    let expected_shape = axis_reduction_shape(source_shape, axis)?;
    if expected_shape != target_shape {
        return Err(format!(
            "effect {effect_index} prod axis reduction target shape must be {:?}, got {:?}",
            expected_shape, target_shape
        ));
    }
    Ok(())
}

fn validate_finite_product(effect_index: usize, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("effect {effect_index} prod result must be finite"))
    }
}

fn axis_reduction_shape(source_shape: &[usize], axis: usize) -> Result<Vec<usize>, String> {
    if axis >= source_shape.len() {
        return Err(format!(
            "prod axis {axis} is outside rank {}",
            source_shape.len()
        ));
    }
    index_without_axis(source_shape, axis)
}

fn shape_size(shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for (index, dimension) in shape.iter().enumerate() {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        if *dimension == 0 {
            return Err("prod shaped values must have non-zero dimensions".to_owned());
        }
        size = size
            .checked_mul(*dimension)
            .ok_or_else(|| "prod shaped value size overflowed".to_owned())?;
    }
    Ok(size)
}

fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Result<Vec<usize>, String> {
    let mut index = filled_product_buffer(shape.len(), 0usize)?;
    for (axis, dimension) in shape.iter().enumerate().rev() {
        if axis % 256 == 0 {
            replay_checkpoint()?;
        }
        if *dimension == 0 {
            return Err("prod index shape dimensions must be positive".to_owned());
        }
        index[axis] = flat_index % dimension;
        flat_index /= dimension;
    }
    Ok(index)
}

fn index_without_axis(index: &[usize], axis: usize) -> Result<Vec<usize>, String> {
    let mut result = reserve_product_buffer(index.len() - usize::from(axis < index.len()))?;
    for (entry_axis, entry) in index.iter().enumerate() {
        if entry_axis % 256 == 0 {
            replay_checkpoint()?;
        }
        if entry_axis != axis {
            result.push(*entry);
        }
    }
    Ok(result)
}

fn ravel_index(index: &[usize], shape: &[usize]) -> Result<usize, String> {
    if index.len() != shape.len() {
        return Err(format!(
            "prod index rank {} does not match shape rank {}",
            index.len(),
            shape.len()
        ));
    }
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (axis, (coordinate, dimension)) in index.iter().zip(shape.iter()).rev().enumerate() {
        if axis % 256 == 0 {
            replay_checkpoint()?;
        }
        if coordinate >= dimension {
            return Err(format!(
                "prod coordinate {coordinate} is outside dimension {dimension}"
            ));
        }
        flat = coordinate
            .checked_mul(stride)
            .and_then(|offset| flat.checked_add(offset))
            .ok_or_else(|| "prod ravel offset overflowed".to_owned())?;
        stride = stride
            .checked_mul(*dimension)
            .ok_or_else(|| "prod ravel stride overflowed".to_owned())?;
    }
    Ok(flat)
}

fn reserve_product_buffer<T>(count: usize) -> Result<Vec<T>, String> {
    replay_checkpoint()?;
    count
        .checked_mul(std::mem::size_of::<T>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "prod buffer bytes exceed native addressability".to_owned())?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(count)
        .map_err(|error| format!("prod buffer allocation refused: {error}"))?;
    Ok(buffer)
}

fn filled_product_buffer<T: Copy>(count: usize, value: T) -> Result<Vec<T>, String> {
    let mut buffer = reserve_product_buffer(count)?;
    for index in 0..count {
        if index % 256 == 0 {
            replay_checkpoint()?;
        }
        buffer.push(value);
    }
    replay_checkpoint()?;
    Ok(buffer)
}
