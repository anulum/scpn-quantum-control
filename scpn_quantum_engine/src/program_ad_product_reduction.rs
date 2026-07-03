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

/// Evaluate the product over every flattened source value.
pub(crate) fn product_all_value(effect_index: usize, source_values: &[f64]) -> Result<f64, String> {
    let product = source_values.iter().copied().product::<f64>();
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
    let mut output = vec![1.0_f64; shape_size(target_shape)?];
    for (flat_index, value) in source_values.iter().enumerate() {
        let source_index = unravel_index(flat_index, source_shape);
        let target_index = index_without_axis(&source_index, axis);
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
    let mut groups = vec![Vec::<(usize, f64)>::new(); cotangent_values.len()];
    for (flat_index, value) in source_values.iter().copied().enumerate() {
        let source_index = unravel_index(flat_index, source_shape);
        let target_index = index_without_axis(&source_index, axis);
        let target_flat = ravel_index(&target_index, &target_shape)?;
        groups[target_flat].push((flat_index, value));
    }
    let mut contribution = vec![0.0_f64; source_values.len()];
    for (group, cotangent) in groups.iter().zip(cotangent_values.iter()) {
        let group_values = group.iter().map(|(_, value)| *value).collect::<Vec<f64>>();
        let group_contribution = product_group_cotangent(effect_index, &group_values, *cotangent)?;
        for ((source_index, _), value) in group.iter().zip(group_contribution.iter()) {
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
    let zero_count = source_values.iter().filter(|value| **value == 0.0).count();
    if zero_count > 1 {
        return Err(format!(
            "effect {effect_index} prod gradient supports at most one zero input per reduction group"
        ));
    }
    if zero_count == 1 {
        let non_zero_product = source_values
            .iter()
            .filter(|value| **value != 0.0)
            .copied()
            .product::<f64>();
        validate_finite_product(effect_index, non_zero_product)?;
        return Ok(source_values
            .iter()
            .map(|value| {
                if *value == 0.0 {
                    cotangent * non_zero_product
                } else {
                    0.0
                }
            })
            .collect());
    }
    let product = product_all_value(effect_index, source_values)?;
    Ok(source_values
        .iter()
        .map(|value| cotangent * product / value)
        .collect())
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
    Ok(source_shape
        .iter()
        .enumerate()
        .filter_map(|(index, dimension)| (index != axis).then_some(*dimension))
        .collect())
}

fn shape_size(shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for dimension in shape {
        if *dimension == 0 {
            return Err("prod shaped values must have non-zero dimensions".to_owned());
        }
        size = size
            .checked_mul(*dimension)
            .ok_or_else(|| "prod shaped value size overflowed".to_owned())?;
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
    for (coordinate, dimension) in index.iter().zip(shape.iter()).rev() {
        if coordinate >= dimension {
            return Err(format!(
                "prod coordinate {coordinate} is outside dimension {dimension}"
            ));
        }
        flat += coordinate * stride;
        stride = stride
            .checked_mul(*dimension)
            .ok_or_else(|| "prod ravel stride overflowed".to_owned())?;
    }
    Ok(flat)
}
