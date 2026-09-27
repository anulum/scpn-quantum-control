// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD cumulative replay

fn output_shape(effect_index: usize, spec: &CumulativeSpec) -> Result<Vec<usize>, String> {
    match spec.kind {
        CumulativeKind::Cumsum | CumulativeKind::Cumprod => match spec.axis {
            CumulativeAxis::Flat => {
                let mut shape = cumulative_buffer(1)?;
                shape.push(shape_size(&spec.source_shape)?);
                Ok(shape)
            },
            CumulativeAxis::Axis(_) => copy_cumulative_buffer(&spec.source_shape),
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
            let mut shape = copy_cumulative_buffer(&spec.source_shape)?;
            shape[axis] = axis_size - spec.order;
            Ok(shape)
        }
    }
}

fn prefix_indices(effect_index: usize, spec: &CumulativeSpec) -> Result<Vec<usize>, String> {
    match spec.axis {
        CumulativeAxis::Flat => {
            let count = spec.output_index.checked_add(1)
                .ok_or_else(|| "cumulative prefix count overflowed".to_owned())?;
            let mut indices = cumulative_buffer(count)?;
            for index in 0..=spec.output_index {
                if index.is_multiple_of(256) { replay_checkpoint()?; }
                indices.push(index);
            }
            Ok(indices)
        },
        CumulativeAxis::Axis(axis) => {
            let target_index = unravel_index(spec.output_index, &spec.source_shape)?;
            let count = target_index[axis].checked_add(1)
                .ok_or_else(|| "cumulative prefix count overflowed".to_owned())?;
            let mut indices = cumulative_buffer(count)?;
            let mut source_index = copy_cumulative_buffer(&target_index)?;
            for axis_index in 0..=target_index[axis] {
                replay_checkpoint()?;
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
    let output_index = unravel_index(spec.output_index, &shape)?;
    let count = spec.order.checked_add(1)
        .ok_or_else(|| "cumulative difference term count overflowed".to_owned())?;
    let mut terms = cumulative_buffer(count)?;
    let mut source_index = copy_cumulative_buffer(&output_index)?;
    for offset in 0..=spec.order {
        replay_checkpoint()?;
        source_index[axis] = output_index[axis].checked_add(offset)
            .ok_or_else(|| "cumulative difference coordinate overflowed".to_owned())?;
        let coefficient = binomial(spec.order, offset)? as f64
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

fn binomial(n: usize, k: usize) -> Result<usize, String> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut accumulator = 1usize;
    for index in 0..k {
        replay_checkpoint()?;
        let mut numerator = n - index;
        let mut denominator = index + 1;
        let common = cumulative_gcd(numerator, denominator);
        numerator /= common;
        denominator /= common;
        let common = cumulative_gcd(accumulator, denominator);
        accumulator /= common;
        denominator /= common;
        if denominator != 1 {
            return Err("cumulative binomial coefficient is not integral".to_owned());
        }
        accumulator = accumulator.checked_mul(numerator)
            .ok_or_else(|| "cumulative binomial coefficient overflowed".to_owned())?;
    }
    Ok(accumulator)
}

fn cumulative_gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn shape_size(shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for dimension in shape {
        replay_checkpoint()?;
        size = size
            .checked_mul(*dimension)
            .ok_or_else(|| "cumulative shaped value size overflowed".to_owned())?;
    }
    Ok(size)
}

fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Result<Vec<usize>, String> {
    if shape.contains(&0) || flat_index >= shape_size(shape)? {
        return Err("cumulative flattened index is outside shape".to_owned());
    }
    let mut index = cumulative_buffer(shape.len())?;
    index.resize(shape.len(), 0usize);
    for (axis, dimension) in shape.iter().enumerate().rev() {
        replay_checkpoint()?;
        index[axis] = flat_index % dimension;
        flat_index /= dimension;
    }
    Ok(index)
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
        replay_checkpoint()?;
        if coordinate >= dimension {
            return Err(format!(
                "cumulative coordinate {coordinate} is outside dimension {dimension}"
            ));
        }
        let term = coordinate.checked_mul(stride)
            .ok_or_else(|| "cumulative coordinate product overflowed".to_owned())?;
        flat = flat.checked_add(term)
            .ok_or_else(|| "cumulative flattened index overflowed".to_owned())?;
        stride = stride
            .checked_mul(*dimension)
            .ok_or_else(|| "cumulative stride overflowed".to_owned())?;
    }
    Ok(flat)
}
