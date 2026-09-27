// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Shaped reduction workspace admission

fn shaped_reduction_workspace_bytes(effect: &ProgramADEffect, operation: &str, shapes: &ProgramADShapeMap<'_>) -> Result<usize, String> {
    let label = operation.split(':').next().unwrap_or_default();
    if effect.inputs.len() != 1 { return Err(format!("effect {} {label} requires one input", effect.index)); }
    let source_shape = metadata_operand_shape(&effect.inputs[0], shapes)?;
    let target_shape = shapes.get(effect.target.as_str()).ok_or_else(|| {
        format!("effect {} target {} is missing SSA shape metadata", effect.index, effect.target)
    })?;
    let axis = if label == "prod" {
        if operation == "prod" { None } else { Some(parse_static_axis(operation, "prod", source_shape.len())?) }
    } else {
        let metadata = parse_moment_reduction_metadata(operation, label, source_shape.len())?;
        let count = match metadata.axis {
            Some(axis) => source_shape[axis], None => shape_size(source_shape)?,
        };
        crate::program_ad_variance_reduction::validate_moment_group_size(effect.index, label, count, metadata.correction)?;
        metadata.axis
    };
    validate_shaped_reduction_target(effect.index, label, source_shape, target_shape, axis)?;
    let source = shape_size(source_shape)?;
    let output = shape_size(target_shape)?;
    let rank = source_shape.len();
    let target_rank = target_shape.len();
    let accumulation = reduction_buffer_bytes(
        &[source, source, source, output], &[rank, rank, rank, rank, rank, target_rank], &[],
    )?;
    let Some(axis) = axis else {
        // All-axis kernels keep numeric statistics/products on the stack.
        // Adapter source, output/contribution and cloned cotangent storage are
        // dominated by the conservative accumulation bound above.
        return Ok(accumulation);
    };
    let axis_size = source_shape[axis];
    let pairs = (source, std::mem::size_of::<(usize, f64)>());
    let group_headers = (output, std::mem::size_of::<Vec<(usize, f64)>>());
    // Pair groups, source/contribution/cotangent and the current group values
    // and VJP are simultaneously live. Coordinate peak is included conservatively.
    let reverse = reduction_buffer_bytes(
        &[source, source, output, axis_size, axis_size],
        &[rank, rank, target_rank, target_rank, target_rank], &[pairs, group_headers],
    )?;
    let forward = if label == "prod" {
        reduction_buffer_bytes(&[source, output], &[rank, rank, target_rank, target_rank], &[])?
    } else {
        // Moments retain a second source-sized grouping and its outer Vec headers.
        reduction_buffer_bytes(&[source, source, output], &[rank, rank, target_rank, target_rank],
            &[(output, std::mem::size_of::<Vec<f64>>())])?
    };
    Ok(accumulation.max(reverse).max(forward))
}

fn validate_shaped_reduction_target(effect: usize, label: &str, source: &[usize], target: &[usize], axis: Option<usize>) -> Result<(), String> {
    let Some(axis) = axis else {
        if target.is_empty() { return Ok(()); }
        return Err(format!("effect {effect} {label} non-scalar target requires static axis metadata {label}:axis:<int>"));
    };
    if axis >= source.len() { return Err(format!("{label} axis {axis} is outside rank {}", source.len())); }
    if target.len() != source.len()-1 {
        return Err(format!("effect {effect} {label} axis reduction target shape must remove axis {axis} from {:?}, got {:?}", source, target));
    }
    let mut target_axis = 0usize;
    for (index, dimension) in source.iter().enumerate() {
        crate::program_ad_lifecycle::replay_checkpoint()?;
        if index != axis {
            if target[target_axis] != *dimension {
                return Err(format!("effect {effect} {label} axis reduction target shape must remove axis {axis} from {:?}, got {:?}", source, target));
            }
            target_axis += 1;
        }
    }
    Ok(())
}

fn reduction_buffer_bytes(floats: &[usize], indices: &[usize], other: &[(usize, usize)]) -> Result<usize, String> {
    let refusal = || "reduction workspace exceeds native addressable memory".to_owned();
    let mut bytes = 0usize;
    for (counts, width) in [(floats, std::mem::size_of::<f64>()), (indices, std::mem::size_of::<usize>())] {
        for count in counts {
            crate::program_ad_lifecycle::replay_checkpoint()?;
            bytes = count.checked_mul(width).and_then(|size| bytes.checked_add(size)).ok_or_else(refusal)?;
        }
    }
    for (count, width) in other {
        crate::program_ad_lifecycle::replay_checkpoint()?;
        bytes = count.checked_mul(*width).and_then(|size| bytes.checked_add(size)).ok_or_else(refusal)?;
    }
    if bytes > isize::MAX as usize { return Err(refusal()); }
    Ok(bytes)
}


fn order_statistic_admission_bytes(effect: &ProgramADEffect, operation: &str, shapes: &ProgramADShapeMap<'_>) -> Result<usize, String> {
    let label = operation.split(':').next().unwrap_or_default();
    if effect.inputs.len() != 1 { return Err(format!("effect {} {operation} requires one input", effect.index)); }
    let source_shape = metadata_operand_shape(&effect.inputs[0], shapes)?;
    let target_shape = shapes.get(effect.target.as_str()).ok_or_else(|| {
        format!("effect {} target {} is missing SSA shape metadata", effect.index, effect.target)
    })?;
    let axis = crate::program_ad_order_statistic_reduction::order_statistic_reduction_axis(
        effect.index, operation, source_shape.len(),
    )?;
    validate_shaped_reduction_target(effect.index, label, source_shape, target_shape, axis)?;
    let source = shape_size(source_shape)?;
    let output = shape_size(target_shape)?;
    let rank = source_shape.len(); let target_rank = target_shape.len();
    let accumulation = reduction_buffer_bytes(
        &[source, source, source, output], &[rank, rank, rank, rank, rank, target_rank], &[],
    )?;
    let group_size = axis.map_or(source, |axis| source_shape[axis]);
    let pair_width = std::mem::size_of::<(usize, f64)>();
    let scratch_width = std::mem::size_of::<f64>().max(std::mem::size_of::<usize>());
    let refusal = || "reduction workspace exceeds native addressable memory".to_owned();
    // Value validation and index ordering use separate scratch vectors; the
    // two-entry selected VJP is created after ordering storage has been dropped.
    let scratch = group_size.checked_mul(scratch_width).ok_or_else(refusal)?
        .max(pair_width.checked_mul(2).ok_or_else(refusal)?);
    let headers = if axis.is_some() { output } else { 0 };
    let coordinates = rank.checked_mul(if axis.is_some() { 2 } else { 1 })
        .and_then(|count| target_rank.checked_mul(3).and_then(|target| count.checked_add(target)))
        .ok_or_else(refusal)?;
    // Reverse source/contribution/cotangent dominate forward source/output.
    // Axis groups also retain their outer Vec headers and construction coordinates.
    let kernel = reduction_buffer_bytes(
        &[source, source, output], &[coordinates],
        &[(source, pair_width), (headers, std::mem::size_of::<Vec<(usize, f64)>>()), (1, scratch)],
    )?;
    Ok(accumulation.max(kernel))
}
