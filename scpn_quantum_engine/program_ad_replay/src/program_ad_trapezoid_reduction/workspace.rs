// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Trapezoid replay workspace admission

/// Declare grid, coordinate and adapter buffers for shaped value/gradient replay.
pub(crate) fn trapezoid_workspace_bytes(
    effect_index: usize,
    operation: &str,
    source_shape: &[usize],
    target_shape: &[usize],
) -> Result<usize, String> {
    let layout = parse_trapezoid_layout(effect_index, operation, source_shape)?;
    let rank = source_shape.len();
    let target_rank = rank.checked_sub(1).ok_or_else(|| "trapezoid requires ranked source values".to_owned())?;
    let mut target_axis = 0usize;
    if target_shape.len() != target_rank {
        return Err(format!("effect {effect_index} trapezoid target shape must remove integration axis {} from {:?}, got {:?}", layout.axis, source_shape, target_shape));
    }
    for (axis, dimension) in source_shape.iter().enumerate() {
        replay_checkpoint()?;
        if axis != layout.axis {
            if target_shape[target_axis] != *dimension {
                return Err(format!("effect {effect_index} trapezoid target shape must remove integration axis {} from {:?}, got {:?}", layout.axis, source_shape, target_shape));
            }
            target_axis += 1;
        }
    }
    let source = shape_size(source_shape)?;
    let output = shape_size(target_shape)?;
    let forward = trapezoid_buffer_bytes(
        &[source, output, layout.grid_count], &[rank, target_rank, target_rank],
    )?;
    // Kernel phase: source copy, contribution, grid and cloned cotangent coexist;
    // shapes cover source, target, selected target coordinates and cotangent.
    let kernel = trapezoid_buffer_bytes(
        &[source, source, output, layout.grid_count],
        &[rank, target_rank, target_rank, target_rank],
    )?;
    // Accumulation phase: source, contribution and reduced contribution coexist.
    // Five rank buffers conservatively cover source/contribution/reduced shapes,
    // inferred broadcast shape and traversal coordinates. Grid has been dropped.
    let accumulation = trapezoid_buffer_bytes(
        &[source, source, source, output], &[rank, rank, rank, rank, rank, target_rank],
    )?;
    Ok(forward.max(kernel).max(accumulation))
}

fn trapezoid_buffer_bytes(floats: &[usize], indices: &[usize]) -> Result<usize, String> {
    let refusal = || "trapezoid workspace exceeds native addressable memory".to_owned();
    let mut bytes = 0usize;
    for (counts, width) in [(floats, std::mem::size_of::<f64>()), (indices, std::mem::size_of::<usize>())] {
        for count in counts {
            replay_checkpoint()?;
            bytes = count.checked_mul(width).and_then(|size| bytes.checked_add(size)).ok_or_else(refusal)?;
        }
    }
    if bytes > isize::MAX as usize { return Err(refusal()); }
    Ok(bytes)
}
