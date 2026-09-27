// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Multi-dot numeric workspace admission

/// Declare flattened operands and peak left-associated multi-dot kernel storage.
pub(crate) fn multi_dot_workspace_bytes(
    effect_index: usize,
    operation: &str,
    input_count: usize,
    requires_adjoint: bool,
) -> Result<usize, String> {
    let metadata = parse_multi_dot_metadata(effect_index, operation, input_count)?;
    let mut current_shape = copy_chain_buffer(&metadata.operand_shapes[0])?;
    let mut chain_peak = shape_size(&current_shape)?;
    for next_shape in &metadata.operand_shapes[1..] {
        replay_checkpoint()?;
        let next_output = multi_dot_step_shape(effect_index, &current_shape, next_shape)?;
        let output_count = shape_size(&next_output)?;
        let live = shape_size(&current_shape)?.checked_add(shape_size(next_shape)?)
            .and_then(|count| count.checked_add(output_count))
            .ok_or_else(|| "multi_dot workspace size overflowed".to_owned())?;
        chain_peak = chain_peak.max(live);
        current_shape = next_output;
    }
    // Reverse keeps source, varied source, reserved adjoints, output and cotangent.
    let retained_count = if requires_adjoint {
        input_count.checked_mul(3)
            .and_then(|count| metadata.output_size.checked_mul(2)?.checked_add(count))
    } else { Some(input_count) };
    retained_count.and_then(|count| count.checked_add(chain_peak))
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "multi_dot workspace exceeds native addressable memory".to_owned())
}
