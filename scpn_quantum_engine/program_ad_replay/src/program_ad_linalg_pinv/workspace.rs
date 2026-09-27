// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Pseudoinverse numeric workspace admission

/// Declare bounded pseudoinverse heap buffers before forward or reverse numeric replay.
pub(crate) fn pinv_workspace_bytes(
    effect_index: usize,
    operation: &str,
    input_count: usize,
    requires_adjoint: bool,
) -> Result<usize, String> {
    let (rows, cols, _, _, _) = validate_pinv_layout(
        effect_index, operation, input_count, requires_adjoint,
    )?;
    let source = matrix_entry_count(rows, cols)?;
    let entries = if requires_adjoint {
        let row_projector = matrix_entry_count(rows, rows)?;
        let col_projector = matrix_entry_count(cols, cols)?;
        // Source, parsed copy, pseudoinverse, cotangent, all VJP transposes/terms
        // and the final adjoint remain live until their enclosing scopes end.
        let retained_vjp = pinv_live_entry_sum(&[(source, 12), (row_projector, 5), (col_projector, 2)])?;
        // Projector subtraction holds identity, product and output simultaneously.
        // For a very wide source this can exceed the final retained VJP inventory.
        let left_construction = pinv_live_entry_sum(&[(source, 4), (col_projector, 3)])?;
        let right_construction = pinv_live_entry_sum(&[(source, 4), (col_projector, 1), (row_projector, 3)])?;
        retained_vjp.max(left_construction).max(right_construction)
    } else {
        // Rank one has source, parsed copy and output. Rank two also transposes.
        let copies = if rows == 1 || cols == 1 { 3 } else { 4 };
        pinv_live_entry_sum(&[(source, copies)])?
    };
    entries.checked_mul(std::mem::size_of::<f64>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "pinv workspace exceeds native addressable memory".to_owned())
}

fn pinv_live_entry_sum(buffers: &[(usize, usize)]) -> Result<usize, String> {
    let mut entries = 0usize;
    for &(length, copies) in buffers {
        replay_checkpoint()?;
        entries = length.checked_mul(copies)
            .and_then(|count| entries.checked_add(count))
            .ok_or_else(|| "pinv workspace exceeds native addressable memory".to_owned())?;
    }
    Ok(entries)
}
