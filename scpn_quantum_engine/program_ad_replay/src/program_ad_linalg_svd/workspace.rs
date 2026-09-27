// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Singular-value retained matrix admission

/// Declare source/matrix/vectors, nalgebra 0.35 bidiagonal scratch and contributions.
///
/// Conservatively sum buffers from all solver phases, rather than a measured peak.
/// Bidiagonal stores k/k-1 coefficients and rows/cols work; VT adds k/cols work;
/// copied off-diagonal adds k-1 entries, while copied diagonal is the spectrum.
pub(crate) fn svdvals_workspace_bytes(
    effect_index: usize,
    operation: &str,
    input_count: usize,
    requires_adjoint: bool,
) -> Result<usize, String> {
    let (rows, cols, _) = validate_svdvals_layout(effect_index, operation, input_count)?;
    let spectrum = rows.min(cols);
    let source = matrix_entry_count(effect_index, rows, cols)?;
    let left = matrix_entry_count(effect_index, rows, spectrum)?;
    let right = matrix_entry_count(effect_index, spectrum, cols)?;
    let scratch = spectrum.checked_mul(4)
        .and_then(|entries| entries.checked_sub(2))
        .and_then(|entries| entries.checked_add(rows))
        .and_then(|entries| cols.checked_mul(2).and_then(|cols| entries.checked_add(cols)))
        .ok_or_else(|| format!("effect {effect_index} svdvals solver scratch exceeds native addressability"))?;
    let copies = if requires_adjoint { 3 } else { 2 };
    source.checked_mul(copies)
        .and_then(|entries| entries.checked_add(left))
        .and_then(|entries| entries.checked_add(right))
        .and_then(|entries| entries.checked_add(spectrum))
        .and_then(|entries| entries.checked_add(scratch))
        .and_then(|entries| entries.checked_mul(std::mem::size_of::<f64>()))
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| format!("effect {effect_index} svdvals retained workspace exceeds native addressability"))
}

fn validate_svdvals_layout(
    effect_index: usize,
    operation: &str,
    input_count: usize,
) -> Result<(usize, usize, usize), String> {
    let (rows, cols, output_index) = parse_svdvals_metadata(effect_index, operation)?;
    let spectrum = rows.min(cols);
    if output_index >= spectrum {
        return Err(format!("effect {effect_index} svdvals output index is outside the singular-value spectrum"));
    }
    let source = matrix_entry_count(effect_index, rows, cols)?;
    let left = matrix_entry_count(effect_index, rows, spectrum)?;
    let right = matrix_entry_count(effect_index, spectrum, cols)?;
    source.checked_add(left)
        .and_then(|entries| entries.checked_add(right))
        .and_then(|entries| entries.checked_add(spectrum))
        .filter(|entries| *entries <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| format!("effect {effect_index} svdvals retained matrix bytes exceed native addressability"))?;
    if input_count != source {
        return Err(format!("effect {effect_index} svdvals requires {source} flattened matrix operands"));
    }
    Ok((rows, cols, output_index))
}
