// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Matrix-power numeric workspace admission

/// Declare numeric source copies and peak matrix-power kernel storage before replay.
pub(crate) fn matrix_power_workspace_bytes(
    effect_index: usize,
    operation: &str,
    input_count: usize,
    requires_adjoint: bool,
) -> Result<usize, String> {
    let metadata = parse_matrix_power_metadata(effect_index, operation, input_count)?;
    let entries = matrix_entries(metadata.size)?;
    let count = if metadata.exponent < 0 {
        exponent_magnitude(effect_index, metadata.exponent)?
    } else {
        exponent_count(effect_index, metadata.exponent)?
    };
    // Forward: source, old accumulator, new product; inverse storage for negative powers.
    // Reverse also retains prefix powers and the nested product/addition temporaries.
    let matrices = match (requires_adjoint, metadata.exponent.cmp(&0)) {
        (false, std::cmp::Ordering::Equal) => 2,
        (false, std::cmp::Ordering::Greater) => 3,
        (false, std::cmp::Ordering::Less) => 4,
        (true, std::cmp::Ordering::Equal) => 3,
        (true, std::cmp::Ordering::Greater) => 8,
        (true, std::cmp::Ordering::Less) => 9,
    };
    let retained = if requires_adjoint && count != 0 {
        retained_power_bytes(entries, count)?
    } else { 0 };
    entries.checked_mul(std::mem::size_of::<f64>())
        .and_then(|bytes| bytes.checked_mul(matrices))
        .and_then(|bytes| bytes.checked_add(retained))
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "matrix_power workspace exceeds native addressable memory".to_owned())
}

fn retained_power_bytes(entries: usize, count: usize) -> Result<usize, String> {
    entries.checked_mul(std::mem::size_of::<f64>())
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<f64>>()))
        .and_then(|bytes| bytes.checked_mul(count))
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "matrix_power retained powers exceed native addressable memory".to_owned())
}
