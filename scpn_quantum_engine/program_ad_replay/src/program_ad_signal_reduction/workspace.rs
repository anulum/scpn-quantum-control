// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Compact signal numeric workspace admission

/// Declare compact signal source copies and optional reverse contribution storage.
pub(crate) fn signal_workspace_bytes(
    effect_index: usize,
    operation: &str,
    input_count: usize,
    requires_adjoint: bool,
) -> Result<usize, String> {
    let spec = parse_signal_operation(effect_index, operation)?;
    let source_count = validate_signal_source_count(effect_index, &spec, input_count)?;
    full_output_index(effect_index, &spec)?;
    // Pair indices are streamed. Only the flattened source is held forward;
    // reverse also retains the full left/right cotangent contribution buffer.
    let copies = if requires_adjoint { 2 } else { 1 };
    source_count.checked_mul(copies)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "signal workspace exceeds native addressable memory".to_owned())
}
