// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Interpolation numeric workspace admission

/// Declare source, static grid and optional reverse contribution before replay.
pub(crate) fn interpolation_workspace_bytes(
    effect_index: usize,
    operation: &str,
    input_count: usize,
    requires_adjoint: bool,
) -> Result<usize, String> {
    // This shared metadata pass validates grid values without allocating a grid.
    let layout = parse_interpolation_layout(effect_index, operation, input_count)?;
    let copies = if requires_adjoint { 2 } else { 1 };
    input_count.checked_mul(copies)
        .and_then(|count| count.checked_add(layout.grid_count))
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "interpolation workspace exceeds native addressable memory".to_owned())
}
