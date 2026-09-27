// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD stencil metadata

/// Declare flattened source, shape/index, coordinates and optional reverse buffers.
pub(crate) fn stencil_workspace_bytes(effect_index: usize, operation: &str, input_count: usize, requires_adjoint: bool) -> Result<usize, String> {
    let layout = parse_stencil_layout(effect_index, operation, input_count)?;
    let coordinates = match layout.spacing { StencilSpacingLabel::Scalar(_) => 0, StencilSpacingLabel::Coordinates(_) => layout.axis_size };
    let refusal = || "stencil workspace exceeds native addressable memory".to_owned();
    let numeric = layout.source_size.checked_mul(if requires_adjoint {2} else {1})
        .and_then(|count| count.checked_add(coordinates))
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>())).ok_or_else(refusal)?;
    let indices = layout.rank.checked_mul(2).and_then(|count| count.checked_mul(std::mem::size_of::<usize>())).ok_or_else(refusal)?;
    numeric.checked_add(indices).filter(|bytes| *bytes <= isize::MAX as usize).ok_or_else(refusal)
}
