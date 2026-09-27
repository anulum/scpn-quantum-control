// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD cumulative replay

/// Declare numeric and index buffers for the selected compact output before replay.
pub(crate) fn cumulative_workspace_bytes(
    effect_index: usize,
    operation: &str,
    input_count: usize,
    requires_adjoint: bool,
) -> Result<usize, String> {
    let layout = parse_cumulative_layout(effect_index, operation, input_count)?;
    let refusal = || "cumulative workspace exceeds native addressable memory".to_owned();
    let source_bytes = layout.source_size.checked_mul(if requires_adjoint { 2 } else { 1 })
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(refusal)?;
    let (rank_copies, index_width) = match (layout.kind, layout.axis) {
        (CumulativeKind::Diff, _) => (4, std::mem::size_of::<(usize, f64)>()),
        (_, CumulativeAxis::Axis(_)) => (3, std::mem::size_of::<usize>()),
        (_, CumulativeAxis::Flat) => (1, std::mem::size_of::<usize>()),
    };
    // Rank copies cover the spec, output shape and live coordinate vectors;
    // selected prefix indices or difference terms coexist with these buffers.
    let shape_bytes = layout.rank.checked_mul(rank_copies)
        .and_then(|count| count.checked_mul(std::mem::size_of::<usize>()))
        .ok_or_else(refusal)?;
    let index_bytes = layout.index_count.checked_mul(index_width).ok_or_else(refusal)?;
    source_bytes.checked_add(shape_bytes).and_then(|bytes| bytes.checked_add(index_bytes))
        .filter(|bytes| *bytes <= isize::MAX as usize).ok_or_else(refusal)
}
