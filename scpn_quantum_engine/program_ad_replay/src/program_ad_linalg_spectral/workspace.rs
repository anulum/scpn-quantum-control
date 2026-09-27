// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Bounded spectral numeric workspace admission

/// Declare flattened 2x2 source and optional reverse contribution heap buffers.
pub(crate) fn spectral_workspace_bytes(
    effect_index: usize,
    operation: &str,
    input_count: usize,
    requires_adjoint: bool,
) -> Result<usize, String> {
    let family = if is_eigvalsh_operation(operation) { "eigvalsh" }
        else if is_eigvals_operation(operation) { "eigvals" }
        else if is_eig_operation(operation) { "eig" }
        else if is_eigh_operation(operation) { "eigh" }
        else { return Err(format!("effect {effect_index} operation {operation} is outside bounded spectral replay")); };
    if input_count != 4 {
        return Err(format!("effect {effect_index} {family} Rust replay supports only 2x2 matrices"));
    }
    match family {
        "eigvalsh" => { parse_eigvalsh_index(effect_index, operation)?; }
        "eigvals" => { parse_eigvals_index(effect_index, operation)?; }
        "eig" => { parse_eig_output(effect_index, operation)?; }
        _ => { parse_eigh_output(effect_index, operation)?; }
    }
    // Eigenpair algebra uses fixed stack arrays. Heap replay retains only the
    // flattened source; reverse also builds the four local cotangent entries.
    let copies = if requires_adjoint { 2 } else { 1 };
    input_count.checked_mul(copies)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "spectral workspace exceeds native addressable memory".to_owned())
}
