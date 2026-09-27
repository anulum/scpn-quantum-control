// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD diagflat linalg replay helpers

//! Bounded diagonal-construction replay helpers for Program AD effect IR.
//!
//! Python Program AD emits one `linalg:diagflat:<shape>:offset:<k>:construct:<i>`
//! SSA node per on-diagonal element when `np.diagflat` is traced; off-diagonal
//! zeros are plain constants and never reach this module. Each node carries the
//! single flattened source operand, so the value replay is the identity map and
//! the reverse contribution passes the output cotangent straight to the source
//! element. The module fails closed on malformed metadata, wrong operand
//! counts, out-of-range source indices, and non-finite inputs or cotangents so
//! broader diagonal-family claims stay honest.

use crate::program_ad_ir::reserve_replay_buffer;
use crate::program_ad_lifecycle::replay_checkpoint;

/// Return whether an operation label belongs to bounded diagflat replay.
pub(crate) fn is_diagflat_operation(operation: &str) -> bool {
    operation.starts_with("linalg:diagflat:")
}

/// Evaluate one on-diagonal element from a diagflat Program AD node.
pub(crate) fn diagflat_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    parse_diagflat(effect_index, operation, input_values)
}

/// Return local reverse contributions for one on-diagonal diagflat node.
pub(crate) fn diagflat_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} diagflat cotangent must be finite"
        ));
    }
    parse_diagflat(effect_index, operation, input_values)?;
    let mut contribution = reserve_replay_buffer(1)?;
    contribution.push(output_cotangent);
    replay_checkpoint()?;
    Ok(contribution)
}

fn parse_diagflat(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    validate_diagflat_layout(effect_index, operation, input_values.len())?;
    let [value] = input_values else {
        return Err(format!(
            "effect {effect_index} diagflat replay requires exactly one source operand"
        ));
    };
    if !value.is_finite() {
        return Err(format!(
            "effect {effect_index} diagflat input must be finite"
        ));
    }
    Ok(*value)
}

fn validate_diagflat_layout(
    effect_index: usize,
    operation: &str,
    input_count: usize,
) -> Result<(), String> {
    replay_checkpoint()?;
    for _ in operation.as_bytes().chunks(256) { replay_checkpoint()?; }
    let mut fields = operation.split(':');
    let mut parts = [""; 7];
    for part in &mut parts {
        *part = fields.next().ok_or_else(|| format!("effect {effect_index} diagflat operation metadata is malformed"))?;
    }
    if fields.next().is_some()
        || parts[0] != "linalg"
        || parts[1] != "diagflat"
        || parts[3] != "offset"
        || parts[5] != "construct"
    {
        return Err(format!(
            "effect {effect_index} diagflat operation metadata is malformed"
        ));
    }
    let source_size = parse_shape_size(effect_index, parts[2])?;
    let offset = parts[4]
        .parse::<i64>()
        .map_err(|_| format!("effect {effect_index} diagflat offset metadata is malformed"))?;
    let output_size = offset.checked_abs()
        .and_then(|offset| usize::try_from(offset).ok())
        .and_then(|shift| source_size.checked_add(shift))
        .ok_or_else(|| format!("effect {effect_index} diagflat construct shape overflows"))?;
    output_size.checked_mul(output_size)
        .filter(|entries| *entries <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| format!("effect {effect_index} diagflat output bytes exceed native addressability"))?;
    let source_index = parts[6].parse::<usize>().map_err(|_| {
        format!("effect {effect_index} diagflat source index metadata is malformed")
    })?;
    if source_index >= source_size {
        return Err(format!(
            "effect {effect_index} diagflat source index is outside the flattened source shape"
        ));
    }
    if input_count != 1 {
        return Err(format!("effect {effect_index} diagflat replay requires exactly one source operand"));
    }
    Ok(())
}

/// Declare the actual scalar source and reverse contribution for compact diagflat replay.
pub(crate) fn diagflat_workspace_bytes(
    effect_index: usize,
    operation: &str,
    input_count: usize,
    requires_adjoint: bool,
) -> Result<usize, String> {
    validate_diagflat_layout(effect_index, operation, input_count)?;
    let copies = if requires_adjoint { 2 } else { 1 };
    input_count.checked_mul(copies)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "diagflat workspace exceeds native addressable memory".to_owned())
}

fn parse_shape_size(effect_index: usize, label: &str) -> Result<usize, String> {
    if label.is_empty() {
        return Err(format!(
            "effect {effect_index} diagflat shape metadata is malformed"
        ));
    }
    let mut size: usize = 1;
    for part in label.split('x') {
        replay_checkpoint()?;
        let dimension = part
            .parse::<usize>()
            .map_err(|_| format!("effect {effect_index} diagflat shape metadata is malformed"))?;
        size = size
            .checked_mul(dimension)
            .filter(|entries| *entries <= isize::MAX as usize / std::mem::size_of::<f64>())
            .ok_or_else(|| format!("effect {effect_index} diagflat shape metadata overflows"))?;
    }
    Ok(size)
}
