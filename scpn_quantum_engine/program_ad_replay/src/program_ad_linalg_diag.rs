// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD diag linalg replay helpers

//! Bounded diagonal gather/scatter replay helpers for Program AD effect IR.
//!
//! Python Program AD emits one
//! `linalg:diag:<shape>:offset:<k>:construct:<source>` SSA node for each
//! differentiable on-diagonal output when `np.diag(vector, k=...)` constructs a
//! matrix, and one `linalg:diag:<shape>:offset:<k>:extract:<index>` node for
//! each output of `np.diag(matrix, k=...)`. Each emitted node is a scalar
//! identity map from exactly one source operand. This module validates the
//! static shape, offset, mode, source index, finite value, and cotangent before
//! replaying that identity value/VJP in Rust.

use crate::program_ad_ir::reserve_replay_buffer;
use crate::program_ad_lifecycle::replay_checkpoint;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiagMode {
    Construct,
    Extract,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiagMetadata {
    shape: [usize; 2],
    rank: usize,
    offset: i64,
    mode: DiagMode,
    source_index: usize,
}

/// Return whether an operation label belongs to bounded `np.diag` replay.
pub(crate) fn is_diag_operation(operation: &str) -> bool {
    operation.starts_with("linalg:diag:")
}

/// Evaluate one scalar output from a compact static `diag` Program AD node.
pub(crate) fn diag_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let [value] = input_values else {
        return Err(format!(
            "effect {effect_index} diag replay requires exactly one source operand"
        ));
    };
    if !value.is_finite() {
        return Err(format!("effect {effect_index} diag input must be finite"));
    }
    parse_diag(effect_index, operation)?;
    Ok(*value)
}

/// Return local reverse contributions for one scalar `diag` output node.
pub(crate) fn diag_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} diag cotangent must be finite"
        ));
    }
    diag_output_value(effect_index, operation, input_values)?;
    let mut contribution = reserve_replay_buffer(1)?;
    contribution.push(output_cotangent);
    replay_checkpoint()?;
    Ok(contribution)
}

/// Declare the actual scalar source and reverse contribution for compact diag replay.
pub(crate) fn diag_workspace_bytes(
    effect_index: usize,
    operation: &str,
    input_count: usize,
    requires_adjoint: bool,
) -> Result<usize, String> {
    if input_count != 1 {
        return Err(format!("effect {effect_index} diag replay requires exactly one source operand"));
    }
    parse_diag(effect_index, operation)?;
    let copies = if requires_adjoint { 2 } else { 1 };
    input_count.checked_mul(copies)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "diag workspace exceeds native addressable memory".to_owned())
}

fn parse_diag(effect_index: usize, operation: &str) -> Result<DiagMetadata, String> {
    replay_checkpoint()?;
    for _ in operation.as_bytes().chunks(256) { replay_checkpoint()?; }
    let mut fields = operation.split(':');
    let mut parts = [""; 7];
    for part in &mut parts {
        *part = fields.next().ok_or_else(|| format!("effect {effect_index} diag operation metadata is malformed"))?;
    }
    if fields.next().is_some() || parts[0] != "linalg" || parts[1] != "diag" || parts[3] != "offset" {
        return Err(format!(
            "effect {effect_index} diag operation metadata is malformed"
        ));
    }
    let (shape, rank) = parse_shape_label(effect_index, parts[2])?;
    let offset = parts[4]
        .parse::<i64>()
        .map_err(|_| format!("effect {effect_index} diag offset metadata is malformed"))?;
    let mode = match parts[5] {
        "construct" => DiagMode::Construct,
        "extract" => DiagMode::Extract,
        _ => {
            return Err(format!(
                "effect {effect_index} diag mode metadata is malformed"
            ));
        }
    };
    let source_index = parts[6]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} diag source index metadata is malformed"))?;
    let metadata = DiagMetadata {
        shape,
        rank,
        offset,
        mode,
        source_index,
    };
    validate_diag_metadata(effect_index, &metadata)?;
    Ok(metadata)
}

fn parse_shape_label(effect_index: usize, label: &str) -> Result<([usize; 2], usize), String> {
    let mut shape = [0usize; 2];
    let mut rank = 0usize;
    for part in label.split('x') {
        replay_checkpoint()?;
        let slot = shape.get_mut(rank).ok_or_else(|| {
            format!("effect {effect_index} diag source shape must have rank 1 or 2")
        })?;
        *slot = part.parse::<usize>()
            .map_err(|_| format!("effect {effect_index} diag shape metadata is malformed"))?;
        if *slot == 0 {
            return Err(format!("effect {effect_index} diag dimensions must be positive"));
        }
        rank += 1;
    }
    let mut size = 1usize;
    for dimension in &shape[..rank] {
        size = size.checked_mul(*dimension)
            .filter(|entries| *entries <= isize::MAX as usize / std::mem::size_of::<f64>())
            .ok_or_else(|| format!("effect {effect_index} diag source bytes exceed native addressability"))?;
    }
    Ok((shape, rank))
}

fn validate_diag_metadata(effect_index: usize, metadata: &DiagMetadata) -> Result<(), String> {
    match metadata.mode {
        DiagMode::Construct => validate_construct_metadata(effect_index, metadata),
        DiagMode::Extract => validate_extract_metadata(effect_index, metadata),
    }
}

fn validate_construct_metadata(effect_index: usize, metadata: &DiagMetadata) -> Result<(), String> {
    if metadata.rank != 1 || metadata.source_index >= metadata.shape[0] {
        return Err(format!(
            "effect {effect_index} diag construct source index is outside vector shape"
        ));
    }
    let output_size = metadata
        .offset
        .checked_abs()
        .and_then(|offset| usize::try_from(offset).ok())
        .and_then(|offset| metadata.shape[0].checked_add(offset))
        .ok_or_else(|| format!("effect {effect_index} diag construct shape overflows"))?;
    output_size.checked_mul(output_size)
        .filter(|entries| *entries <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| format!("effect {effect_index} diag construct output bytes exceed native addressability"))?;
    replay_checkpoint()?;
    Ok(())
}

fn validate_extract_metadata(effect_index: usize, metadata: &DiagMetadata) -> Result<(), String> {
    if metadata.rank != 2 {
        return Err(format!(
            "effect {effect_index} diag extract requires rank-2 source metadata"
        ));
    }
    let diagonal_length =
        selected_diagonal_length(metadata.shape[0], metadata.shape[1], metadata.offset);
    if metadata.source_index >= diagonal_length {
        return Err(format!(
            "effect {effect_index} diag extract source index is outside diagonal shape"
        ));
    }
    Ok(())
}

fn selected_diagonal_length(rows: usize, cols: usize, offset: i64) -> usize {
    let Ok(shift) = usize::try_from(offset.unsigned_abs()) else { return 0; };
    if offset >= 0 {
        rows.min(cols.saturating_sub(shift))
    } else {
        rows.saturating_sub(shift).min(cols)
    }
}
