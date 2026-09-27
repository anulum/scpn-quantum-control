// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD static source-map indexing

//! Static source-map indexing replay for bounded Program AD IR.
//!
//! The `index_map:` opcode is a lowered, explicit representation of static
//! indexing and constant assembly. Each output slot is either `sN`, selecting
//! flattened source slot `N`, or `cVALUE`, embedding a finite constant. Reverse
//! replay scatters cotangents only through `sN` entries and ignores constants.

use crate::program_ad_lifecycle::replay_checkpoint;

#[derive(Debug, Clone, PartialEq)]
enum StaticSourceMapEntry {
    Source(usize),
    Constant(f64),
}

/// Checked declared storage for the owned tagged source-map entries.
pub(crate) fn static_source_map_storage_bytes(entries: usize) -> Result<usize, String> {
    entries
        .checked_mul(std::mem::size_of::<StaticSourceMapEntry>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "static source-map storage exceeds native addressability".to_owned())
}

const INDEX_MAP_PREFIX: &str = "index_map:";

/// Materialize a flattened target vector from explicit static source-map slots.
pub(crate) fn apply_static_source_map(
    effect_index: usize,
    operation: &str,
    source_values: &[f64],
    target_size: usize,
) -> Result<Vec<f64>, String> {
    let entries = parse_static_source_map(effect_index, operation, target_size, "target")?;
    replay_checkpoint()?;
    let mut values = Vec::new();
    values.try_reserve_exact(target_size).map_err(|error| {
        format!("effect {effect_index} index_map target allocation refused: {error}")
    })?;
    replay_checkpoint()?;
    for (position, entry) in entries.into_iter().enumerate() {
        if position % 256 == 0 {
            replay_checkpoint()?;
        }
        let value = match entry {
            StaticSourceMapEntry::Source(index) => source_values.get(index).copied().ok_or_else(|| {
                format!(
                    "effect {effect_index} index_map source index {index} is outside source size {}",
                    source_values.len()
                )
            }),
            StaticSourceMapEntry::Constant(value) => Ok(value),
        }?;
        values.push(value);
    }
    replay_checkpoint()?;
    Ok(values)
}

/// Scatter output cotangents back into the flattened source slots of a source map.
pub(crate) fn scatter_static_source_map_cotangent(
    effect_index: usize,
    operation: &str,
    source_size: usize,
    cotangent_values: &[f64],
) -> Result<Vec<f64>, String> {
    let entries =
        parse_static_source_map(effect_index, operation, cotangent_values.len(), "cotangent")?;
    replay_checkpoint()?;
    let mut contribution = Vec::new();
    contribution
        .try_reserve_exact(source_size)
        .map_err(|error| {
            format!("effect {effect_index} index_map cotangent allocation refused: {error}")
        })?;
    for position in 0..source_size {
        if position % 256 == 0 {
            replay_checkpoint()?;
        }
        contribution.push(0.0_f64);
    }
    for (position, (entry, cotangent)) in entries.iter().zip(cotangent_values.iter()).enumerate() {
        if position % 256 == 0 {
            replay_checkpoint()?;
        }
        if let StaticSourceMapEntry::Source(index) = entry {
            let Some(slot) = contribution.get_mut(*index) else {
                return Err(format!(
                    "effect {effect_index} index_map source index {index} is outside source size {source_size}"
                ));
            };
            *slot += cotangent;
        }
    }
    replay_checkpoint()?;
    Ok(contribution)
}

fn parse_static_source_map(
    effect_index: usize,
    operation: &str,
    expected_entries: usize,
    size_kind: &str,
) -> Result<Vec<StaticSourceMapEntry>, String> {
    replay_checkpoint()?;
    let Some(raw_map) = operation.strip_prefix(INDEX_MAP_PREFIX) else {
        return Err(format!(
            "effect {effect_index} index_map requires static source-map metadata index_map:<sN|cVALUE,...>"
        ));
    };
    if raw_map.is_empty() {
        return Err(format!(
            "effect {effect_index} index_map static source-map metadata must not be empty"
        ));
    }
    let mut entry_count = 1usize;
    for chunk in raw_map.as_bytes().chunks(256) {
        replay_checkpoint()?;
        entry_count = entry_count
            .checked_add(chunk.iter().filter(|byte| **byte == b',').count())
            .ok_or_else(|| format!("effect {effect_index} index_map entry count overflows"))?;
    }
    if entry_count != expected_entries {
        return Err(format!(
            "effect {effect_index} index_map {size_kind} size must be {expected_entries}, got {entry_count} source-map entries"
        ));
    }
    replay_checkpoint()?;
    let mut entries = Vec::new();
    entries.try_reserve_exact(entry_count).map_err(|error| {
        format!("effect {effect_index} index_map metadata allocation refused: {error}")
    })?;
    for (position, token) in raw_map.split(',').enumerate() {
        if position % 256 == 0 {
            replay_checkpoint()?;
        }
        entries.push(parse_static_source_map_token(effect_index, token)?);
    }
    replay_checkpoint()?;
    Ok(entries)
}

fn parse_static_source_map_token(
    effect_index: usize,
    token: &str,
) -> Result<StaticSourceMapEntry, String> {
    replay_checkpoint()?;
    if let Some(raw_index) = token.strip_prefix('s') {
        if raw_index.is_empty() {
            return Err(format!(
                "effect {effect_index} index_map source token must include a flattened source index"
            ));
        }
        let index = raw_index.parse::<usize>().map_err(|_| {
            format!("effect {effect_index} index_map source token {token:?} is not a usize")
        })?;
        return Ok(StaticSourceMapEntry::Source(index));
    }
    if let Some(raw_value) = token.strip_prefix('c') {
        if raw_value.is_empty() {
            return Err(format!(
                "effect {effect_index} index_map constant token must include a finite value"
            ));
        }
        let value = raw_value.parse::<f64>().map_err(|_| {
            format!("effect {effect_index} index_map constant token {token:?} is not finite f64")
        })?;
        if !value.is_finite() {
            return Err(format!(
                "effect {effect_index} index_map constant token {token:?} must be finite"
            ));
        }
        return Ok(StaticSourceMapEntry::Constant(value));
    }
    Err(format!(
        "effect {effect_index} index_map token {token:?} must start with 's' or 'c'"
    ))
}
