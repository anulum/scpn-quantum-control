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

#[derive(Debug, Clone, PartialEq)]
enum StaticSourceMapEntry {
    Source(usize),
    Constant(f64),
}

const INDEX_MAP_PREFIX: &str = "index_map:";

/// Materialize a flattened target vector from explicit static source-map slots.
pub(crate) fn apply_static_source_map(
    effect_index: usize,
    operation: &str,
    source_values: &[f64],
    target_size: usize,
) -> Result<Vec<f64>, String> {
    let entries = parse_static_source_map(effect_index, operation)?;
    if entries.len() != target_size {
        return Err(format!(
            "effect {effect_index} index_map target size must be {}, got {} source-map entries",
            target_size,
            entries.len()
        ));
    }
    entries
        .into_iter()
        .map(|entry| match entry {
            StaticSourceMapEntry::Source(index) => source_values.get(index).copied().ok_or_else(|| {
                format!(
                    "effect {effect_index} index_map source index {index} is outside source size {}",
                    source_values.len()
                )
            }),
            StaticSourceMapEntry::Constant(value) => Ok(value),
        })
        .collect()
}

/// Scatter output cotangents back into the flattened source slots of a source map.
pub(crate) fn scatter_static_source_map_cotangent(
    effect_index: usize,
    operation: &str,
    source_size: usize,
    cotangent_values: &[f64],
) -> Result<Vec<f64>, String> {
    let entries = parse_static_source_map(effect_index, operation)?;
    if entries.len() != cotangent_values.len() {
        return Err(format!(
            "effect {effect_index} index_map cotangent size must be {}, got {} source-map entries",
            cotangent_values.len(),
            entries.len()
        ));
    }
    let mut contribution = vec![0.0_f64; source_size];
    for (entry, cotangent) in entries.iter().zip(cotangent_values.iter()) {
        if let StaticSourceMapEntry::Source(index) = entry {
            let Some(slot) = contribution.get_mut(*index) else {
                return Err(format!(
                    "effect {effect_index} index_map source index {index} is outside source size {source_size}"
                ));
            };
            *slot += cotangent;
        }
    }
    Ok(contribution)
}

fn parse_static_source_map(
    effect_index: usize,
    operation: &str,
) -> Result<Vec<StaticSourceMapEntry>, String> {
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
    raw_map
        .split(',')
        .map(|token| parse_static_source_map_token(effect_index, token))
        .collect()
}

fn parse_static_source_map_token(
    effect_index: usize,
    token: &str,
) -> Result<StaticSourceMapEntry, String> {
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
