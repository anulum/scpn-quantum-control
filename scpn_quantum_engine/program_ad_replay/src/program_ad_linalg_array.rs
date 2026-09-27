// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD linalg-array replay helpers

//! Static linalg-array replay helpers for Program AD effect IR.
//!
//! Python emits compact `linalg:multi_dot:*` SSA nodes for fixed matrix-chain
//! signatures. This module owns the Rust-side contract parsing, rank/dimension
//! validation, forward value replay, and local VJP replay for those nodes so the
//! main Program AD IR evaluator stays a dispatcher instead of absorbing another
//! linalg kernel family.

use crate::program_ad_ir::{filled_replay_buffer, reserve_replay_buffer};
use crate::program_ad_lifecycle::replay_checkpoint;

#[derive(Debug, Clone, PartialEq, Eq)]
struct MultiDotMetadata {
    operand_shapes: Vec<Vec<usize>>,
    output_index: usize,
    output_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct TensorValue {
    shape: Vec<usize>,
    values: Vec<f64>,
}

/// Return whether an operation label belongs to bounded `np.linalg.multi_dot` replay.
pub(crate) fn is_multi_dot_operation(operation: &str) -> bool {
    operation.starts_with("linalg:multi_dot:")
}

/// Evaluate one scalar output from a compact static `multi_dot` Program AD node.
pub(crate) fn multi_dot_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let metadata = parse_multi_dot_metadata(effect_index, operation, input_values.len())?;
    let output = multi_dot_flat_values(effect_index, &metadata.operand_shapes, input_values)?;
    if output.len() != metadata.output_size {
        return Err(format!(
            "effect {effect_index} multi_dot output metadata does not match evaluated output"
        ));
    }
    Ok(output[metadata.output_index])
}

/// Return local reverse contributions for one scalar `multi_dot` output node.
pub(crate) fn multi_dot_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} multi_dot cotangent must be finite"
        ));
    }
    let metadata = parse_multi_dot_metadata(effect_index, operation, input_values.len())?;
    let output = multi_dot_flat_values(effect_index, &metadata.operand_shapes, input_values)?;
    if output.len() != metadata.output_size {
        return Err(format!(
            "effect {effect_index} multi_dot output metadata does not match evaluated output"
        ));
    }
    let mut output_cotangent_vector = filled_replay_buffer(metadata.output_size, 0.0_f64)?;
    output_cotangent_vector[metadata.output_index] = output_cotangent;
    let mut adjoints = reserve_replay_buffer(input_values.len())?;
    let mut cursor = 0usize;
    for shape in &metadata.operand_shapes {
        let operand_size = shape_size(shape)?;
        let end = cursor.checked_add(operand_size)
            .ok_or_else(|| "multi_dot operand offset overflows".to_owned())?;
        let mut varied_values = copy_chain_buffer(input_values)?;
        let operand = varied_values.get_mut(cursor..end)
            .ok_or_else(|| "multi_dot operand slice is outside inputs".to_owned())?;
        for chunk in operand.chunks_mut(256) {
            replay_checkpoint()?;
            chunk.fill(0.0);
        }
        for element_index in 0..operand_size {
            replay_checkpoint()?;
            varied_values[cursor + element_index] = 1.0;
            let contribution =
                multi_dot_flat_values(effect_index, &metadata.operand_shapes, &varied_values)?;
            let local = dot(&output_cotangent_vector, &contribution)?;
            adjoints.push(local);
            varied_values[cursor + element_index] = 0.0;
        }
        cursor = end;
    }
    Ok(adjoints)
}

include!("program_ad_linalg_array/metadata.rs");
include!("program_ad_linalg_array/workspace.rs");

fn multi_dot_flat_values(
    effect_index: usize,
    operand_shapes: &[Vec<usize>],
    input_values: &[f64],
) -> Result<Vec<f64>, String> {
    validate_chain_values(input_values)?;
    let mut total = None;
    let mut cursor = 0usize;
    for shape in operand_shapes {
        replay_checkpoint()?;
        let size = shape_size(shape)?;
        let end = cursor.checked_add(size).ok_or_else(|| "multi_dot operand offset overflows".to_owned())?;
        let values = input_values.get(cursor..end).ok_or_else(|| {
            format!("effect {effect_index} multi_dot input count must match flattened operand shapes")
        })?;
        let operand = TensorValue::new(copy_chain_buffer(shape)?, copy_chain_buffer(values)?)?;
        total = Some(match total {
            None => operand,
            Some(previous) => multiply_tensors(effect_index, &previous, &operand)?,
        });
        cursor = end;
    }
    if cursor != input_values.len() {
        return Err(format!("effect {effect_index} multi_dot input count must match flattened operand shapes"));
    }
    let total = total.ok_or_else(|| format!("effect {effect_index} multi_dot requires at least two operands"))?;
    validate_chain_values(&total.values)?;
    Ok(total.values)
}

impl TensorValue {
    fn new(shape: Vec<usize>, values: Vec<f64>) -> Result<Self, String> {
        if values.len() != shape_size(&shape)? {
            return Err("multi_dot tensor value size must match shape".to_owned());
        }
        validate_chain_values(&values)?;
        Ok(Self { shape, values })
    }
}

fn multiply_tensors(
    effect_index: usize,
    left: &TensorValue,
    right: &TensorValue,
) -> Result<TensorValue, String> {
    match (left.shape.as_slice(), right.shape.as_slice()) {
        ([left_len], [right_len]) => {
            if left_len != right_len {
                return Err(format!(
                    "effect {effect_index} multi_dot dimensions must align"
                ));
            }
            TensorValue::new(Vec::new(), copy_chain_buffer(&[dot(&left.values, &right.values)?])?)
        }
        ([left_len], [right_rows, right_cols]) => {
            if left_len != right_rows {
                return Err(format!(
                    "effect {effect_index} multi_dot dimensions must align"
                ));
            }
            let mut output = reserve_replay_buffer(*right_cols)?;
            for col in 0..*right_cols {
                replay_checkpoint()?;
                let mut value = 0.0;
                for row in 0..*right_rows {
                    if row % 256 == 0 { replay_checkpoint()?; }
                    value += left.values[row] * right.values[row * right_cols + col];
                }
                output.push(value);
            }
            TensorValue::new(copy_chain_buffer(&[*right_cols])?, output)
        }
        ([left_rows, left_cols], [right_len]) => {
            if left_cols != right_len {
                return Err(format!(
                    "effect {effect_index} multi_dot dimensions must align"
                ));
            }
            let mut output = reserve_replay_buffer(*left_rows)?;
            for row in 0..*left_rows {
                replay_checkpoint()?;
                let mut value = 0.0;
                for col in 0..*left_cols {
                    if col % 256 == 0 { replay_checkpoint()?; }
                    value += left.values[row * left_cols + col] * right.values[col];
                }
                output.push(value);
            }
            TensorValue::new(copy_chain_buffer(&[*left_rows])?, output)
        }
        ([left_rows, left_cols], [right_rows, right_cols]) => {
            if left_cols != right_rows {
                return Err(format!(
                    "effect {effect_index} multi_dot dimensions must align"
                ));
            }
            let mut output = reserve_replay_buffer(shape_size(&[*left_rows, *right_cols])?)?;
            for row in 0..*left_rows {
                replay_checkpoint()?;
                for col in 0..*right_cols {
                    replay_checkpoint()?;
                    let mut value = 0.0;
                    for inner in 0..*left_cols {
                        if inner % 256 == 0 { replay_checkpoint()?; }
                        value += left.values[row * left_cols + inner]
                            * right.values[inner * right_cols + col];
                    }
                    output.push(value);
                }
            }
            TensorValue::new(copy_chain_buffer(&[*left_rows, *right_cols])?, output)
        }
        _ => Err(format!(
            "effect {effect_index} multi_dot encountered a scalar intermediate"
        )),
    }
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, String> {
    if left.len() != right.len() {
        return Err("dot operands must have equal length".to_owned());
    }
    let mut value = -0.0_f64;
    for (index, (lhs, rhs)) in left.iter().zip(right).enumerate() {
        if index % 256 == 0 { replay_checkpoint()?; }
        value += lhs * rhs;
    }
    replay_checkpoint()?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err("dot result must be finite".to_owned())
    }
}

fn shape_size(shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for dimension in shape {
        replay_checkpoint()?;
        size = size
            .checked_mul(*dimension)
            .filter(|size| *size <= isize::MAX as usize / std::mem::size_of::<f64>())
            .ok_or_else(|| "multi_dot shape size overflowed".to_owned())?;
    }
    Ok(size)
}

fn copy_chain_buffer<T: Copy>(source: &[T]) -> Result<Vec<T>, String> {
    let mut values = reserve_replay_buffer(source.len())?;
    for chunk in source.chunks(256) {
        replay_checkpoint()?;
        values.extend_from_slice(chunk);
    }
    replay_checkpoint()?;
    Ok(values)
}

fn validate_chain_values(values: &[f64]) -> Result<(), String> {
    for (index, value) in values.iter().enumerate() {
        if index % 256 == 0 { replay_checkpoint()?; }
        if !value.is_finite() { return Err("multi_dot tensor values must be finite".to_owned()); }
    }
    replay_checkpoint()?;
    Ok(())
}
