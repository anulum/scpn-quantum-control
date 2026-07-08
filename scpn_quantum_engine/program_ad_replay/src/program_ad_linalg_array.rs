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
    let mut output_cotangent_vector = vec![0.0; metadata.output_size];
    output_cotangent_vector[metadata.output_index] = output_cotangent;
    let mut adjoints = Vec::with_capacity(input_values.len());
    let mut cursor = 0usize;
    for shape in &metadata.operand_shapes {
        let operand_size = shape_size(shape)?;
        for element_index in 0..operand_size {
            let mut varied_values = input_values.to_vec();
            varied_values[cursor..cursor + operand_size].fill(0.0);
            varied_values[cursor + element_index] = 1.0;
            let contribution =
                multi_dot_flat_values(effect_index, &metadata.operand_shapes, &varied_values)?;
            let local = dot(&output_cotangent_vector, &contribution)?;
            adjoints.push(local);
        }
        cursor += operand_size;
    }
    Ok(adjoints)
}

fn parse_multi_dot_metadata(
    effect_index: usize,
    operation: &str,
    input_count: usize,
) -> Result<MultiDotMetadata, String> {
    let parts = operation.split(':').collect::<Vec<&str>>();
    if parts.len() != 5 && parts.len() != 6 {
        return Err(format!(
            "effect {effect_index} multi_dot operation metadata is malformed"
        ));
    }
    if parts[0] != "linalg" || parts[1] != "multi_dot" || parts[3] != "out" {
        return Err(format!(
            "effect {effect_index} multi_dot operation metadata is malformed"
        ));
    }
    let operand_shapes = parse_operand_shapes(effect_index, parts[2])?;
    validate_operand_shapes(effect_index, &operand_shapes)?;
    let expected_inputs = operand_shapes
        .iter()
        .map(|shape| shape_size(shape))
        .try_fold(0usize, |total, size| {
            total
                .checked_add(size?)
                .ok_or_else(|| "multi_dot input size overflowed".to_owned())
        })?;
    if input_count != expected_inputs {
        return Err(format!(
            "effect {effect_index} multi_dot input count must match flattened operand shapes \
             (expected {expected_inputs}, got {input_count})"
        ));
    }
    let (output_shape, output_index) = parse_output_metadata(effect_index, &parts[4..])?;
    let inferred_output_shape = infer_multi_dot_output_shape(effect_index, &operand_shapes)?;
    if output_shape != inferred_output_shape {
        return Err(format!(
            "effect {effect_index} multi_dot output shape metadata {:?} does not match inferred shape {:?}",
            output_shape, inferred_output_shape
        ));
    }
    let output_size = shape_size(&output_shape)?;
    if output_index >= output_size {
        return Err(format!(
            "effect {effect_index} multi_dot output index is outside result shape"
        ));
    }
    Ok(MultiDotMetadata {
        operand_shapes,
        output_index,
        output_size,
    })
}

fn parse_operand_shapes(
    effect_index: usize,
    shape_signature: &str,
) -> Result<Vec<Vec<usize>>, String> {
    let labels = shape_signature.split("__").collect::<Vec<&str>>();
    if labels.len() < 2 {
        return Err(format!(
            "effect {effect_index} multi_dot requires at least two operand shapes"
        ));
    }
    labels
        .iter()
        .map(|label| parse_shape_label(effect_index, label))
        .collect()
}

fn parse_shape_label(effect_index: usize, label: &str) -> Result<Vec<usize>, String> {
    if label.is_empty() {
        return Err(format!(
            "effect {effect_index} multi_dot shape metadata is malformed"
        ));
    }
    let shape = label
        .split('x')
        .map(|part| {
            if part.is_empty() {
                return Err(format!(
                    "effect {effect_index} multi_dot shape metadata is malformed"
                ));
            }
            part.parse::<usize>()
                .map_err(|_| format!("effect {effect_index} multi_dot shape metadata is malformed"))
        })
        .collect::<Result<Vec<usize>, String>>()?;
    if shape.is_empty() || shape.contains(&0) {
        return Err(format!(
            "effect {effect_index} multi_dot dimensions must be positive"
        ));
    }
    Ok(shape)
}

fn validate_operand_shapes(effect_index: usize, shapes: &[Vec<usize>]) -> Result<(), String> {
    if shapes.len() < 2 {
        return Err(format!(
            "effect {effect_index} multi_dot requires at least two operands"
        ));
    }
    for (index, shape) in shapes.iter().enumerate() {
        if shape.len() != 1 && shape.len() != 2 {
            return Err(format!(
                "effect {effect_index} multi_dot supports rank-1 and rank-2 operands"
            ));
        }
        if 0 < index && index + 1 < shapes.len() && shape.len() != 2 {
            return Err(format!(
                "effect {effect_index} multi_dot middle operands must be rank-2"
            ));
        }
    }
    Ok(())
}

fn parse_output_metadata(
    effect_index: usize,
    output_parts: &[&str],
) -> Result<(Vec<usize>, usize), String> {
    if output_parts.len() == 1 && output_parts[0] == "scalar" {
        return Ok((Vec::new(), 0));
    }
    if output_parts.len() != 2 {
        return Err(format!(
            "effect {effect_index} multi_dot output metadata must be scalar or shape plus index"
        ));
    }
    let shape = parse_shape_label(effect_index, output_parts[0])?;
    let output_index = output_parts[1].parse::<usize>().map_err(|_| {
        format!("effect {effect_index} multi_dot output index metadata is malformed")
    })?;
    Ok((shape, output_index))
}

fn infer_multi_dot_output_shape(
    effect_index: usize,
    operand_shapes: &[Vec<usize>],
) -> Result<Vec<usize>, String> {
    let mut result_shape = operand_shapes[0].clone();
    for next_shape in &operand_shapes[1..] {
        result_shape = match (result_shape.len(), next_shape.len()) {
            (1, 1) => {
                if result_shape[0] != next_shape[0] {
                    return Err(format!(
                        "effect {effect_index} multi_dot dimensions must align"
                    ));
                }
                Vec::new()
            }
            (1, 2) => {
                if result_shape[0] != next_shape[0] {
                    return Err(format!(
                        "effect {effect_index} multi_dot dimensions must align"
                    ));
                }
                vec![next_shape[1]]
            }
            (2, 1) => {
                if result_shape[1] != next_shape[0] {
                    return Err(format!(
                        "effect {effect_index} multi_dot dimensions must align"
                    ));
                }
                vec![result_shape[0]]
            }
            (2, 2) => {
                if result_shape[1] != next_shape[0] {
                    return Err(format!(
                        "effect {effect_index} multi_dot dimensions must align"
                    ));
                }
                vec![result_shape[0], next_shape[1]]
            }
            _ => {
                return Err(format!(
                    "effect {effect_index} multi_dot encountered a scalar intermediate"
                ));
            }
        };
    }
    Ok(result_shape)
}

fn multi_dot_flat_values(
    effect_index: usize,
    operand_shapes: &[Vec<usize>],
    input_values: &[f64],
) -> Result<Vec<f64>, String> {
    let mut operands = split_operands(effect_index, operand_shapes, input_values)?;
    let Some(first) = operands.first().cloned() else {
        return Err(format!(
            "effect {effect_index} multi_dot requires at least two operands"
        ));
    };
    let mut total = first;
    for operand in operands.drain(1..) {
        total = multiply_tensors(effect_index, &total, &operand)?;
    }
    if total.values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} multi_dot output entries must be finite"
        ));
    }
    Ok(total.values)
}

fn split_operands(
    effect_index: usize,
    operand_shapes: &[Vec<usize>],
    input_values: &[f64],
) -> Result<Vec<TensorValue>, String> {
    let mut operands = Vec::with_capacity(operand_shapes.len());
    let mut cursor = 0usize;
    for shape in operand_shapes {
        let size = shape_size(shape)?;
        let Some(values) = input_values.get(cursor..cursor + size) else {
            return Err(format!(
                "effect {effect_index} multi_dot input count must match flattened operand shapes"
            ));
        };
        operands.push(TensorValue::new(shape.clone(), values.to_vec())?);
        cursor += size;
    }
    if cursor != input_values.len() {
        return Err(format!(
            "effect {effect_index} multi_dot input count must match flattened operand shapes"
        ));
    }
    Ok(operands)
}

impl TensorValue {
    fn new(shape: Vec<usize>, values: Vec<f64>) -> Result<Self, String> {
        if values.len() != shape_size(&shape)? {
            return Err("multi_dot tensor value size must match shape".to_owned());
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("multi_dot tensor values must be finite".to_owned());
        }
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
            TensorValue::new(Vec::new(), vec![dot(&left.values, &right.values)?])
        }
        ([left_len], [right_rows, right_cols]) => {
            if left_len != right_rows {
                return Err(format!(
                    "effect {effect_index} multi_dot dimensions must align"
                ));
            }
            let mut output = Vec::with_capacity(*right_cols);
            for col in 0..*right_cols {
                let mut value = 0.0;
                for row in 0..*right_rows {
                    value += left.values[row] * right.values[row * right_cols + col];
                }
                output.push(value);
            }
            TensorValue::new(vec![*right_cols], output)
        }
        ([left_rows, left_cols], [right_len]) => {
            if left_cols != right_len {
                return Err(format!(
                    "effect {effect_index} multi_dot dimensions must align"
                ));
            }
            let mut output = Vec::with_capacity(*left_rows);
            for row in 0..*left_rows {
                let mut value = 0.0;
                for col in 0..*left_cols {
                    value += left.values[row * left_cols + col] * right.values[col];
                }
                output.push(value);
            }
            TensorValue::new(vec![*left_rows], output)
        }
        ([left_rows, left_cols], [right_rows, right_cols]) => {
            if left_cols != right_rows {
                return Err(format!(
                    "effect {effect_index} multi_dot dimensions must align"
                ));
            }
            let mut output = Vec::with_capacity(left_rows * right_cols);
            for row in 0..*left_rows {
                for col in 0..*right_cols {
                    let mut value = 0.0;
                    for inner in 0..*left_cols {
                        value += left.values[row * left_cols + inner]
                            * right.values[inner * right_cols + col];
                    }
                    output.push(value);
                }
            }
            TensorValue::new(vec![*left_rows, *right_cols], output)
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
    let value = left
        .iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err("dot result must be finite".to_owned())
    }
}

fn shape_size(shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for dimension in shape {
        size = size
            .checked_mul(*dimension)
            .ok_or_else(|| "multi_dot shape size overflowed".to_owned())?;
    }
    Ok(size)
}
