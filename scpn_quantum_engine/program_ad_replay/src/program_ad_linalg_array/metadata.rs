// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD multi-dot metadata

fn parse_multi_dot_metadata(
    effect_index: usize,
    operation: &str,
    input_count: usize,
) -> Result<MultiDotMetadata, String> {
    replay_checkpoint()?;
    for _ in operation.as_bytes().chunks(256) { replay_checkpoint()?; }
    let mut parts = [""; 6];
    let mut count = 0usize;
    for field in operation.split(':') {
        let slot = parts.get_mut(count).ok_or_else(|| format!("effect {effect_index} multi_dot operation metadata is malformed"))?;
        *slot = field;
        count += 1;
    }
    if count != 5 && count != 6 {
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
    let mut expected_inputs = 0usize;
    for shape in &operand_shapes {
        replay_checkpoint()?;
        expected_inputs = expected_inputs.checked_add(shape_size(shape)?)
            .filter(|size| *size <= isize::MAX as usize / std::mem::size_of::<f64>())
            .ok_or_else(|| "multi_dot input size overflowed".to_owned())?;
    }
    if input_count != expected_inputs {
        return Err(format!(
            "effect {effect_index} multi_dot input count must match flattened operand shapes \
             (expected {expected_inputs}, got {input_count})"
        ));
    }
    let (output_shape, output_index) = parse_output_metadata(effect_index, &parts[4..count])?;
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
    let mut count = 0usize;
    for _ in shape_signature.split("__") {
        replay_checkpoint()?;
        count = count.checked_add(1).ok_or_else(|| "multi_dot operand count overflows".to_owned())?;
    }
    if count < 2 {
        return Err(format!("effect {effect_index} multi_dot requires at least two operand shapes"));
    }
    let mut shapes = reserve_replay_buffer(count)?;
    for label in shape_signature.split("__") {
        replay_checkpoint()?;
        shapes.push(parse_shape_label(effect_index, label)?);
    }
    Ok(shapes)
}

fn parse_shape_label(effect_index: usize, label: &str) -> Result<Vec<usize>, String> {
    let mut dimensions = [0usize; 2];
    let mut rank = 0usize;
    for part in label.split('x') {
        replay_checkpoint()?;
        let slot = dimensions.get_mut(rank).ok_or_else(|| {
            format!("effect {effect_index} multi_dot supports rank-1 and rank-2 operands")
        })?;
        *slot = part.parse::<usize>().map_err(|_| format!("effect {effect_index} multi_dot shape metadata is malformed"))?;
        if *slot == 0 { return Err(format!("effect {effect_index} multi_dot dimensions must be positive")); }
        rank += 1;
    }
    shape_size(&dimensions[..rank])?;
    copy_chain_buffer(&dimensions[..rank])
}

fn validate_operand_shapes(effect_index: usize, shapes: &[Vec<usize>]) -> Result<(), String> {
    if shapes.len() < 2 {
        return Err(format!(
            "effect {effect_index} multi_dot requires at least two operands"
        ));
    }
    for (index, shape) in shapes.iter().enumerate() {
        replay_checkpoint()?;
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
    let mut result_shape = copy_chain_buffer(&operand_shapes[0])?;
    for next_shape in &operand_shapes[1..] {
        replay_checkpoint()?;
        result_shape = multi_dot_step_shape(effect_index, &result_shape, next_shape)?;
        shape_size(&result_shape)?;
    }
    Ok(result_shape)
}

fn multi_dot_step_shape(
    effect_index: usize,
    left_shape: &[usize],
    next_shape: &[usize],
) -> Result<Vec<usize>, String> {
    let result_shape = match (left_shape.len(), next_shape.len()) {
        (1, 1) => {
            if left_shape[0] != next_shape[0] {
                return Err(format!(
                    "effect {effect_index} multi_dot dimensions must align"
                ));
            }
            Vec::new()
        }
        (1, 2) => {
            if left_shape[0] != next_shape[0] {
                return Err(format!(
                    "effect {effect_index} multi_dot dimensions must align"
                ));
            }
            copy_chain_buffer(&[next_shape[1]])?
        }
        (2, 1) => {
            if left_shape[1] != next_shape[0] {
                return Err(format!(
                    "effect {effect_index} multi_dot dimensions must align"
                ));
            }
            copy_chain_buffer(&[left_shape[0]])?
        }
        (2, 2) => {
            if left_shape[1] != next_shape[0] {
                return Err(format!(
                    "effect {effect_index} multi_dot dimensions must align"
                ));
            }
            copy_chain_buffer(&[left_shape[0], next_shape[1]])?
        }
        _ => {
            return Err(format!(
                "effect {effect_index} multi_dot encountered a scalar intermediate"
            ));
        }
    };
    shape_size(&result_shape)?;
    Ok(result_shape)
}
