// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD numeric structural evaluation

fn reduce_axis_values(
    effect_index: usize,
    operation: &str,
    prefix: &str,
    source: &ProgramADNumericValue,
    target_shape: &[usize],
    scale: f64,
) -> Result<ProgramADNumericValue, String> {
    let axis = parse_static_axis(operation, prefix, source.shape.len())?;
    let expected_shape = axis_reduction_shape(&source.shape, axis);
    if expected_shape != target_shape {
        return Err(format!(
            "effect {effect_index} {prefix} axis reduction target shape must be {:?}, got {:?}",
            expected_shape, target_shape
        ));
    }
    let mut output = vec![0.0_f64; shape_size(target_shape)?];
    for (flat_index, value) in source.values.iter().enumerate() {
        let source_index = unravel_index(flat_index, &source.shape);
        let target_index = index_without_axis(&source_index, axis);
        let target_flat = ravel_index(&target_index, target_shape)?;
        output[target_flat] += value * scale;
    }
    ProgramADNumericValue::new(target_shape.to_vec(), output)
}

fn numeric_reshape(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    let target = target_shape(effect, shapes_by_target)?;
    numeric_reshape_to_target(effect, values, target)
}

fn numeric_ravel(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    let target = target_shape(effect, shapes_by_target)?;
    if target.len() != 1 {
        return Err(format!(
            "effect {} ravel target must be rank-1",
            effect.index
        ));
    }
    numeric_reshape_to_target(effect, values, target)
}

fn numeric_reshape_to_target(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    target_shape: Vec<usize>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} reshape/ravel requires one input",
            effect.index
        ));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    ProgramADNumericValue::new(target_shape, input.values)
}

fn numeric_broadcast_to(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} broadcast_to requires one input",
            effect.index
        ));
    }
    let source = numeric_operand(&effect.inputs[0], values)?;
    let target = target_shape(effect, shapes_by_target)?;
    broadcast_to(&source, &target)
}

fn numeric_transpose(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} transpose requires one input",
            effect.index
        ));
    }
    let source = numeric_operand(&effect.inputs[0], values)?;
    let target = target_shape(effect, shapes_by_target)?;
    transpose_reversed_axes(&source, &target)
}

fn numeric_concatenate(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    let operands = numeric_operands(effect, values)?;
    let target = target_shape(effect, shapes_by_target)?;
    concatenate_values(effect.index, operation, &operands, &target)
}

fn numeric_stack(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    let operands = numeric_operands(effect, values)?;
    let target = target_shape(effect, shapes_by_target)?;
    stack_values(effect.index, operation, &operands, &target)
}

fn numeric_index_map(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} index_map requires one input",
            effect.index
        ));
    }
    let target = target_shape(effect, shapes_by_target)?;
    let source = numeric_operand(&effect.inputs[0], values)?;
    let target_size = shape_size(&target)?;
    let mapped = apply_static_source_map(effect.index, operation, &source.values, target_size)?;
    ProgramADNumericValue::new(target, mapped)
}

fn numeric_binary(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    function: impl Fn(f64, f64) -> Result<f64, String>,
) -> Result<ProgramADNumericValue, String> {
    let (lhs, rhs, shape) = binary_operands(effect, values)?;
    let lhs = broadcast_to(&lhs, &shape)?;
    let rhs = broadcast_to(&rhs, &shape)?;
    elementwise_binary(&lhs, &rhs, function)
}

fn binary_operands(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<(ProgramADNumericValue, ProgramADNumericValue, Vec<usize>), String> {
    if effect.inputs.len() != 2 {
        return Err(format!("effect {} requires two inputs", effect.index));
    }
    let lhs = numeric_operand(&effect.inputs[0], values)?;
    let rhs = numeric_operand(&effect.inputs[1], values)?;
    let shape = broadcast_shape(&lhs.shape, &rhs.shape)?;
    Ok((lhs, rhs, shape))
}

fn scale_value(value: &ProgramADNumericValue, scale: f64) -> Result<ProgramADNumericValue, String> {
    ProgramADNumericValue::new(
        value.shape.clone(),
        value.values.iter().map(|item| item * scale).collect(),
    )
}

fn elementwise_mul(
    left: &ProgramADNumericValue,
    right: &ProgramADNumericValue,
) -> Result<ProgramADNumericValue, String> {
    elementwise_binary(left, right, |lhs, rhs| Ok(lhs * rhs))
}

fn elementwise_binary(
    left: &ProgramADNumericValue,
    right: &ProgramADNumericValue,
    function: impl Fn(f64, f64) -> Result<f64, String>,
) -> Result<ProgramADNumericValue, String> {
    if left.shape != right.shape {
        return Err(format!(
            "Program AD elementwise operands must share shape, got {:?} and {:?}",
            left.shape, right.shape
        ));
    }
    ProgramADNumericValue::new(
        left.shape.clone(),
        left.values
            .iter()
            .zip(right.values.iter())
            .map(|(lhs, rhs)| function(*lhs, *rhs))
            .collect::<Result<Vec<f64>, String>>()?,
    )
}

fn elementwise_binary3(
    first: &ProgramADNumericValue,
    second: &ProgramADNumericValue,
    third: &ProgramADNumericValue,
    function: impl Fn(f64, f64, f64) -> Result<f64, String>,
) -> Result<ProgramADNumericValue, String> {
    if first.shape != second.shape || second.shape != third.shape {
        return Err("Program AD ternary elementwise operands must share shape".to_owned());
    }
    ProgramADNumericValue::new(
        first.shape.clone(),
        first
            .values
            .iter()
            .zip(second.values.iter())
            .zip(third.values.iter())
            .map(|((a, b), c)| function(*a, *b, *c))
            .collect::<Result<Vec<f64>, String>>()?,
    )
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, String> {
    let rank = left.len().max(right.len());
    let mut shape = Vec::with_capacity(rank);
    for axis in 0..rank {
        let left_dim = broadcast_dim(left, rank, axis);
        let right_dim = broadcast_dim(right, rank, axis);
        if left_dim == right_dim || left_dim == 1 || right_dim == 1 {
            shape.push(left_dim.max(right_dim));
        } else {
            return Err(format!(
                "Program AD operands with shapes {left:?} and {right:?} cannot broadcast"
            ));
        }
    }
    Ok(shape)
}

fn broadcast_dim(shape: &[usize], rank: usize, axis: usize) -> usize {
    let offset = rank - shape.len();
    if axis < offset {
        1
    } else {
        shape[axis - offset]
    }
}

fn broadcast_to(
    value: &ProgramADNumericValue,
    shape: &[usize],
) -> Result<ProgramADNumericValue, String> {
    let expected = broadcast_shape(&value.shape, shape)?;
    if expected != shape {
        return Err(format!(
            "Program AD value with shape {:?} cannot broadcast to {:?}",
            value.shape, shape
        ));
    }
    let size = shape_size(shape)?;
    let mut values = Vec::with_capacity(size);
    for flat_index in 0..size {
        let index = unravel_index(flat_index, shape);
        values.push(value.values[broadcast_source_flat_index(&value.shape, &index)?]);
    }
    ProgramADNumericValue::new(shape.to_vec(), values)
}

fn reduce_to_shape(
    value: &ProgramADNumericValue,
    target_shape: &[usize],
) -> Result<ProgramADNumericValue, String> {
    let expected = broadcast_shape(target_shape, &value.shape)?;
    if expected != value.shape {
        return Err(format!(
            "Program AD contribution shape {:?} cannot reduce to {:?}",
            value.shape, target_shape
        ));
    }
    let mut reduced = vec![0.0_f64; shape_size(target_shape)?];
    for (flat_index, item) in value.values.iter().enumerate() {
        let index = unravel_index(flat_index, &value.shape);
        let source_index = broadcast_source_flat_index(target_shape, &index)?;
        reduced[source_index] += item;
    }
    ProgramADNumericValue::new(target_shape.to_vec(), reduced)
}

fn expand_axis_reduction_cotangent(
    effect_index: usize,
    operation: &str,
    prefix: &str,
    input: &ProgramADNumericValue,
    cotangent: &ProgramADNumericValue,
    scale: f64,
) -> Result<ProgramADNumericValue, String> {
    let axis = parse_static_axis(operation, prefix, input.shape.len())?;
    let expected_shape = axis_reduction_shape(&input.shape, axis);
    if cotangent.shape != expected_shape {
        return Err(format!(
            "effect {effect_index} {prefix} axis reduction cotangent shape must be {:?}, got {:?}",
            expected_shape, cotangent.shape
        ));
    }
    let mut contribution = Vec::with_capacity(input.values.len());
    for flat_index in 0..input.values.len() {
        let input_index = unravel_index(flat_index, &input.shape);
        let cotangent_index = index_without_axis(&input_index, axis);
        let cotangent_flat = ravel_index(&cotangent_index, &cotangent.shape)?;
        contribution.push(cotangent.values[cotangent_flat] * scale);
    }
    ProgramADNumericValue::new(input.shape.clone(), contribution)
}

fn transpose_reversed_axes(
    value: &ProgramADNumericValue,
    target_shape: &[usize],
) -> Result<ProgramADNumericValue, String> {
    let expected_shape = value.shape.iter().rev().copied().collect::<Vec<usize>>();
    if expected_shape != target_shape {
        return Err(format!(
            "Program AD transpose from shape {:?} requires target shape {:?}, got {:?}",
            value.shape, expected_shape, target_shape
        ));
    }
    let mut transposed = Vec::with_capacity(shape_size(target_shape)?);
    for flat_index in 0..shape_size(target_shape)? {
        let output_index = unravel_index(flat_index, target_shape);
        let source_index = output_index.iter().rev().copied().collect::<Vec<usize>>();
        transposed.push(value.values[ravel_index(&source_index, &value.shape)?]);
    }
    ProgramADNumericValue::new(target_shape.to_vec(), transposed)
}

fn concatenate_values(
    effect_index: usize,
    operation: &str,
    operands: &[ProgramADNumericValue],
    target_shape: &[usize],
) -> Result<ProgramADNumericValue, String> {
    let (axis, expected_shape, offsets) = concatenate_metadata(effect_index, operation, operands)?;
    if expected_shape != target_shape {
        return Err(format!(
            "effect {effect_index} concatenate target shape must be {:?}, got {:?}",
            expected_shape, target_shape
        ));
    }
    let mut output = Vec::with_capacity(shape_size(target_shape)?);
    for flat_index in 0..shape_size(target_shape)? {
        let output_index = unravel_index(flat_index, target_shape);
        let (operand_index, offset) =
            concatenate_operand_at_axis(output_index[axis], operands, axis, &offsets)?;
        let mut source_index = output_index;
        source_index[axis] -= offset;
        let source_flat = ravel_index(&source_index, &operands[operand_index].shape)?;
        output.push(operands[operand_index].values[source_flat]);
    }
    ProgramADNumericValue::new(target_shape.to_vec(), output)
}

fn split_concatenate_cotangent(
    effect_index: usize,
    operation: &str,
    operands: &[ProgramADNumericValue],
    cotangent: &ProgramADNumericValue,
) -> Result<Vec<ProgramADNumericValue>, String> {
    let (axis, expected_shape, offsets) = concatenate_metadata(effect_index, operation, operands)?;
    if cotangent.shape != expected_shape {
        return Err(format!(
            "effect {effect_index} concatenate cotangent shape must be {:?}, got {:?}",
            expected_shape, cotangent.shape
        ));
    }
    let mut contributions = operands
        .iter()
        .map(|operand| vec![0.0_f64; operand.values.len()])
        .collect::<Vec<Vec<f64>>>();
    for (flat_index, cotangent_value) in cotangent.values.iter().enumerate() {
        let output_index = unravel_index(flat_index, &cotangent.shape);
        let (operand_index, offset) =
            concatenate_operand_at_axis(output_index[axis], operands, axis, &offsets)?;
        let mut source_index = output_index;
        source_index[axis] -= offset;
        let source_flat = ravel_index(&source_index, &operands[operand_index].shape)?;
        contributions[operand_index][source_flat] += cotangent_value;
    }
    operands
        .iter()
        .zip(contributions)
        .map(|(operand, values)| ProgramADNumericValue::new(operand.shape.clone(), values))
        .collect()
}

fn concatenate_metadata(
    effect_index: usize,
    operation: &str,
    operands: &[ProgramADNumericValue],
) -> Result<(usize, Vec<usize>, Vec<usize>), String> {
    if operands.is_empty() {
        return Err(format!("effect {effect_index} concatenate requires inputs"));
    }
    let rank = operands[0].shape.len();
    if rank == 0 {
        return Err(format!(
            "effect {effect_index} concatenate requires ranked array operands"
        ));
    }
    let axis = parse_static_axis(operation, "concatenate", rank)?;
    let mut expected = operands[0].shape.clone();
    expected[axis] = 0;
    let mut offsets = Vec::with_capacity(operands.len());
    let mut axis_total = 0usize;
    for operand in operands {
        if operand.shape.len() != rank {
            return Err(format!(
                "effect {effect_index} concatenate operands must share rank"
            ));
        }
        for (dimension_index, (actual, expected_dimension)) in
            operand.shape.iter().zip(expected.iter()).enumerate()
        {
            if dimension_index != axis && actual != expected_dimension {
                return Err(format!(
                    "effect {effect_index} concatenate non-axis dimensions must match"
                ));
            }
        }
        offsets.push(axis_total);
        axis_total = axis_total
            .checked_add(operand.shape[axis])
            .ok_or_else(|| "Program AD concatenate axis size overflowed".to_owned())?;
    }
    expected[axis] = axis_total;
    Ok((axis, expected, offsets))
}

fn concatenate_operand_at_axis(
    axis_coordinate: usize,
    operands: &[ProgramADNumericValue],
    axis: usize,
    offsets: &[usize],
) -> Result<(usize, usize), String> {
    for (operand_index, (operand, offset)) in operands.iter().zip(offsets.iter()).enumerate() {
        let end = offset + operand.shape[axis];
        if axis_coordinate >= *offset && axis_coordinate < end {
            return Ok((operand_index, *offset));
        }
    }
    Err("Program AD concatenate output coordinate is outside operand ranges".to_owned())
}

fn stack_values(
    effect_index: usize,
    operation: &str,
    operands: &[ProgramADNumericValue],
    target_shape: &[usize],
) -> Result<ProgramADNumericValue, String> {
    let (axis, expected_shape) = stack_metadata(effect_index, operation, operands)?;
    if expected_shape != target_shape {
        return Err(format!(
            "effect {effect_index} stack target shape must be {:?}, got {:?}",
            expected_shape, target_shape
        ));
    }
    let mut output = Vec::with_capacity(shape_size(target_shape)?);
    for flat_index in 0..shape_size(target_shape)? {
        let output_index = unravel_index(flat_index, target_shape);
        let operand_index = output_index[axis];
        let source_index = index_without_axis(&output_index, axis);
        let source_flat = ravel_index(&source_index, &operands[operand_index].shape)?;
        output.push(operands[operand_index].values[source_flat]);
    }
    ProgramADNumericValue::new(target_shape.to_vec(), output)
}

fn split_stack_cotangent(
    effect_index: usize,
    operation: &str,
    operands: &[ProgramADNumericValue],
    cotangent: &ProgramADNumericValue,
) -> Result<Vec<ProgramADNumericValue>, String> {
    let (axis, expected_shape) = stack_metadata(effect_index, operation, operands)?;
    if cotangent.shape != expected_shape {
        return Err(format!(
            "effect {effect_index} stack cotangent shape must be {:?}, got {:?}",
            expected_shape, cotangent.shape
        ));
    }
    let mut contributions = operands
        .iter()
        .map(|operand| vec![0.0_f64; operand.values.len()])
        .collect::<Vec<Vec<f64>>>();
    for (flat_index, cotangent_value) in cotangent.values.iter().enumerate() {
        let output_index = unravel_index(flat_index, &cotangent.shape);
        let operand_index = output_index[axis];
        let source_index = index_without_axis(&output_index, axis);
        let source_flat = ravel_index(&source_index, &operands[operand_index].shape)?;
        contributions[operand_index][source_flat] += cotangent_value;
    }
    operands
        .iter()
        .zip(contributions)
        .map(|(operand, values)| ProgramADNumericValue::new(operand.shape.clone(), values))
        .collect()
}

fn stack_metadata(
    effect_index: usize,
    operation: &str,
    operands: &[ProgramADNumericValue],
) -> Result<(usize, Vec<usize>), String> {
    if operands.is_empty() {
        return Err(format!("effect {effect_index} stack requires inputs"));
    }
    let source_shape = operands[0].shape.clone();
    for operand in operands {
        if operand.shape != source_shape {
            return Err(format!(
                "effect {effect_index} stack operands must have identical shapes"
            ));
        }
    }
    let output_rank = source_shape.len() + 1;
    let axis = parse_static_axis(operation, "stack", output_rank)?;
    let mut expected = source_shape;
    expected.insert(axis, operands.len());
    Ok((axis, expected))
}

fn index_without_axis(index: &[usize], axis: usize) -> Vec<usize> {
    index
        .iter()
        .enumerate()
        .filter_map(|(index_axis, value)| (index_axis != axis).then_some(*value))
        .collect()
}

fn axis_reduction_shape(shape: &[usize], axis: usize) -> Vec<usize> {
    index_without_axis(shape, axis)
}

fn parse_static_axis(operation: &str, prefix: &str, rank: usize) -> Result<usize, String> {
    let expected_prefix = format!("{prefix}:axis:");
    let Some(raw_axis) = operation.strip_prefix(&expected_prefix) else {
        return Err(format!(
            "{prefix} operation requires static axis metadata {prefix}:axis:<int>"
        ));
    };
    let axis = raw_axis
        .parse::<isize>()
        .map_err(|_| format!("{prefix} axis metadata must be an integer"))?;
    normalise_static_axis(axis, rank)
        .map_err(|reason| format!("{prefix} axis metadata is invalid: {reason}"))
}

fn normalise_static_axis(axis: isize, rank: usize) -> Result<usize, String> {
    if rank == 0 {
        return Err("rank must be positive".to_owned());
    }
    let rank_isize =
        isize::try_from(rank).map_err(|_| "rank exceeds axis metadata range".to_owned())?;
    let normalised = if axis < 0 { rank_isize + axis } else { axis };
    if normalised < 0 || normalised >= rank_isize {
        return Err(format!("axis {axis} is outside rank {rank}"));
    }
    usize::try_from(normalised).map_err(|_| "axis normalisation overflowed".to_owned())
}

fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut index = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dimension = shape[axis];
        index[axis] = flat_index % dimension;
        flat_index /= dimension;
    }
    index
}

fn ravel_index(index: &[usize], shape: &[usize]) -> Result<usize, String> {
    if index.len() != shape.len() {
        return Err("Program AD index rank does not match shape rank".to_owned());
    }
    let mut flat = 0usize;
    for (axis_index, dimension) in index.iter().zip(shape.iter()) {
        if axis_index >= dimension {
            return Err("Program AD index is outside shape bounds".to_owned());
        }
        flat = flat
            .checked_mul(*dimension)
            .and_then(|value| value.checked_add(*axis_index))
            .ok_or_else(|| "Program AD flat index overflowed".to_owned())?;
    }
    Ok(flat)
}

fn broadcast_source_flat_index(
    source_shape: &[usize],
    output_index: &[usize],
) -> Result<usize, String> {
    if source_shape.is_empty() {
        return Ok(0);
    }
    if source_shape.len() > output_index.len() {
        return Err("Program AD source rank exceeds output rank".to_owned());
    }
    let offset = output_index.len() - source_shape.len();
    let mut source_index = Vec::with_capacity(source_shape.len());
    for (axis, dimension) in source_shape.iter().enumerate() {
        source_index.push(if *dimension == 1 {
            0
        } else {
            output_index[offset + axis]
        });
    }
    ravel_index(&source_index, source_shape)
}

fn read_3x3_numeric(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<[f64; 9], String> {
    let mut matrix = [0.0_f64; 9];
    for (slot, input) in matrix.iter_mut().zip(effect.inputs.iter()) {
        *slot = operand_scalar_value(input, values)?;
    }
    Ok(matrix)
}
