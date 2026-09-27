// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Elementwise and structural replay workspaces

fn structural_workspace_bytes(
    effect: &ProgramADEffect,
    operation: &str,
    shapes: &ProgramADShapeMap<'_>,
) -> Result<usize, String> {
    let target = shapes.get(effect.target.as_str()).ok_or_else(|| {
        format!("effect {} target {} is missing SSA shape metadata", effect.index, effect.target)
    })?;
    let output = shape_size(target)?;
    let mut input_values = 0usize;
    let mut input_dimensions = 0usize;
    let mut largest_rank = target.len();
    let refusal = || "structural replay workspace exceeds native addressability".to_owned();
    for input in &effect.inputs {
        crate::program_ad_lifecycle::replay_checkpoint()?;
        let shape = metadata_operand_shape(input, shapes)?;
        input_values = input_values.checked_add(shape_size(shape)?).ok_or_else(refusal)?;
        input_dimensions = input_dimensions.checked_add(shape.len()).ok_or_else(refusal)?;
        largest_rank = largest_rank.max(shape.len());
    }
    if largest_rank == 0 {
        return Ok(0);
    }
    if matches!(operation.split(':').next(), Some("sum" | "mean")) {
        if effect.inputs.len() != 1 {
            return Err(format!("effect {} sum/mean requires one input", effect.index));
        }
        let source = metadata_operand_shape(&effect.inputs[0], shapes)?;
        let count = shape_size(source)?;
        // Cloned source, contribution and reduced contribution; copied
        // cotangent plus expected shapes and coordinate vectors.
        return reduction_buffer_bytes(
            &[count, count, count, output],
            &[source.len(), source.len(), source.len(), source.len(), source.len(), target.len()],
            &[],
        );
    }
    // Reverse div/pow dominates elementwise forward: cloned cotangent,
    // two broadcast operands, two contributions and a reduced contribution.
    // Operand copies and split/accumulated contributions each retain an
    // input-sized set. Unary and structural kernels fit within the same bound.
    let floats = [output, output, output, output, output, output, input_values, input_values];
    // Shape vectors accompany those values; broadcast/transpose/axis helpers
    // also keep expected shapes and two simultaneous coordinate vectors.
    let dimensions = [
        target.len(), target.len(), target.len(), target.len(), target.len(), target.len(),
        input_dimensions, input_dimensions, largest_rank, largest_rank, largest_rank, largest_rank,
    ];
    let operands = effect.inputs.len();
    let mut bytes = reduction_buffer_bytes(
        &floats, &dimensions,
        &[
            (operands, std::mem::size_of::<ProgramADNumericValue>()),
            (operands, std::mem::size_of::<ProgramADNumericValue>()),
            (operands, std::mem::size_of::<Vec<f64>>()),
            (operands, std::mem::size_of::<usize>()),
        ],
    )?;
    if operation.starts_with("index_map:") {
        // The map owns tagged Source(usize)/Constant(f64) entries in addition
        // to its copied source, output, cotangent and scatter buffers.
        let map = crate::program_ad_static_source_map::static_source_map_storage_bytes(output)?;
        bytes = bytes.checked_add(map).ok_or_else(refusal)?;
    }
    if bytes > isize::MAX as usize { return Err(refusal()); }
    Ok(bytes)
}

fn uses_structural_workspace(operation: &str) -> bool {
    matches!(operation,
        "add" | "sub" | "mul" | "div" | "pow"
        | "sin" | "cos" | "exp" | "expm1" | "log" | "log1p" | "sqrt"
        | "tan" | "tanh" | "arcsin" | "arccos" | "reciprocal" | "abs"
        | "reshape" | "ravel" | "broadcast_to" | "transpose"
        | "sum" | "mean" | "concatenate" | "stack" | "index_map"
    ) || ["sum:", "mean:", "concatenate:", "stack:", "index_map:"].iter()
        .any(|prefix| operation.starts_with(prefix))
}
