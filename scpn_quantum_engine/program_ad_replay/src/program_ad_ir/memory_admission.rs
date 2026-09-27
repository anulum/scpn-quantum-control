// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Declared retained numeric replay admission

fn retained_replay_bytes(count: usize) -> Result<usize, String> {
    count.checked_mul(std::mem::size_of::<f64>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "Program AD retained replay bytes exceed native addressability".to_owned())
}

fn admit_scalar_replay_memory(effects: &[&ProgramADEffect], intermediate_bytes: usize) -> Result<(), String> {
    crate::program_ad_lifecycle::admit_replay_memory(
        crate::program_ad_lifecycle::ReplayMemoryRequest {
            forward_bytes: retained_replay_bytes(effects.len())?,
            adjoint_bytes: 0,
            intermediate_bytes,
        },
    )
}

fn admit_numeric_replay_memory(
    effects: &[&ProgramADEffect],
    shapes: &ProgramADShapeMap<'_>,
    parameter_count: usize,
) -> Result<(), String> {
    let mut forward_count = 0usize;
    for effect in effects {
        crate::program_ad_lifecycle::replay_checkpoint()?;
        let shape = shapes.get(effect.target.as_str()).ok_or_else(|| {
            format!("effect {} target {} is missing SSA shape metadata", effect.index, effect.target)
        })?;
        validate_retained_target_shape(effect, shape, shapes)?;
        forward_count = forward_count.checked_add(shape_size(shape)?)
            .ok_or_else(|| "Program AD retained replay bytes exceed native addressability".to_owned())?;
    }
    let forward_bytes = retained_replay_bytes(forward_count)?;
    let adjoint_count = forward_count.checked_add(parameter_count)
        .ok_or_else(|| "Program AD retained replay bytes exceed native addressability".to_owned())?;
    crate::program_ad_lifecycle::admit_replay_memory(
        crate::program_ad_lifecycle::ReplayMemoryRequest {
            forward_bytes,
            adjoint_bytes: retained_replay_bytes(adjoint_count)?,
            intermediate_bytes: replay_kernel_workspace_bytes(effects, true, Some(shapes))?,
        },
    )
}

fn metadata_operand_shape<'a>(
    name: &str,
    shapes: &ProgramADShapeMap<'a>,
) -> Result<&'a [usize], String> {
    if let Some(shape) = shapes.get(name) {
        return Ok(*shape);
    }
    name.parse::<f64>().map_err(|_|
        format!("operand {name} is neither an SSA value nor a scalar literal"))?;
    Ok(&[])
}

fn validate_elementwise_arity(effect: &ProgramADEffect, operation: &str) -> Result<(), String> {
    let expected = match operation {
        "add" | "sub" | "mul" | "div" | "pow" => 2,
        "sin" | "cos" | "exp" | "expm1" | "log" | "log1p" | "sqrt"
        | "tan" | "tanh" | "arcsin" | "arccos" | "reciprocal" | "abs" => 1,
        _ => return Ok(()),
    };
    if effect.inputs.len() != expected {
        let reason = if expected == 2 { "requires two inputs" } else { "requires one input" };
        return Err(format!("effect {} {reason}", effect.index));
    }
    Ok(())
}

fn validate_retained_target_shape(
    effect: &ProgramADEffect,
    target: &[usize],
    shapes: &ProgramADShapeMap<'_>,
) -> Result<(), String> {
    let Some(operation) = effect.operation.as_deref() else { return Ok(()); };
    validate_elementwise_arity(effect, operation)?;
    let binary = matches!(operation, "add" | "sub" | "mul" | "div" | "pow");
    let unary = matches!(operation, "sin" | "cos" | "exp" | "expm1" | "log" | "log1p"
        | "sqrt" | "tan" | "tanh" | "arcsin" | "arccos" | "reciprocal" | "abs");
    if binary && effect.inputs.len() == 2 {
        let lhs = metadata_operand_shape(&effect.inputs[0], shapes)?;
        let rhs = metadata_operand_shape(&effect.inputs[1], shapes)?;
        let rank = lhs.len().max(rhs.len());
        let mut matches = target.len() == rank;
        for axis in 0..rank {
            if axis % 256 == 0 { crate::program_ad_lifecycle::replay_checkpoint()?; }
            let dimension = broadcast_axis_extent(lhs, rhs, rank, axis)?;
            matches &= target.get(axis) == Some(&dimension);
        }
        if !matches {
            return Err(format!("effect {} target shape metadata does not match binary broadcast shape", effect.index));
        }
    } else if unary && effect.inputs.len() == 1
        && metadata_operand_shape(&effect.inputs[0], shapes)? != target
    {
        return Err(format!("effect {} target shape metadata does not match unary source shape", effect.index));
    }
    Ok(())
}

fn replay_kernel_workspace_bytes(
    effects: &[&ProgramADEffect],
    requires_adjoint: bool,
    shapes: Option<&ProgramADShapeMap<'_>>,
) -> Result<usize, String> {
    let mut peak = 0usize;
    for effect in effects {
        crate::program_ad_lifecycle::replay_checkpoint()?;
        let Some(operation) = effect.operation.as_deref() else { continue; };
        validate_elementwise_arity(effect, operation)?;
        let bytes = if is_multi_dot_operation(operation) {
            crate::program_ad_linalg_array::multi_dot_workspace_bytes(
                effect.index, operation, effect.inputs.len(), requires_adjoint,
            )?
        } else if is_matrix_power_operation(operation) {
            crate::program_ad_linalg_matrix_power::matrix_power_workspace_bytes(
                effect.index, operation, effect.inputs.len(), requires_adjoint,
            )?
        } else if is_svdvals_operation(operation) {
            crate::program_ad_linalg_svd::svdvals_workspace_bytes(
                effect.index, operation, effect.inputs.len(), requires_adjoint,
            )?
        } else if is_pinv_operation(operation) {
            crate::program_ad_linalg_pinv::pinv_workspace_bytes(
                effect.index, operation, effect.inputs.len(), requires_adjoint,
            )?
        } else if operation.starts_with("linalg:det:")
            || operation.starts_with("linalg:inv:")
            || operation.starts_with("linalg:solve:")
        {
            general_linalg_workspace_bytes(effect, operation, requires_adjoint)?
        } else if matches!(operation.split(':').next(), Some("prod" | "var" | "std")) {
            let shapes = shapes.ok_or_else(|| "product/moment reductions require numeric replay".to_owned())?;
            shaped_reduction_workspace_bytes(effect, operation, shapes)?
        } else if is_order_statistic_operation(operation) {
            let shapes = shapes.ok_or_else(|| "order-statistic reductions require numeric replay".to_owned())?;
            order_statistic_admission_bytes(effect, operation, shapes)?
        } else if is_trapezoid_operation(operation) {
            if effect.inputs.len() != 1 {
                return Err(format!("effect {} trapezoid requires one input", effect.index));
            }
            let shapes = shapes.ok_or_else(|| "trapezoid requires ranked numeric replay".to_owned())?;
            let source_shape = metadata_operand_shape(&effect.inputs[0], shapes)?;
            let target_shape = shapes.get(effect.target.as_str()).ok_or_else(|| {
                format!("effect {} target {} is missing SSA shape metadata", effect.index, effect.target)
            })?;
            crate::program_ad_trapezoid_reduction::trapezoid_workspace_bytes(
                effect.index, operation, source_shape, target_shape,
            )?
        } else if is_stencil_operation(operation) {
            crate::program_ad_stencil_reduction::stencil_workspace_bytes(
                effect.index, operation, effect.inputs.len(), requires_adjoint,
            )?
        } else if is_cumulative_operation(operation) {
            crate::program_ad_cumulative_reduction::cumulative_workspace_bytes(
                effect.index, operation, effect.inputs.len(), requires_adjoint,
            )?
        } else if is_signal_operation(operation) {
            crate::program_ad_signal_reduction::signal_workspace_bytes(
                effect.index, operation, effect.inputs.len(), requires_adjoint,
            )?
        } else if is_interpolation_operation(operation) {
            crate::program_ad_interpolation_reduction::interpolation_workspace_bytes(
                effect.index, operation, effect.inputs.len(), requires_adjoint,
            )?
        } else if is_diag_operation(operation) {
            crate::program_ad_linalg_diag::diag_workspace_bytes(
                effect.index, operation, effect.inputs.len(), requires_adjoint,
            )?
        } else if is_diagflat_operation(operation) {
            crate::program_ad_linalg_diagflat::diagflat_workspace_bytes(
                effect.index, operation, effect.inputs.len(), requires_adjoint,
            )?
        } else if is_eigvalsh_operation(operation) || is_eigvals_operation(operation)
            || is_eig_operation(operation) || is_eigh_operation(operation)
        {
            crate::program_ad_linalg_spectral::spectral_workspace_bytes(
                effect.index, operation, effect.inputs.len(), requires_adjoint,
            )?
        } else if let Some(shapes) = shapes.filter(|_| uses_structural_workspace(operation)) {
            structural_workspace_bytes(effect, operation, shapes)?
        } else {
            // Scalar stack operations and unsupported opcodes create no shaped workspace.
            0
        };
        peak = peak.max(bytes);
    }
    Ok(peak)
}

// std's hashbrown table uses power-of-two buckets, a maximum 7/8 load,
// one control byte per bucket and a trailing control group of at most 16
// bytes on the maintained native and wasm targets (hashbrown 0.17.1 raw.rs).
// Twice the requested entries, rounded up with a minimum of 16 buckets,
// conservatively covers that layout; allocator bookkeeping is not measured.
fn admit_replay_table<T>(count: usize) -> Result<(), String> {
    if count == 0 {
        return Ok(());
    }
    let alignment = std::mem::align_of::<T>().max(16);
    let bytes = count
        .checked_mul(2)
        .and_then(|buckets| buckets.max(16).checked_next_power_of_two())
        .and_then(|buckets| {
            buckets.checked_mul(std::mem::size_of::<T>().checked_add(1)?)
        })
        .and_then(|bytes| bytes.checked_add(alignment - 1))
        .and_then(|bytes| bytes.checked_add(16))
        .ok_or_else(|| "Program AD table metadata exceeds native addressability".to_owned())?;
    crate::program_ad_lifecycle::admit_replay_metadata(bytes)
}
