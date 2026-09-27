// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Gaussian/LU/solve numeric workspace admission

fn general_linalg_workspace_bytes(
    effect: &ProgramADEffect,
    operation: &str,
    requires_adjoint: bool,
) -> Result<usize, String> {
    for _ in operation.as_bytes().chunks(256) {
        crate::program_ad_lifecycle::replay_checkpoint()?;
    }
    let entries = if operation.starts_with("linalg:solve:") {
        let output = parse_solve_output(operation).ok_or_else(|| {
            format!("effect {} {operation} has no solution index", effect.index)
        })?;
        let matrix = output.matrix_size()?;
        let rhs = output.rhs_size()?;
        let input = output.input_size()?;
        require_linalg_operand_count(effect, operation, input)?;
        let inverse_copies = if matches!(output.n, 2 | 3) { 1 } else { 3 };
        let inversion = checked_linalg_workspace_entries(&[(input, 1), (matrix, inverse_copies)])?;
        if requires_adjoint {
            let retained_solution = checked_linalg_workspace_entries(&[(input, 1), (matrix, 1), (rhs, 1)])?;
            inversion.max(retained_solution)
        } else { inversion }
    } else {
        let determinant = operation.starts_with("linalg:det:");
        let n = if determinant {
            parse_det_dim(operation).ok_or_else(|| {
                format!("effect {} {operation} has no determinant dimension", effect.index)
            })?
        } else {
            parse_inv_index(operation).ok_or_else(|| {
                format!("effect {} {operation} has no inverse index", effect.index)
            })?.0
        };
        let matrix = shape_size(&[n, n])?;
        if determinant && matches!(operation, "linalg:det:2x2" | "linalg:det:3x3") {
            if effect.inputs.len() != matrix {
                let arity = if n == 2 { "four" } else { "nine" };
                return Err(format!("effect {} {operation} requires {arity} operands", effect.index));
            }
            // Existing small determinants/cofactors use fixed stack arrays only.
            0
        } else {
            require_linalg_operand_count(effect, operation, matrix)?;
            let copies = if determinant && !requires_adjoint {
                2 // Flattened source and LU work copy.
            } else if matches!(n, 2 | 3) {
                2 // Flattened source and copied closed-form inverse.
            } else {
                4 // Source, double-width augmented matrix and inverse output.
            };
            checked_linalg_workspace_entries(&[(matrix, copies)])?
        }
    };
    entries.checked_mul(std::mem::size_of::<f64>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| "Program AD linalg workspace exceeds native addressable memory".to_owned())
}

fn require_linalg_operand_count(
    effect: &ProgramADEffect,
    operation: &str,
    expected: usize,
) -> Result<(), String> {
    if effect.inputs.len() != expected {
        return Err(format!("effect {} {operation} requires {expected} operands", effect.index));
    }
    Ok(())
}

fn checked_linalg_workspace_entries(buffers: &[(usize, usize)]) -> Result<usize, String> {
    let mut entries = 0usize;
    for &(length, copies) in buffers {
        crate::program_ad_lifecycle::replay_checkpoint()?;
        entries = length.checked_mul(copies)
            .and_then(|count| entries.checked_add(count))
            .ok_or_else(|| "Program AD linalg workspace exceeds native addressable memory".to_owned())?;
    }
    Ok(entries)
}
