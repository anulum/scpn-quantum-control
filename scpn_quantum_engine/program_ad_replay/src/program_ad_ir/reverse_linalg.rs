// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD reverse linalg accumulation

fn accumulate_multi_dot(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = numeric_scalar_operands(effect, values)?;
    let contributions =
        multi_dot_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions.iter()).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_matrix_power(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = numeric_scalar_operands(effect, values)?;
    let contributions =
        matrix_power_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions.iter()).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_stencil(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = numeric_scalar_operands(effect, values)?;
    let contributions =
        stencil_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions.iter()).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_eigvalsh(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = numeric_scalar_operands(effect, values)?;
    let contributions =
        eigvalsh_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions.iter()).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_eigvals(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = numeric_scalar_operands(effect, values)?;
    let contributions =
        eigvals_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions.iter()).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_eig(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = numeric_scalar_operands(effect, values)?;
    let contributions =
        eig_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions.iter()).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_eigh(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = numeric_scalar_operands(effect, values)?;
    let contributions =
        eigh_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions.iter()).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_svdvals(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = numeric_scalar_operands(effect, values)?;
    let contributions =
        svdvals_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions.iter()).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_diagflat(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = numeric_scalar_operands(effect, values)?;
    let contributions =
        diagflat_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions.iter()).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_diag(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = numeric_scalar_operands(effect, values)?;
    let contributions =
        diag_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions.iter()).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_pinv(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let cotangent_scalar = cotangent.scalar_value()?;
    let input_values = numeric_scalar_operands(effect, values)?;
    let contributions =
        pinv_output_cotangent(effect.index, operation, &input_values, cotangent_scalar)?;
    for (index, (input, contribution)) in effect.inputs.iter().zip(contributions.iter()).enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        add_scalar_adjoint(input, *contribution, values, adjoints)?;
    }
    Ok(())
}
