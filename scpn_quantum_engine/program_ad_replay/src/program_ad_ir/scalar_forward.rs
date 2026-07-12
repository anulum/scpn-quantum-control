// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD scalar forward replay

/// Interpret a scalar opcode-bearing Program AD IR payload in Rust.
pub fn interpret_program_ad_effect_ir_forward(
    serialization: &str,
    inputs: &[f64],
) -> Result<ProgramADRustInterpreterResult, String> {
    let ir = parse_program_ad_effect_ir(serialization)?;
    if ir.effects.is_empty() {
        return Ok(ProgramADRustInterpreterResult::unsupported(
            0,
            0,
            vec!["program AD IR contains no effects".to_owned()],
        ));
    }
    if inputs.iter().any(|value| !value.is_finite()) {
        return Ok(ProgramADRustInterpreterResult::unsupported(
            ir.effects.len(),
            0,
            vec!["Rust Program AD interpreter inputs must be finite".to_owned()],
        ));
    }
    if has_replay_unsafe_alias(&ir) {
        return Ok(ProgramADRustInterpreterResult::unsupported(
            ir.effects.len(),
            0,
            vec![
                "non-view alias-bearing Program AD IR is outside the bounded Rust scalar interpreter"
                    .to_owned(),
            ],
        ));
    }
    if let Err(reason) = validate_executed_branch_metadata(&ir) {
        return Ok(ProgramADRustInterpreterResult::unsupported(
            ir.effects.len(),
            0,
            vec![reason],
        ));
    }

    let mut ordered_effects: Vec<&ProgramADEffect> = ir.effects.iter().collect();
    ordered_effects.sort_by_key(|effect| effect.ordering);
    let expected_parameters = ordered_effects
        .iter()
        .filter(|effect| effect.kind == "parameter")
        .count();
    if expected_parameters != inputs.len() {
        return Ok(ProgramADRustInterpreterResult::unsupported(
            ir.effects.len(),
            0,
            vec![format!(
                "Program AD IR parameter count {expected_parameters} does not match input count {}",
                inputs.len()
            )],
        ));
    }

    let mut values: HashMap<String, f64> = HashMap::new();
    let mut input_index = 0usize;
    let mut supported_effect_count = 0usize;
    let mut blocked_reasons: Vec<String> = Vec::new();
    for effect in ordered_effects {
        let Some(operation) = effect.operation.as_deref() else {
            blocked_reasons.push(format!(
                "effect {} target {} has no opcode-bearing operation metadata",
                effect.index, effect.target
            ));
            break;
        };
        let evaluated = evaluate_effect(effect, operation, inputs, &mut input_index, &values);
        match evaluated {
            Ok(value) => {
                values.insert(effect.target.clone(), value);
                supported_effect_count += 1;
            }
            Err(reason) => {
                blocked_reasons.push(reason);
                break;
            }
        }
    }
    if !blocked_reasons.is_empty() {
        return Ok(ProgramADRustInterpreterResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            blocked_reasons,
        ));
    }
    let Some(final_effect) = ir.effects.iter().max_by_key(|effect| effect.ordering) else {
        return Ok(ProgramADRustInterpreterResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            vec!["Program AD IR has no final effect".to_owned()],
        ));
    };
    if final_effect_is_indexed_multi_output_linalg(final_effect) {
        return Ok(ProgramADRustInterpreterResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            vec![
                "indexed multi-output linalg result (inverse/solve) is outside bounded Rust replay"
                    .to_owned(),
            ],
        ));
    }
    let Some(value) = values.get(&final_effect.target) else {
        return Ok(ProgramADRustInterpreterResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            vec!["final Program AD IR target was not evaluated".to_owned()],
        ));
    };
    if !value.is_finite() {
        return Ok(ProgramADRustInterpreterResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            vec!["Rust Program AD interpreter final value is not finite".to_owned()],
        ));
    }
    Ok(ProgramADRustInterpreterResult::supported(
        *value,
        ir.effects.len(),
    ))
}

/// Replay scalar value and reverse-mode gradients for a bounded opcode-bearing IR subset.
pub fn interpret_program_ad_effect_ir_value_and_gradient(
    serialization: &str,
    inputs: &[f64],
) -> Result<ProgramADRustValueAndGradientResult, String> {
    let ir = parse_program_ad_effect_ir(serialization)?;
    let (ordered_effects, parameter_targets, values, supported_effect_count) =
        match evaluate_program_ad_ir(&ir, inputs) {
            Ok(result) => result,
            Err(result) => return Ok(*result),
        };
    let Some(final_effect) = ordered_effects.last() else {
        return Ok(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            vec!["Program AD IR has no final effect".to_owned()],
        ));
    };
    if final_effect_is_indexed_multi_output_linalg(final_effect) {
        return Ok(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            vec![
                "indexed multi-output linalg result (inverse/solve) is outside bounded Rust replay"
                    .to_owned(),
            ],
        ));
    }
    let Some(final_value) = values.get(&final_effect.target) else {
        return Ok(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            vec!["final Program AD IR target was not evaluated".to_owned()],
        ));
    };
    let final_scalar = match final_value.scalar_value() {
        Ok(value) => value,
        Err(reason) => {
            return Ok(ProgramADRustValueAndGradientResult::unsupported(
                ir.effects.len(),
                supported_effect_count,
                vec![format!(
                    "{reason}; Rust Program AD value+gradient requires a scalar objective"
                )],
            ));
        }
    };
    if !final_scalar.is_finite() {
        return Ok(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            vec!["Rust Program AD value+gradient final value is not finite".to_owned()],
        ));
    }

    let mut adjoints: HashMap<String, ProgramADNumericValue> = HashMap::new();
    adjoints.insert(
        final_effect.target.clone(),
        ProgramADNumericValue::scalar(1.0),
    );
    for effect in ordered_effects.iter().rev() {
        let cotangent = adjoints
            .get(&effect.target)
            .cloned()
            .unwrap_or_else(|| ProgramADNumericValue::scalar(0.0));
        if cotangent.is_all_zero() {
            continue;
        }
        let Some(operation) = effect.operation.as_deref() else {
            return Ok(ProgramADRustValueAndGradientResult::unsupported(
                ir.effects.len(),
                supported_effect_count,
                vec![format!(
                    "effect {} target {} has no opcode-bearing operation metadata",
                    effect.index, effect.target
                )],
            ));
        };
        if operation == "parameter" {
            continue;
        }
        if let Err(reason) =
            accumulate_reverse_effect(effect, operation, cotangent, &values, &mut adjoints)
        {
            return Ok(ProgramADRustValueAndGradientResult::unsupported(
                ir.effects.len(),
                supported_effect_count,
                vec![reason],
            ));
        }
    }

    let gradient = parameter_targets
        .iter()
        .map(|target| {
            adjoints
                .get(&target.source)
                .and_then(|value| value.values.get(target.flat_index))
                .copied()
                .unwrap_or(0.0)
        })
        .collect::<Vec<f64>>();
    let parameter_target_labels = parameter_targets
        .iter()
        .map(|target| target.label.clone())
        .collect::<Vec<String>>();
    if gradient.iter().any(|value| !value.is_finite()) {
        return Ok(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            vec!["Rust Program AD value+gradient produced a non-finite gradient".to_owned()],
        ));
    }
    Ok(ProgramADRustValueAndGradientResult::supported(
        final_scalar,
        gradient,
        parameter_target_labels,
        ir.effects.len(),
    ))
}
