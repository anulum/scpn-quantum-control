// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD numeric replay state

type ProgramADEvaluation<'a> = (
    Vec<&'a ProgramADEffect>,
    Vec<ScalarParameterTarget>,
    HashMap<String, ProgramADNumericValue>,
    usize,
);

#[derive(Debug, Clone, PartialEq)]
struct ProgramADNumericValue {
    shape: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ScalarParameterTarget {
    label: String,
    source: String,
    flat_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SolveOutput {
    n: usize,
    rhs_columns: usize,
    row: usize,
    column: usize,
}

impl SolveOutput {
    fn rhs_size(self) -> usize {
        self.n * self.rhs_columns
    }
}

impl ProgramADNumericValue {
    fn scalar(value: f64) -> Self {
        Self {
            shape: Vec::new(),
            values: vec![value],
        }
    }

    fn new(shape: Vec<usize>, values: Vec<f64>) -> Result<Self, String> {
        let expected = shape_size(&shape)?;
        if values.len() != expected {
            return Err(format!(
                "Program AD shaped value {:?} requires {expected} values, got {}",
                shape,
                values.len()
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("Program AD shaped value entries must be finite".to_owned());
        }
        Ok(Self { shape, values })
    }

    fn filled(shape: &[usize], value: f64) -> Result<Self, String> {
        if !value.is_finite() {
            return Err("Program AD filled value must be finite".to_owned());
        }
        Ok(Self {
            shape: shape.to_vec(),
            values: vec![value; shape_size(shape)?],
        })
    }

    fn scalar_value(&self) -> Result<f64, String> {
        if self.shape.is_empty() && self.values.len() == 1 {
            Ok(self.values[0])
        } else {
            Err(format!(
                "Program AD value with shape {:?} is not scalar",
                self.shape
            ))
        }
    }

    fn is_all_zero(&self) -> bool {
        self.values.iter().all(|value| *value == 0.0)
    }
}

fn evaluate_program_ad_ir<'a>(
    ir: &'a ProgramADEffectIR,
    inputs: &[f64],
) -> Result<ProgramADEvaluation<'a>, Box<ProgramADRustValueAndGradientResult>> {
    if ir.effects.is_empty() {
        return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            0,
            0,
            vec!["program AD IR contains no effects".to_owned()],
        )));
    }
    if inputs.iter().any(|value| !value.is_finite()) {
        return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            0,
            vec!["Rust Program AD value+gradient inputs must be finite".to_owned()],
        )));
    }
    if has_replay_unsafe_alias(ir) {
        return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            0,
            vec![
                "non-view alias-bearing Program AD IR is outside bounded Rust scalar value+gradient replay"
                    .to_owned(),
            ],
        )));
    }
    if let Err(reason) = validate_executed_branch_metadata(ir) {
        return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            0,
            vec![reason],
        )));
    }

    let mut ordered_effects: Vec<&ProgramADEffect> = ir.effects.iter().collect();
    ordered_effects.sort_by_key(|effect| effect.ordering);
    let shapes_by_target = ssa_shapes_by_target(ir);
    let expected_parameters = ordered_effects
        .iter()
        .filter(|effect| effect.kind == "parameter")
        .map(|effect| {
            target_shape(effect, &shapes_by_target).and_then(|shape| shape_size(shape.as_slice()))
        })
        .collect::<Result<Vec<usize>, String>>()
        .map(|counts| counts.into_iter().sum::<usize>());
    let expected_parameters = match expected_parameters {
        Ok(count) => count,
        Err(reason) => {
            return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
                ir.effects.len(),
                0,
                vec![reason],
            )));
        }
    };
    if expected_parameters != inputs.len() {
        return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            0,
            vec![format!(
                "Program AD IR flattened parameter count {expected_parameters} does not match input count {}",
                inputs.len()
            )],
        )));
    }

    let mut values: HashMap<String, ProgramADNumericValue> = HashMap::new();
    let mut input_index = 0usize;
    let mut supported_effect_count = 0usize;
    let mut parameter_targets = Vec::new();
    for effect in &ordered_effects {
        let Some(operation) = effect.operation.as_deref() else {
            return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
                ir.effects.len(),
                supported_effect_count,
                vec![format!(
                    "effect {} target {} has no opcode-bearing operation metadata",
                    effect.index, effect.target
                )],
            )));
        };
        let evaluated = evaluate_numeric_effect(
            effect,
            operation,
            inputs,
            &mut input_index,
            &values,
            &shapes_by_target,
        );
        match evaluated {
            Ok(value) => {
                if operation == "parameter" {
                    parameter_targets.extend(parameter_targets_for_effect(effect, &value));
                }
                values.insert(effect.target.clone(), value);
                supported_effect_count += 1;
            }
            Err(reason) => {
                return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
                    ir.effects.len(),
                    supported_effect_count,
                    vec![reason],
                )));
            }
        }
    }
    Ok((
        ordered_effects,
        parameter_targets,
        values,
        supported_effect_count,
    ))
}
