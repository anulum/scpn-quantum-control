// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD numeric replay state

type ProgramADShapeMap<'a> = HashMap<&'a str, &'a [usize]>;

type ProgramADEvaluation<'a> = (
    Vec<&'a ProgramADEffect>,
    Vec<ScalarParameterTarget>,
    HashMap<String, ProgramADNumericValue>,
    usize,
);

#[derive(Debug, PartialEq)]
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
    fn matrix_size(self) -> Result<usize, String> {
        shape_size(&[self.n, self.n])
    }

    fn rhs_size(self) -> Result<usize, String> {
        shape_size(&[self.n, self.rhs_columns])
    }

    fn input_size(self) -> Result<usize, String> {
        self.matrix_size()?
            .checked_add(self.rhs_size()?)
            .filter(|size| *size <= isize::MAX as usize / std::mem::size_of::<f64>())
            .ok_or_else(|| "Program AD solve input bytes exceed native addressability".to_owned())
    }
}

impl ProgramADNumericValue {
    fn try_clone(&self) -> Result<Self, String> {
        let shape = copy_replay_buffer(&self.shape)?;
        let values = copy_replay_buffer(&self.values)?;
        Ok(Self { shape, values })
    }

    fn scalar(value: f64) -> Result<Self, String> {
        let mut values = reserve_replay_buffer(1)?;
        values.push(value);
        Ok(Self {
            shape: Vec::new(),
            values,
        })
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
        for (index, value) in values.iter().enumerate() {
            if index % 256 == 0 {
                crate::program_ad_lifecycle::replay_checkpoint()?;
            }
            if !value.is_finite() {
                return Err("Program AD shaped value entries must be finite".to_owned());
            }
        }
        Ok(Self { shape, values })
    }

    fn filled(shape: &[usize], value: f64) -> Result<Self, String> {
        if !value.is_finite() {
            return Err("Program AD filled value must be finite".to_owned());
        }
        let size = shape_size(shape)?;
        let dimensions = copy_replay_buffer(shape)?;
        let values = filled_replay_buffer(size, value)?;
        Ok(Self {
            shape: dimensions,
            values,
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

    fn is_all_zero(&self) -> Result<bool, String> {
        for (index, value) in self.values.iter().enumerate() {
            if index % 256 == 0 {
                crate::program_ad_lifecycle::replay_checkpoint()?;
            }
            if *value != 0.0 {
                return Ok(false);
            }
        }
        Ok(true)
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
    if let Err(reason) = validate_replay_inputs(
        ir,
        inputs,
        "Rust Program AD value+gradient inputs must be finite",
        "non-view alias-bearing Program AD IR is outside bounded Rust scalar value+gradient replay",
    ) {
        return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            0,
            vec![reason],
        )));
    }

    let ordered_effects = match ordered_replay_effects(ir) {
        Ok(effects) => effects,
        Err(reason) => {
            return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
                ir.effects.len(),
                0,
                vec![reason],
            )));
        }
    };
    let shapes_by_target = match ssa_shapes_by_target(ir) {
        Ok(shapes) => shapes,
        Err(reason) => {
            return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
                ir.effects.len(),
                0,
                vec![reason],
            )));
        }
    };
    let expected_parameters = ordered_effects.iter().try_fold(0usize, |total, effect| {
        crate::program_ad_lifecycle::replay_checkpoint()?;
        if effect.kind != "parameter" {
            return Ok(total);
        }
        let shape = target_shape(effect, &shapes_by_target)?;
        let count = shape_size(&shape)?;
        total
            .checked_add(count)
            .filter(|count| *count <= isize::MAX as usize / std::mem::size_of::<f64>())
            .ok_or_else(|| "Program AD parameter bytes exceed native addressability".to_owned())
    });
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

    let parameter_bytes = expected_parameters.checked_mul(std::mem::size_of::<ScalarParameterTarget>())
        .ok_or_else(|| "Program AD parameter metadata size overflowed".to_owned());
    if let Err(reason) = parameter_bytes.and_then(crate::program_ad_lifecycle::admit_replay_metadata) {
        return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(), 0, vec![reason],
        )));
    }
    if let Err(reason) = admit_replay_table::<(String, ProgramADNumericValue)>(ir.effects.len()) {
        return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(), 0, vec![reason],
        )));
    }
    if let Err(reason) = admit_numeric_replay_memory(&ordered_effects, &shapes_by_target, expected_parameters) {
        return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(), 0, vec![reason],
        )));
    }

    let mut values: HashMap<String, ProgramADNumericValue> = HashMap::new();
    if let Err(error) = values.try_reserve(ir.effects.len()) {
        return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            0,
            vec![format!("Program AD value-map allocation refused: {error}")],
        )));
    }
    let mut parameter_targets = match reserve_replay_buffer(expected_parameters) {
        Ok(targets) => targets,
        Err(reason) => return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(), 0, vec![reason],
        ))),
    };
    let mut input_index = 0usize;
    let mut supported_effect_count = 0usize;
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
        )
        .and_then(|value| {
            if operation == "parameter" {
                append_parameter_targets_for_effect(effect, &value, &mut parameter_targets)?;
            }
            let target = copy_replay_symbol(&effect.target)?;
            values.insert(target, value);
            Ok(())
        });
        match evaluated {
            Ok(()) => {
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

pub(crate) fn reserve_replay_buffer<T>(count: usize) -> Result<Vec<T>, String> {
    crate::program_ad_lifecycle::replay_checkpoint()?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(count)
        .map_err(|error| format!("Program AD replay-buffer allocation refused: {error}"))?;
    Ok(buffer)
}

pub(crate) fn filled_replay_buffer<T: Copy>(count: usize, value: T) -> Result<Vec<T>, String> {
    let mut buffer = reserve_replay_buffer(count)?;
    for index in 0..count {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        buffer.push(value);
    }
    crate::program_ad_lifecycle::replay_checkpoint()?;
    Ok(buffer)
}

fn copy_replay_buffer<T: Copy>(source: &[T]) -> Result<Vec<T>, String> {
    let mut buffer = reserve_replay_buffer(source.len())?;
    for chunk in source.chunks(256) {
        crate::program_ad_lifecycle::replay_checkpoint()?;
        buffer.extend_from_slice(chunk);
    }
    crate::program_ad_lifecycle::replay_checkpoint()?;
    Ok(buffer)
}

fn ordered_replay_effects(ir: &ProgramADEffectIR) -> Result<Vec<&ProgramADEffect>, String> {
    let bytes_per_effect = std::mem::size_of::<(usize, &ProgramADEffect)>()
        .checked_add(std::mem::size_of::<&ProgramADEffect>())
        .ok_or_else(|| "Program AD effect ordering size overflowed".to_owned())?;
    let ordering_bytes = ir.effects.len().checked_mul(bytes_per_effect)
        .ok_or_else(|| "Program AD effect ordering size overflowed".to_owned())?;
    crate::program_ad_lifecycle::admit_replay_metadata(ordering_bytes)?;
    let mut indexed = reserve_replay_buffer(ir.effects.len())?;
    for (index, effect) in ir.effects.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        indexed.push((index, effect));
    }
    // Original row position gives equal ordering keys the same stable replay order.
    crate::program_ad_order_statistic_reduction::checked_order(&mut indexed, |left, right| {
        (left.1.ordering, left.0) < (right.1.ordering, right.0)
    })?;
    let mut ordered = reserve_replay_buffer(ir.effects.len())?;
    for (index, (_, effect)) in indexed.into_iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        ordered.push(effect);
    }
    Ok(ordered)
}
