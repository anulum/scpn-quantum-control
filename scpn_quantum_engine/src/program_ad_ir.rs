// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD IR metadata parity

//! Rust metadata parser for Python-emitted `program_ad_effect_ir.v1` payloads.
//!
//! This module mirrors the bounded Python Program AD IR schema so Rust-side
//! tooling can inspect evidence metadata and execute a narrow scalar forward
//! interpreter when opcode-bearing rows are present. It does not promote LLVM
//! lowering, JIT execution, reverse-mode compiler AD, hardware execution, or
//! performance claims.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const PROGRAM_AD_EFFECT_IR_FORMAT: &str = "program_ad_effect_ir.v1";
const PROGRAM_AD_IR_CLAIM_BOUNDARY: &str = "metadata_only_no_program_execution";
const PROGRAM_AD_RUST_INTERPRETER_CLAIM_BOUNDARY: &str =
    "bounded_rust_program_ad_ir_scalar_primitives_executed_branch_no_alias_no_llvm_jit";
const PROGRAM_AD_RUST_VALUE_AND_GRADIENT_CLAIM_BOUNDARY: &str =
    "bounded_rust_program_ad_ir_scalar_primitives_value_and_gradient_executed_branch_no_alias_no_llvm_jit";

/// One SSA value record from Python-emitted Program AD metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProgramADSSAValue {
    pub name: String,
    pub producer: usize,
    pub version: usize,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub effect: usize,
}

/// One ordered effect record from Python-emitted Program AD metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProgramADEffect {
    pub index: usize,
    pub kind: String,
    pub target: String,
    pub inputs: Vec<String>,
    pub version: usize,
    pub ordering: usize,
    #[serde(default)]
    pub operation: Option<String>,
}

/// One alias edge record from Python-emitted Program AD metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProgramADAliasEdge {
    pub source: String,
    pub target: String,
    pub kind: String,
    pub version: usize,
}

/// One control-flow region record from Python-emitted Program AD metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProgramADControlRegion {
    pub index: usize,
    pub kind: String,
    pub predicate: Option<String>,
    pub entered: bool,
    pub source_line: Option<usize>,
}

/// One metadata-only phi record from Python-emitted Program AD metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProgramADPhiNode {
    pub index: usize,
    pub target: String,
    pub incoming: Vec<String>,
    pub control_region: Option<usize>,
    pub selected: Option<String>,
    pub source_line: Option<usize>,
}

/// Parsed Rust view of a `program_ad_effect_ir.v1` payload.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProgramADEffectIR {
    pub format: String,
    pub ssa_values: Vec<ProgramADSSAValue>,
    pub effects: Vec<ProgramADEffect>,
    pub alias_edges: Vec<ProgramADAliasEdge>,
    pub control_regions: Vec<ProgramADControlRegion>,
    #[serde(default)]
    pub phi_nodes: Vec<ProgramADPhiNode>,
    pub bytecode_offsets: Vec<usize>,
}

/// JSON-ready summary for Rust Program AD IR metadata inspection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProgramADEffectIRMetadataSummary {
    pub format: String,
    pub ssa_value_count: usize,
    pub effect_count: usize,
    pub alias_edge_count: usize,
    pub control_region_count: usize,
    pub phi_node_count: usize,
    pub bytecode_offset_count: usize,
    pub claim_boundary: String,
}

/// JSON-ready result for bounded Rust scalar Program AD IR interpretation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProgramADRustInterpreterResult {
    pub supported: bool,
    pub value: Option<f64>,
    pub effect_count: usize,
    pub supported_effect_count: usize,
    pub blocked_reasons: Vec<String>,
    pub claim_boundary: String,
}

/// JSON-ready result for bounded Rust scalar Program AD value and gradient replay.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProgramADRustValueAndGradientResult {
    pub supported: bool,
    pub value: Option<f64>,
    pub gradient: Vec<f64>,
    pub parameter_targets: Vec<String>,
    pub effect_count: usize,
    pub supported_effect_count: usize,
    pub blocked_reasons: Vec<String>,
    pub claim_boundary: String,
}

impl ProgramADEffectIR {
    /// Return a claim-bounded metadata summary without executing Program AD.
    pub fn metadata_summary(&self) -> ProgramADEffectIRMetadataSummary {
        ProgramADEffectIRMetadataSummary {
            format: self.format.clone(),
            ssa_value_count: self.ssa_values.len(),
            effect_count: self.effects.len(),
            alias_edge_count: self.alias_edges.len(),
            control_region_count: self.control_regions.len(),
            phi_node_count: self.phi_nodes.len(),
            bytecode_offset_count: self.bytecode_offsets.len(),
            claim_boundary: PROGRAM_AD_IR_CLAIM_BOUNDARY.to_owned(),
        }
    }
}

impl ProgramADRustInterpreterResult {
    fn unsupported(
        effect_count: usize,
        supported_effect_count: usize,
        blocked_reasons: Vec<String>,
    ) -> Self {
        Self {
            supported: false,
            value: None,
            effect_count,
            supported_effect_count,
            blocked_reasons,
            claim_boundary: PROGRAM_AD_RUST_INTERPRETER_CLAIM_BOUNDARY.to_owned(),
        }
    }

    fn supported(value: f64, effect_count: usize) -> Self {
        Self {
            supported: true,
            value: Some(value),
            effect_count,
            supported_effect_count: effect_count,
            blocked_reasons: Vec::new(),
            claim_boundary: PROGRAM_AD_RUST_INTERPRETER_CLAIM_BOUNDARY.to_owned(),
        }
    }
}

impl ProgramADRustValueAndGradientResult {
    fn unsupported(
        effect_count: usize,
        supported_effect_count: usize,
        blocked_reasons: Vec<String>,
    ) -> Self {
        Self {
            supported: false,
            value: None,
            gradient: Vec::new(),
            parameter_targets: Vec::new(),
            effect_count,
            supported_effect_count,
            blocked_reasons,
            claim_boundary: PROGRAM_AD_RUST_VALUE_AND_GRADIENT_CLAIM_BOUNDARY.to_owned(),
        }
    }

    fn supported(
        value: f64,
        gradient: Vec<f64>,
        parameter_targets: Vec<String>,
        effect_count: usize,
    ) -> Self {
        Self {
            supported: true,
            value: Some(value),
            gradient,
            parameter_targets,
            effect_count,
            supported_effect_count: effect_count,
            blocked_reasons: Vec::new(),
            claim_boundary: PROGRAM_AD_RUST_VALUE_AND_GRADIENT_CLAIM_BOUNDARY.to_owned(),
        }
    }
}

/// Parse Python-emitted `program_ad_effect_ir.v1` metadata and fail closed.
pub fn parse_program_ad_effect_ir(serialization: &str) -> Result<ProgramADEffectIR, String> {
    if serialization.trim().is_empty() {
        return Err("program AD IR serialization must be non-empty".to_owned());
    }
    let payload: Value = serde_json::from_str(serialization)
        .map_err(|error| format!("program AD IR serialization is invalid JSON: {error}"))?;
    validate_program_ad_payload_shape(&payload)?;
    let ir: ProgramADEffectIR = serde_json::from_value(payload)
        .map_err(|error| format!("program AD IR serialization does not match schema: {error}"))?;
    validate_program_ad_effect_ir(&ir)?;
    Ok(ir)
}

fn validate_program_ad_payload_shape(payload: &Value) -> Result<(), String> {
    let Some(object) = payload.as_object() else {
        return Err("program AD IR serialization must decode to an object".to_owned());
    };
    for field in [
        "ssa_values",
        "effects",
        "alias_edges",
        "control_regions",
        "phi_nodes",
        "bytecode_offsets",
    ] {
        let Some(value) = object.get(field) else {
            return Err(format!("program AD IR {field} must be present"));
        };
        if !value.is_array() {
            return Err(format!("program AD IR {field} must be a list"));
        }
    }
    Ok(())
}

fn validate_program_ad_effect_ir(ir: &ProgramADEffectIR) -> Result<(), String> {
    if ir.format != PROGRAM_AD_EFFECT_IR_FORMAT {
        return Err("program AD IR format must be program_ad_effect_ir.v1".to_owned());
    }
    for value in &ir.ssa_values {
        require_non_empty(&value.name, "ssa_values name")?;
        require_non_empty(&value.dtype, "ssa_values dtype")?;
    }
    for effect in &ir.effects {
        require_non_empty(&effect.kind, "effects kind")?;
        require_non_empty(&effect.target, "effects target")?;
        for input in &effect.inputs {
            require_non_empty(input, "effects inputs")?;
        }
        if let Some(operation) = &effect.operation {
            require_non_empty(operation, "effects operation")?;
        }
    }
    for edge in &ir.alias_edges {
        require_non_empty(&edge.source, "alias_edges source")?;
        require_non_empty(&edge.target, "alias_edges target")?;
        require_non_empty(&edge.kind, "alias_edges kind")?;
    }
    for region in &ir.control_regions {
        require_non_empty(&region.kind, "control_regions kind")?;
        if let Some(predicate) = &region.predicate {
            require_non_empty(predicate, "control_regions predicate")?;
        }
        require_positive_optional(region.source_line, "control_regions source_line")?;
    }
    for phi in &ir.phi_nodes {
        require_non_empty(&phi.target, "phi_nodes target")?;
        if phi.incoming.len() < 2 {
            return Err(
                "program AD IR phi_nodes incoming must contain at least two entries".to_owned(),
            );
        }
        for incoming in &phi.incoming {
            require_non_empty(incoming, "phi_nodes incoming")?;
        }
        if let Some(selected) = &phi.selected {
            require_non_empty(selected, "phi_nodes selected")?;
        }
        require_positive_optional(phi.source_line, "phi_nodes source_line")?;
    }
    Ok(())
}

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
    if !ir.alias_edges.is_empty() {
        return Ok(ProgramADRustInterpreterResult::unsupported(
            ir.effects.len(),
            0,
            vec![
                "alias-bearing Program AD IR is outside the bounded Rust scalar interpreter"
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
        match evaluate_scalar_program_ad_ir(&ir, inputs) {
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
    let Some(final_value) = values.get(&final_effect.target) else {
        return Ok(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            vec!["final Program AD IR target was not evaluated".to_owned()],
        ));
    };
    if !final_value.is_finite() {
        return Ok(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            vec!["Rust Program AD value+gradient final value is not finite".to_owned()],
        ));
    }

    let mut adjoints: HashMap<String, f64> = HashMap::new();
    adjoints.insert(final_effect.target.clone(), 1.0);
    for effect in ordered_effects.iter().rev() {
        let cotangent = *adjoints.get(&effect.target).unwrap_or(&0.0);
        if cotangent == 0.0 {
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
        .map(|target| *adjoints.get(target).unwrap_or(&0.0))
        .collect::<Vec<f64>>();
    if gradient.iter().any(|value| !value.is_finite()) {
        return Ok(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            supported_effect_count,
            vec!["Rust Program AD value+gradient produced a non-finite gradient".to_owned()],
        ));
    }
    Ok(ProgramADRustValueAndGradientResult::supported(
        *final_value,
        gradient,
        parameter_targets,
        ir.effects.len(),
    ))
}

type ScalarEvaluation<'a> = (
    Vec<&'a ProgramADEffect>,
    Vec<String>,
    HashMap<String, f64>,
    usize,
);

fn evaluate_scalar_program_ad_ir<'a>(
    ir: &'a ProgramADEffectIR,
    inputs: &[f64],
) -> Result<ScalarEvaluation<'a>, Box<ProgramADRustValueAndGradientResult>> {
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
    if !ir.alias_edges.is_empty() {
        return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            0,
            vec![
                "alias-bearing Program AD IR is outside bounded Rust scalar value+gradient replay"
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
    let expected_parameters = ordered_effects
        .iter()
        .filter(|effect| effect.kind == "parameter")
        .count();
    if expected_parameters != inputs.len() {
        return Err(Box::new(ProgramADRustValueAndGradientResult::unsupported(
            ir.effects.len(),
            0,
            vec![format!(
                "Program AD IR parameter count {expected_parameters} does not match input count {}",
                inputs.len()
            )],
        )));
    }

    let mut values: HashMap<String, f64> = HashMap::new();
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
        let evaluated = evaluate_effect(effect, operation, inputs, &mut input_index, &values);
        match evaluated {
            Ok(value) => {
                if operation == "parameter" {
                    parameter_targets.push(effect.target.clone());
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

fn accumulate_reverse_effect(
    effect: &ProgramADEffect,
    operation: &str,
    cotangent: f64,
    values: &HashMap<String, f64>,
    adjoints: &mut HashMap<String, f64>,
) -> Result<(), String> {
    match operation {
        operation if operation.starts_with("branch:") => Ok(()),
        "add" => accumulate_binary(effect, values, adjoints, cotangent, 1.0, 1.0),
        "sub" => accumulate_binary(effect, values, adjoints, cotangent, 1.0, -1.0),
        "mul" => {
            let lhs = operand_value(&effect.inputs[0], values)?;
            let rhs = operand_value(&effect.inputs[1], values)?;
            accumulate_binary(effect, values, adjoints, cotangent, rhs, lhs)
        }
        "div" => {
            let lhs = operand_value(&effect.inputs[0], values)?;
            let rhs = operand_value(&effect.inputs[1], values)?;
            if rhs == 0.0 {
                return Err("division denominator must be non-zero".to_owned());
            }
            accumulate_binary(
                effect,
                values,
                adjoints,
                cotangent,
                1.0 / rhs,
                -lhs / (rhs * rhs),
            )
        }
        "pow" => {
            let lhs = operand_value(&effect.inputs[0], values)?;
            let rhs = operand_value(&effect.inputs[1], values)?;
            if lhs <= 0.0 {
                return Err("pow gradient requires a positive base".to_owned());
            }
            let value = lhs.powf(rhs);
            accumulate_binary(
                effect,
                values,
                adjoints,
                cotangent,
                rhs * lhs.powf(rhs - 1.0),
                value * lhs.ln(),
            )
        }
        "sin" => accumulate_unary(effect, values, adjoints, cotangent, f64::cos),
        "cos" => accumulate_unary(effect, values, adjoints, cotangent, |value| -value.sin()),
        "exp" => accumulate_unary(effect, values, adjoints, cotangent, f64::exp),
        "expm1" => accumulate_unary(effect, values, adjoints, cotangent, f64::exp),
        "log" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            cotangent,
            |value| value > 0.0,
            |value| 1.0 / value,
            "log input must be positive",
        ),
        "log1p" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            cotangent,
            |value| value > -1.0,
            |value| 1.0 / (1.0 + value),
            "log1p input must be greater than -1",
        ),
        "sqrt" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            cotangent,
            |value| value > 0.0,
            |value| 0.5 / value.sqrt(),
            "sqrt input must be positive",
        ),
        "tan" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            cotangent,
            |value| value.cos().abs() > 1.0e-15,
            |value| 1.0 / (value.cos() * value.cos()),
            "tan input must have non-zero cosine",
        ),
        "tanh" => accumulate_unary(effect, values, adjoints, cotangent, |value| {
            let tanh = value.tanh();
            1.0 - tanh * tanh
        }),
        "arcsin" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            cotangent,
            |value| value.abs() < 1.0,
            |value| 1.0 / (1.0 - value * value).sqrt(),
            "arcsin input must be strictly inside (-1, 1)",
        ),
        "arccos" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            cotangent,
            |value| value.abs() < 1.0,
            |value| -1.0 / (1.0 - value * value).sqrt(),
            "arccos input must be strictly inside (-1, 1)",
        ),
        "reciprocal" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            cotangent,
            |value| value != 0.0,
            |value| -1.0 / (value * value),
            "reciprocal input must be non-zero",
        ),
        "abs" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            cotangent,
            |value| value != 0.0,
            |value| value.signum(),
            "abs gradient is undefined at zero",
        ),
        _ => Err(format!(
            "effect {} operation {operation} is outside bounded Rust scalar value+gradient replay",
            effect.index
        )),
    }
}

fn accumulate_unary(
    effect: &ProgramADEffect,
    values: &HashMap<String, f64>,
    adjoints: &mut HashMap<String, f64>,
    cotangent: f64,
    derivative: impl Fn(f64) -> f64,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let input = &effect.inputs[0];
    let value = operand_value(input, values)?;
    add_adjoint(input, cotangent * derivative(value), values, adjoints)
}

fn accumulate_unary_domain(
    effect: &ProgramADEffect,
    values: &HashMap<String, f64>,
    adjoints: &mut HashMap<String, f64>,
    cotangent: f64,
    predicate: impl Fn(f64) -> bool,
    derivative: impl Fn(f64) -> f64,
    domain_error: &str,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let input = &effect.inputs[0];
    let value = operand_value(input, values)?;
    if !predicate(value) {
        return Err(domain_error.to_owned());
    }
    add_adjoint(input, cotangent * derivative(value), values, adjoints)
}

fn accumulate_binary(
    effect: &ProgramADEffect,
    values: &HashMap<String, f64>,
    adjoints: &mut HashMap<String, f64>,
    cotangent: f64,
    lhs_derivative: f64,
    rhs_derivative: f64,
) -> Result<(), String> {
    if effect.inputs.len() != 2 {
        return Err(format!("effect {} requires two inputs", effect.index));
    }
    add_adjoint(
        &effect.inputs[0],
        cotangent * lhs_derivative,
        values,
        adjoints,
    )?;
    add_adjoint(
        &effect.inputs[1],
        cotangent * rhs_derivative,
        values,
        adjoints,
    )
}

fn add_adjoint(
    input: &str,
    contribution: f64,
    values: &HashMap<String, f64>,
    adjoints: &mut HashMap<String, f64>,
) -> Result<(), String> {
    if !contribution.is_finite() {
        return Err(format!("adjoint contribution for {input} must be finite"));
    }
    if values.contains_key(input) {
        *adjoints.entry(input.to_owned()).or_insert(0.0) += contribution;
    }
    Ok(())
}

fn evaluate_effect(
    effect: &ProgramADEffect,
    operation: &str,
    inputs: &[f64],
    input_index: &mut usize,
    values: &HashMap<String, f64>,
) -> Result<f64, String> {
    if operation == "parameter" {
        if effect.kind != "parameter" {
            return Err(format!(
                "effect {} operation parameter must have kind parameter",
                effect.index
            ));
        }
        let Some(value) = inputs.get(*input_index) else {
            return Err(format!(
                "effect {} parameter input is missing",
                effect.index
            ));
        };
        *input_index += 1;
        return Ok(*value);
    }
    if operation.starts_with("branch:") {
        return evaluate_branch_effect(effect, operation);
    }
    match operation {
        "add" => binary(effect, values, |lhs, rhs| Ok(lhs + rhs)),
        "sub" => binary(effect, values, |lhs, rhs| Ok(lhs - rhs)),
        "mul" => binary(effect, values, |lhs, rhs| Ok(lhs * rhs)),
        "div" => binary(effect, values, |lhs, rhs| {
            if rhs == 0.0 {
                Err("division denominator must be non-zero".to_owned())
            } else {
                Ok(lhs / rhs)
            }
        }),
        "pow" => binary(effect, values, |lhs, rhs| {
            let value = lhs.powf(rhs);
            if value.is_finite() {
                Ok(value)
            } else {
                Err("power result must be finite".to_owned())
            }
        }),
        "sin" => unary(effect, values, f64::sin),
        "cos" => unary(effect, values, f64::cos),
        "exp" => unary_checked(effect, values, f64::exp, "exp result must be finite"),
        "expm1" => unary_checked(effect, values, f64::exp_m1, "expm1 result must be finite"),
        "log" => unary_domain(
            effect,
            values,
            |value| value > 0.0,
            f64::ln,
            "log input must be positive",
        ),
        "log1p" => unary_domain(
            effect,
            values,
            |value| value > -1.0,
            f64::ln_1p,
            "log1p input must be greater than -1",
        ),
        "sqrt" => unary_domain(
            effect,
            values,
            |value| value > 0.0,
            f64::sqrt,
            "sqrt input must be positive",
        ),
        "tan" => unary_domain(
            effect,
            values,
            |value| value.cos().abs() > 1.0e-15,
            f64::tan,
            "tan input must have non-zero cosine",
        ),
        "tanh" => unary(effect, values, f64::tanh),
        "arcsin" => unary_domain(
            effect,
            values,
            |value| value.abs() < 1.0,
            f64::asin,
            "arcsin input must be strictly inside (-1, 1)",
        ),
        "arccos" => unary_domain(
            effect,
            values,
            |value| value.abs() < 1.0,
            f64::acos,
            "arccos input must be strictly inside (-1, 1)",
        ),
        "reciprocal" => unary_domain(
            effect,
            values,
            |value| value != 0.0,
            |value| 1.0 / value,
            "reciprocal input must be non-zero",
        ),
        "abs" => unary(effect, values, f64::abs),
        _ => Err(format!(
            "effect {} operation {operation} is outside the bounded Rust scalar interpreter",
            effect.index
        )),
    }
}

fn validate_executed_branch_metadata(ir: &ProgramADEffectIR) -> Result<(), String> {
    let mut branch_effects_by_operation: HashMap<&str, usize> = HashMap::new();
    for effect in &ir.effects {
        let Some(operation) = effect.operation.as_deref() else {
            continue;
        };
        if !operation.starts_with("branch:") {
            continue;
        }
        if effect.kind != "control_branch" {
            return Err(format!(
                "branch effect {} must have kind control_branch",
                effect.index
            ));
        }
        if !effect.inputs.is_empty() {
            return Err(format!(
                "branch effect {} must not carry differentiable inputs",
                effect.index
            ));
        }
        branch_effects_by_operation.insert(operation, effect.index);
    }

    if ir.control_regions.is_empty() && ir.phi_nodes.is_empty() {
        return Ok(());
    }
    if ir.control_regions.is_empty() || ir.phi_nodes.is_empty() {
        return Err(
            "runtime branch metadata must include both control regions and phi nodes".to_owned(),
        );
    }

    let mut runtime_region_entered_by_index: HashMap<usize, bool> = HashMap::new();
    let mut source_region_indices: HashSet<usize> = HashSet::new();
    for region in &ir.control_regions {
        if region.kind == "source_control_flow" {
            source_region_indices.insert(region.index);
            continue;
        }
        if region.kind != "runtime_branch" {
            return Err(
                "only executed runtime_branch metadata is supported by bounded Rust branch replay"
                    .to_owned(),
            );
        }
        let Some(predicate) = region.predicate.as_deref() else {
            return Err("runtime branch metadata must include a predicate".to_owned());
        };
        if !predicate.starts_with("branch:") {
            return Err("runtime branch predicate must reference a branch operation".to_owned());
        }
        if !branch_effects_by_operation.contains_key(predicate) {
            return Err("runtime branch predicate must match a control_branch effect".to_owned());
        }
        let predicate_entered = branch_operation_value(predicate)?;
        if predicate_entered != region.entered {
            return Err("runtime branch predicate and entered flag disagree".to_owned());
        }
        runtime_region_entered_by_index.insert(region.index, region.entered);
    }

    let mut phi_count_by_region: HashMap<usize, usize> = HashMap::new();
    for phi in &ir.phi_nodes {
        let Some(region_index) = phi.control_region else {
            return Err("runtime branch phi metadata must reference a control region".to_owned());
        };
        if source_region_indices.contains(&region_index) {
            continue;
        }
        let Some(entered) = runtime_region_entered_by_index.get(&region_index) else {
            return Err(
                "runtime branch phi metadata must reference a runtime_branch region".to_owned(),
            );
        };
        let Some(selected) = phi.selected.as_deref() else {
            return Err("runtime branch phi metadata must record selected path".to_owned());
        };
        let expected_selected = if *entered {
            "executed_true"
        } else {
            "executed_false"
        };
        if selected != expected_selected {
            return Err(
                "runtime branch phi selected path disagrees with executed branch".to_owned(),
            );
        }
        let has_true = phi.incoming.iter().any(|value| value == "executed_true");
        let has_false = phi.incoming.iter().any(|value| value == "executed_false");
        if !has_true || !has_false {
            return Err(
                "runtime branch phi incoming paths must include executed_true and executed_false"
                    .to_owned(),
            );
        }
        *phi_count_by_region.entry(region_index).or_insert(0) += 1;
    }
    for region_index in runtime_region_entered_by_index.keys() {
        if phi_count_by_region.get(region_index) != Some(&1) {
            return Err("each runtime branch region must have exactly one phi node".to_owned());
        }
    }
    Ok(())
}

fn evaluate_branch_effect(effect: &ProgramADEffect, operation: &str) -> Result<f64, String> {
    if effect.kind != "control_branch" {
        return Err(format!(
            "effect {} branch operation must have kind control_branch",
            effect.index
        ));
    }
    if !effect.inputs.is_empty() {
        return Err(format!(
            "effect {} branch operation must not carry differentiable inputs",
            effect.index
        ));
    }
    Ok(if branch_operation_value(operation)? {
        1.0
    } else {
        0.0
    })
}

fn branch_operation_value(operation: &str) -> Result<bool, String> {
    if operation.ends_with(":True") {
        Ok(true)
    } else if operation.ends_with(":False") {
        Ok(false)
    } else {
        Err("branch operation must end with :True or :False".to_owned())
    }
}

fn unary(
    effect: &ProgramADEffect,
    values: &HashMap<String, f64>,
    function: fn(f64) -> f64,
) -> Result<f64, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let value = operand_value(&effect.inputs[0], values)?;
    Ok(function(value))
}

fn unary_checked(
    effect: &ProgramADEffect,
    values: &HashMap<String, f64>,
    function: fn(f64) -> f64,
    finite_error: &str,
) -> Result<f64, String> {
    let value = unary(effect, values, function)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(finite_error.to_owned())
    }
}

fn unary_domain(
    effect: &ProgramADEffect,
    values: &HashMap<String, f64>,
    predicate: fn(f64) -> bool,
    function: fn(f64) -> f64,
    domain_error: &str,
) -> Result<f64, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let value = operand_value(&effect.inputs[0], values)?;
    if !predicate(value) {
        return Err(domain_error.to_owned());
    }
    let result = function(value);
    if result.is_finite() {
        Ok(result)
    } else {
        Err(format!("effect {} result must be finite", effect.index))
    }
}

fn binary(
    effect: &ProgramADEffect,
    values: &HashMap<String, f64>,
    function: impl Fn(f64, f64) -> Result<f64, String>,
) -> Result<f64, String> {
    if effect.inputs.len() != 2 {
        return Err(format!("effect {} requires two inputs", effect.index));
    }
    let lhs = operand_value(&effect.inputs[0], values)?;
    let rhs = operand_value(&effect.inputs[1], values)?;
    let value = function(lhs, rhs)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("effect {} result must be finite", effect.index))
    }
}

fn operand_value(name: &str, values: &HashMap<String, f64>) -> Result<f64, String> {
    if let Some(value) = values.get(name) {
        return Ok(*value);
    }
    name.parse::<f64>()
        .map_err(|_| format!("operand {name} is neither an SSA value nor a scalar literal"))
}

fn require_non_empty(value: &str, name: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("program AD IR {name} must be non-empty"));
    }
    Ok(())
}

fn require_positive_optional(value: Option<usize>, name: &str) -> Result<(), String> {
    if value == Some(0) {
        return Err(format!(
            "program AD IR {name} must be positive when present"
        ));
    }
    Ok(())
}

/// PyO3 wrapper returning a JSON metadata summary for a Program AD IR payload.
#[pyfunction]
pub fn program_ad_effect_ir_metadata_summary(serialization: &str) -> PyResult<String> {
    let ir = parse_program_ad_effect_ir(serialization).map_err(PyValueError::new_err)?;
    serde_json::to_string(&ir.metadata_summary()).map_err(|error| {
        PyValueError::new_err(format!("failed to encode Program AD IR summary: {error}"))
    })
}

/// PyO3 wrapper returning JSON for bounded Rust scalar Program AD interpretation.
#[pyfunction]
pub fn program_ad_effect_ir_interpret_forward(
    serialization: &str,
    inputs: Vec<f64>,
) -> PyResult<String> {
    let result = interpret_program_ad_effect_ir_forward(serialization, &inputs)
        .map_err(PyValueError::new_err)?;
    serde_json::to_string(&result).map_err(|error| {
        PyValueError::new_err(format!(
            "failed to encode Program AD IR interpreter result: {error}"
        ))
    })
}

/// PyO3 wrapper returning JSON for bounded Rust scalar Program AD value+gradient replay.
#[pyfunction]
pub fn program_ad_effect_ir_interpret_value_and_gradient(
    serialization: &str,
    inputs: Vec<f64>,
) -> PyResult<String> {
    let result = interpret_program_ad_effect_ir_value_and_gradient(serialization, &inputs)
        .map_err(PyValueError::new_err)?;
    serde_json::to_string(&result).map_err(|error| {
        PyValueError::new_err(format!(
            "failed to encode Program AD IR value+gradient result: {error}"
        ))
    })
}
