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
    "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_executed_branch_view_alias_only_no_llvm_jit";
const PROGRAM_AD_RUST_VALUE_AND_GRADIENT_CLAIM_BOUNDARY: &str =
    "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_value_and_gradient_executed_branch_view_alias_only_no_llvm_jit";

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

/// Return true if the IR carries an alias edge that is not an inert read-only view.
///
/// `view_alias` edges record reshape, transpose and slice views. The forward-AD trace has
/// already resolved those views into canonical scalar SSA targets, so the scalar replay is
/// unaffected: an op-effect that still referenced a view name would fail closed in
/// [`operand_value`] rather than read a wrong value. Every other alias kind (mutation,
/// control-path, rebinding, list) can change a value's content and stays outside the bounded
/// scalar replay.
fn has_non_view_alias(ir: &ProgramADEffectIR) -> bool {
    ir.alias_edges.iter().any(|edge| edge.kind != "view_alias")
}

/// Return true when the final effect is a raw element of a multi-output linalg op.
///
/// Inverse and linear solve emit one effect per output element. The IR does not record
/// which element a program ultimately returns, so the last-ordered effect is not a reliable
/// proxy when the result is an indexed element (for example `solve(A, b)[0]`); such programs
/// fail closed rather than replaying the wrong component. Single-output linalg ops
/// (determinant, trace) are unaffected because their one effect is the result.
fn final_effect_is_indexed_multi_output_linalg(effect: &ProgramADEffect) -> bool {
    effect
        .operation
        .as_deref()
        .is_some_and(|op| op.starts_with("linalg:inv:") || op.starts_with("linalg:solve:"))
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
    if has_non_view_alias(&ir) {
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
    if has_non_view_alias(ir) {
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
        name if name.starts_with("linalg:trace:") => {
            // d(trace)/d(diagonal element) = 1 for each on-diagonal operand.
            for input in &effect.inputs {
                add_adjoint(input, cotangent, values, adjoints)?;
            }
            Ok(())
        }
        "linalg:det:2x2" => {
            // Cofactor adjoints for det = a*d - b*c: d/da = d, d/db = -c, d/dc = -b, d/dd = a.
            if effect.inputs.len() != 4 {
                return Err(format!(
                    "effect {} linalg:det:2x2 requires four operands",
                    effect.index
                ));
            }
            let a = operand_value(&effect.inputs[0], values)?;
            let b = operand_value(&effect.inputs[1], values)?;
            let c = operand_value(&effect.inputs[2], values)?;
            let d = operand_value(&effect.inputs[3], values)?;
            add_adjoint(&effect.inputs[0], cotangent * d, values, adjoints)?;
            add_adjoint(&effect.inputs[1], cotangent * (-c), values, adjoints)?;
            add_adjoint(&effect.inputs[2], cotangent * (-b), values, adjoints)?;
            add_adjoint(&effect.inputs[3], cotangent * a, values, adjoints)?;
            Ok(())
        }
        "linalg:det:3x3" => {
            // d(det)/dA_{ij} is the (i,j) cofactor of the row-major 3x3 matrix.
            if effect.inputs.len() != 9 {
                return Err(format!("effect {} linalg:det:3x3 requires nine operands", effect.index));
            }
            let [a, b, c, d, e, f, g, h, i] = read_3x3(effect, values)?;
            let cofactors = [
                e * i - f * h,
                f * g - d * i,
                d * h - e * g,
                c * h - b * i,
                a * i - c * g,
                b * g - a * h,
                b * f - c * e,
                c * d - a * f,
                a * e - b * d,
            ];
            for (input, cofactor) in effect.inputs.iter().zip(cofactors.iter()) {
                add_adjoint(input, cotangent * cofactor, values, adjoints)?;
            }
            Ok(())
        }
        name if name.starts_with("linalg:det:") => {
            // General determinant (4x4 and up): d(det)/dA_{ij} = det * (A^{-1})_{ji}.
            let n = parse_det_dim(name).ok_or_else(|| {
                format!("effect {} {name} has no determinant dimension", effect.index)
            })?;
            if effect.inputs.len() != n * n {
                return Err(format!(
                    "effect {} {name} requires {} operands",
                    effect.index,
                    n * n
                ));
            }
            let matrix = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            let determinant = determinant_general(&matrix, n)?;
            let inverse = invert_square(&matrix, n)?;
            for i in 0..n {
                for j in 0..n {
                    let cofactor = determinant * inverse[j * n + i];
                    add_adjoint(&effect.inputs[i * n + j], cotangent * cofactor, values, adjoints)?;
                }
            }
            Ok(())
        }
        name if name.starts_with("linalg:inv:") => {
            // d(A^{-1})_{ij}/dA_{kl} = -(A^{-1})_{ik} (A^{-1})_{lj}.
            let (n, row, column) = parse_inv_index(name)
                .ok_or_else(|| format!("effect {} {name} has no inverse index", effect.index))?;
            if effect.inputs.len() != n * n {
                return Err(format!(
                    "effect {} {name} requires {} operands",
                    effect.index,
                    n * n
                ));
            }
            let matrix = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            let m = invert_square(&matrix, n)?;
            for k in 0..n {
                for l in 0..n {
                    let contribution = cotangent * (-m[row * n + k] * m[l * n + column]);
                    add_adjoint(&effect.inputs[k * n + l], contribution, values, adjoints)?;
                }
            }
            Ok(())
        }
        name if name.starts_with("linalg:solve:") => {
            // x = A^{-1} b: dx_i/db_j = (A^{-1})_{ij}; dx_i/dA_{kl} = -(A^{-1})_{ik} x_l.
            let (n, row) = parse_solve_index(name)
                .ok_or_else(|| format!("effect {} {name} has no solution index", effect.index))?;
            if effect.inputs.len() != n * n + n {
                return Err(format!(
                    "effect {} {name} requires {} operands",
                    effect.index,
                    n * n + n
                ));
            }
            let operands = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            let m = invert_square(&operands[..n * n], n)?;
            let rhs = &operands[n * n..];
            let x: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| m[i * n + j] * rhs[j]).sum())
                .collect();
            for j in 0..n {
                add_adjoint(&effect.inputs[n * n + j], cotangent * m[row * n + j], values, adjoints)?;
            }
            for k in 0..n {
                for l in 0..n {
                    let contribution = cotangent * (-m[row * n + k] * x[l]);
                    add_adjoint(&effect.inputs[k * n + l], contribution, values, adjoints)?;
                }
            }
            Ok(())
        }
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
        name if name.starts_with("linalg:trace:") => {
            // The trace opcode carries the on-diagonal element operands; its value is their sum.
            let mut total = 0.0;
            for input in &effect.inputs {
                total += operand_value(input, values)?;
            }
            Ok(total)
        }
        "linalg:det:2x2" => {
            // Row-major operands [a, b, c, d]; det = a*d - b*c.
            if effect.inputs.len() != 4 {
                return Err(format!(
                    "effect {} linalg:det:2x2 requires four operands",
                    effect.index
                ));
            }
            let a = operand_value(&effect.inputs[0], values)?;
            let b = operand_value(&effect.inputs[1], values)?;
            let c = operand_value(&effect.inputs[2], values)?;
            let d = operand_value(&effect.inputs[3], values)?;
            Ok(a * d - b * c)
        }
        "linalg:det:3x3" => {
            // Row-major operands [a,b,c, d,e,f, g,h,i]; Laplace expansion along the first row.
            if effect.inputs.len() != 9 {
                return Err(format!("effect {} linalg:det:3x3 requires nine operands", effect.index));
            }
            let m = read_3x3(effect, values)?;
            let [a, b, c, d, e, f, g, h, i] = m;
            Ok(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))
        }
        name if name.starts_with("linalg:det:") => {
            // General determinant (4x4 and up) via LU factorisation with partial pivoting.
            let n = parse_det_dim(name).ok_or_else(|| {
                format!("effect {} {name} has no determinant dimension", effect.index)
            })?;
            if effect.inputs.len() != n * n {
                return Err(format!(
                    "effect {} {name} requires {} operands",
                    effect.index,
                    n * n
                ));
            }
            let matrix = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            determinant_general(&matrix, n)
        }
        name if name.starts_with("linalg:inv:") => {
            // Each opcode emits one element (row, column) of the matrix inverse.
            let (n, row, column) = parse_inv_index(name)
                .ok_or_else(|| format!("effect {} {name} has no inverse index", effect.index))?;
            if effect.inputs.len() != n * n {
                return Err(format!(
                    "effect {} {name} requires {} operands",
                    effect.index,
                    n * n
                ));
            }
            let matrix = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            Ok(invert_square(&matrix, n)?[row * n + column])
        }
        name if name.starts_with("linalg:solve:") => {
            // Each opcode emits one component i of x = A^{-1} b.
            let (n, row) = parse_solve_index(name)
                .ok_or_else(|| format!("effect {} {name} has no solution index", effect.index))?;
            if effect.inputs.len() != n * n + n {
                return Err(format!(
                    "effect {} {name} requires {} operands",
                    effect.index,
                    n * n + n
                ));
            }
            let operands = effect
                .inputs
                .iter()
                .map(|input| operand_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            let inverse = invert_square(&operands[..n * n], n)?;
            let rhs = &operands[n * n..];
            Ok((0..n).map(|j| inverse[row * n + j] * rhs[j]).sum())
        }
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

/// Invert a row-major 2x2 matrix `[a, b; c, d]`, returning `[m00, m01, m10, m11]`.
///
/// Fails closed on a singular or non-finite determinant so a degenerate inverse is never
/// silently replayed.
fn invert_2x2(a: f64, b: f64, c: f64, d: f64) -> Result<[f64; 4], String> {
    let det = a * d - b * c;
    if det == 0.0 || !det.is_finite() {
        return Err("linalg 2x2 matrix is singular".to_owned());
    }
    Ok([d / det, -b / det, -c / det, a / det])
}

/// Read the nine row-major operands of a 3x3 linalg opcode as `[a, b, c, d, e, f, g, h, i]`.
fn read_3x3(effect: &ProgramADEffect, values: &HashMap<String, f64>) -> Result<[f64; 9], String> {
    let mut matrix = [0.0_f64; 9];
    for (slot, input) in matrix.iter_mut().zip(effect.inputs.iter()) {
        *slot = operand_value(input, values)?;
    }
    Ok(matrix)
}

/// Invert a row-major 3x3 matrix via the adjugate, returning the inverse row-major.
///
/// Fails closed on a singular or non-finite determinant.
fn invert_3x3(m: [f64; 9]) -> Result<[f64; 9], String> {
    let [a, b, c, d, e, f, g, h, i] = m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det == 0.0 || !det.is_finite() {
        return Err("linalg 3x3 matrix is singular".to_owned());
    }
    // inverse = adjugate / det = cofactor-transpose / det.
    Ok([
        (e * i - f * h) / det,
        (c * h - b * i) / det,
        (b * f - c * e) / det,
        (f * g - d * i) / det,
        (a * i - c * g) / det,
        (c * d - a * f) / det,
        (d * h - e * g) / det,
        (b * g - a * h) / det,
        (a * e - b * d) / det,
    ])
}

/// Invert an `n x n` row-major matrix for the bounded dimensions; fail closed otherwise.
fn invert_square(matrix: &[f64], n: usize) -> Result<Vec<f64>, String> {
    match n {
        2 => invert_2x2(matrix[0], matrix[1], matrix[2], matrix[3]).map(|m| m.to_vec()),
        3 => {
            let mut m = [0.0_f64; 9];
            m.copy_from_slice(&matrix[..9]);
            invert_3x3(m).map(|inv| inv.to_vec())
        }
        _ => invert_general(matrix, n),
    }
}

/// Invert an `n x n` row-major matrix by Gauss-Jordan elimination with partial pivoting.
///
/// Fails closed on a singular or non-finite system. Used for dimensions above the closed-form
/// 2x2/3x3 paths.
fn invert_general(matrix: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let width = 2 * n;
    let mut augmented = vec![0.0_f64; n * width];
    for row in 0..n {
        for column in 0..n {
            augmented[row * width + column] = matrix[row * n + column];
        }
        augmented[row * width + n + row] = 1.0;
    }
    for column in 0..n {
        let mut pivot = column;
        let mut best = augmented[column * width + column].abs();
        for row in (column + 1)..n {
            let candidate = augmented[row * width + column].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        if best == 0.0 || !best.is_finite() {
            return Err(format!("linalg {n}x{n} matrix is singular"));
        }
        if pivot != column {
            for c in 0..width {
                augmented.swap(pivot * width + c, column * width + c);
            }
        }
        let pivot_value = augmented[column * width + column];
        for c in 0..width {
            augmented[column * width + c] /= pivot_value;
        }
        for row in 0..n {
            if row != column {
                let factor = augmented[row * width + column];
                if factor != 0.0 {
                    for c in 0..width {
                        augmented[row * width + c] -= factor * augmented[column * width + c];
                    }
                }
            }
        }
    }
    let mut inverse = vec![0.0_f64; n * n];
    for row in 0..n {
        for column in 0..n {
            inverse[row * n + column] = augmented[row * width + n + column];
        }
    }
    if inverse.iter().any(|value| !value.is_finite()) {
        return Err(format!("linalg {n}x{n} inverse is non-finite"));
    }
    Ok(inverse)
}

/// Determinant of an `n x n` row-major matrix by LU factorisation with partial pivoting.
fn determinant_general(matrix: &[f64], n: usize) -> Result<f64, String> {
    let mut work = matrix.to_vec();
    let mut sign = 1.0_f64;
    for column in 0..n {
        let mut pivot = column;
        let mut best = work[column * n + column].abs();
        for row in (column + 1)..n {
            let candidate = work[row * n + column].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        if best == 0.0 {
            return Ok(0.0);
        }
        if pivot != column {
            for c in 0..n {
                work.swap(pivot * n + c, column * n + c);
            }
            sign = -sign;
        }
        let pivot_value = work[column * n + column];
        for row in (column + 1)..n {
            let factor = work[row * n + column] / pivot_value;
            for c in column..n {
                work[row * n + c] -= factor * work[column * n + c];
            }
        }
    }
    let mut determinant = sign;
    for k in 0..n {
        determinant *= work[k * n + k];
    }
    if !determinant.is_finite() {
        return Err(format!("linalg {n}x{n} determinant is non-finite"));
    }
    Ok(determinant)
}

/// Parse the square dimension `n` from a `linalg:det:NxN` opcode.
fn parse_det_dim(operation: &str) -> Option<usize> {
    let parts: Vec<&str> = operation.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    parse_square_dim(parts[2])
}

/// Parse the square dimension `n` from an `NxN` opcode token.
fn parse_square_dim(token: &str) -> Option<usize> {
    let (rows, columns) = token.split_once('x')?;
    let n: usize = rows.parse().ok()?;
    (n > 0 && columns.parse::<usize>().ok()? == n).then_some(n)
}

/// Parse `(n, row, column)` from a `linalg:inv:NxN:I:J` opcode.
fn parse_inv_index(operation: &str) -> Option<(usize, usize, usize)> {
    let parts: Vec<&str> = operation.split(':').collect();
    if parts.len() != 5 {
        return None;
    }
    let n = parse_square_dim(parts[2])?;
    let row: usize = parts[3].parse().ok()?;
    let column: usize = parts[4].parse().ok()?;
    (row < n && column < n).then_some((n, row, column))
}

/// Parse `(n, component)` from a `linalg:solve:NxN:rhs:<m>:I` opcode.
fn parse_solve_index(operation: &str) -> Option<(usize, usize)> {
    let parts: Vec<&str> = operation.split(':').collect();
    if parts.len() != 6 {
        return None;
    }
    let n = parse_square_dim(parts[2])?;
    let component: usize = parts[5].parse().ok()?;
    (component < n).then_some((n, component))
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
