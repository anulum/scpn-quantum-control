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
//! tooling can inspect evidence metadata, execute a narrow scalar forward
//! interpreter, and replay bounded scalar, elementwise-array, static structural,
//! static source-map indexing, static product reductions, and static-linalg
//! value+gradient traces when opcode-bearing rows are present.
//! It does not promote LLVM lowering, JIT execution, reverse-mode compiler AD,
//! hardware execution, or performance claims.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::program_ad_product_reduction::{
    product_all_cotangent, product_all_value, product_axis_cotangent, product_axis_values,
};
pub use crate::program_ad_registry_mirror::{
    mirror_program_ad_registry_metadata, ProgramADRegistryMetadataMirrorSummary,
};
use crate::program_ad_static_source_map::{
    apply_static_source_map, scatter_static_source_map_cotangent,
};

const PROGRAM_AD_EFFECT_IR_FORMAT: &str = "program_ad_effect_ir.v1";
const PROGRAM_AD_IR_CLAIM_BOUNDARY: &str = "metadata_only_no_program_execution";
const PROGRAM_AD_RUST_INTERPRETER_CLAIM_BOUNDARY: &str =
    "bounded_rust_program_ad_ir_scalar_and_static_linalg_primitives_executed_branch_view_alias_only_no_llvm_jit";
const PROGRAM_AD_RUST_VALUE_AND_GRADIENT_CLAIM_BOUNDARY: &str =
    "bounded_rust_program_ad_ir_elementwise_structural_array_and_static_linalg_primitives_value_and_gradient_executed_branch_view_alias_only_no_llvm_jit";

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

/// JSON-ready result for bounded Rust Program AD value and gradient replay.
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

fn accumulate_reverse_effect(
    effect: &ProgramADEffect,
    operation: &str,
    cotangent: ProgramADNumericValue,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
) -> Result<(), String> {
    match operation {
        operation if operation.starts_with("branch:") => Ok(()),
        name if name == "sum" || name.starts_with("sum:") => {
            accumulate_sum(effect, name, values, adjoints, &cotangent)
        }
        name if name == "mean" || name.starts_with("mean:") => {
            accumulate_mean(effect, name, values, adjoints, &cotangent)
        }
        name if name == "prod" || name.starts_with("prod:") => {
            accumulate_prod(effect, name, values, adjoints, &cotangent)
        }
        "reshape" | "ravel" => accumulate_reshape_like(effect, values, adjoints, &cotangent),
        "broadcast_to" => accumulate_broadcast_to(effect, values, adjoints, &cotangent),
        "transpose" => accumulate_transpose(effect, values, adjoints, &cotangent),
        name if name == "concatenate" || name.starts_with("concatenate:") => {
            accumulate_concatenate(effect, name, values, adjoints, &cotangent)
        }
        name if name == "stack" || name.starts_with("stack:") => {
            accumulate_stack(effect, name, values, adjoints, &cotangent)
        }
        name if name == "index_map" || name.starts_with("index_map:") => {
            accumulate_index_map(effect, name, values, adjoints, &cotangent)
        }
        "add" => accumulate_add_sub(effect, values, adjoints, &cotangent, 1.0, 1.0),
        "sub" => accumulate_add_sub(effect, values, adjoints, &cotangent, 1.0, -1.0),
        "mul" => {
            let (lhs, rhs, _shape) = binary_operands(effect, values)?;
            let lhs_contribution =
                elementwise_mul(&cotangent, &broadcast_to(&rhs, &cotangent.shape)?)?;
            let rhs_contribution =
                elementwise_mul(&cotangent, &broadcast_to(&lhs, &cotangent.shape)?)?;
            add_numeric_adjoint(&effect.inputs[0], lhs_contribution, values, adjoints)?;
            add_numeric_adjoint(&effect.inputs[1], rhs_contribution, values, adjoints)
        }
        "div" => {
            let (lhs, rhs, _shape) = binary_operands(effect, values)?;
            if rhs.values.contains(&0.0) {
                return Err("division denominator must be non-zero".to_owned());
            }
            let rhs_broadcast = broadcast_to(&rhs, &cotangent.shape)?;
            let lhs_broadcast = broadcast_to(&lhs, &cotangent.shape)?;
            let lhs_contribution =
                elementwise_binary(&cotangent, &rhs_broadcast, |cot, r| Ok(cot / r))?;
            let rhs_contribution =
                elementwise_binary3(&cotangent, &lhs_broadcast, &rhs_broadcast, |cot, l, r| {
                    Ok(cot * (-l / (r * r)))
                })?;
            add_numeric_adjoint(&effect.inputs[0], lhs_contribution, values, adjoints)?;
            add_numeric_adjoint(&effect.inputs[1], rhs_contribution, values, adjoints)
        }
        "pow" => {
            let (lhs, rhs, _shape) = binary_operands(effect, values)?;
            if lhs.values.iter().any(|value| *value <= 0.0) {
                return Err("pow gradient requires a positive base".to_owned());
            }
            let lhs_broadcast = broadcast_to(&lhs, &cotangent.shape)?;
            let rhs_broadcast = broadcast_to(&rhs, &cotangent.shape)?;
            let lhs_contribution =
                elementwise_binary3(&cotangent, &lhs_broadcast, &rhs_broadcast, |cot, l, r| {
                    Ok(cot * r * l.powf(r - 1.0))
                })?;
            let rhs_contribution =
                elementwise_binary3(&cotangent, &lhs_broadcast, &rhs_broadcast, |cot, l, r| {
                    Ok(cot * l.powf(r) * l.ln())
                })?;
            add_numeric_adjoint(&effect.inputs[0], lhs_contribution, values, adjoints)?;
            add_numeric_adjoint(&effect.inputs[1], rhs_contribution, values, adjoints)
        }
        "sin" => accumulate_unary(effect, values, adjoints, &cotangent, f64::cos),
        "cos" => accumulate_unary(effect, values, adjoints, &cotangent, |value| -value.sin()),
        "exp" => accumulate_unary(effect, values, adjoints, &cotangent, f64::exp),
        "expm1" => accumulate_unary(effect, values, adjoints, &cotangent, f64::exp),
        "log" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value > 0.0,
            |value| 1.0 / value,
            "log input must be positive",
        ),
        "log1p" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value > -1.0,
            |value| 1.0 / (1.0 + value),
            "log1p input must be greater than -1",
        ),
        "sqrt" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value > 0.0,
            |value| 0.5 / value.sqrt(),
            "sqrt input must be positive",
        ),
        "tan" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value.cos().abs() > 1.0e-15,
            |value| 1.0 / (value.cos() * value.cos()),
            "tan input must have non-zero cosine",
        ),
        "tanh" => accumulate_unary(effect, values, adjoints, &cotangent, |value| {
            let tanh = value.tanh();
            1.0 - tanh * tanh
        }),
        "arcsin" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value.abs() < 1.0,
            |value| 1.0 / (1.0 - value * value).sqrt(),
            "arcsin input must be strictly inside (-1, 1)",
        ),
        "arccos" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value.abs() < 1.0,
            |value| -1.0 / (1.0 - value * value).sqrt(),
            "arccos input must be strictly inside (-1, 1)",
        ),
        "reciprocal" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value != 0.0,
            |value| -1.0 / (value * value),
            "reciprocal input must be non-zero",
        ),
        "abs" => accumulate_unary_domain(
            effect,
            values,
            adjoints,
            &cotangent,
            |value| value != 0.0,
            |value| value.signum(),
            "abs gradient is undefined at zero",
        ),
        name if name.starts_with("linalg:trace:") => {
            let cotangent_scalar = cotangent.scalar_value()?;
            // d(trace)/d(diagonal element) = 1 for each on-diagonal operand.
            for input in &effect.inputs {
                add_scalar_adjoint(input, cotangent_scalar, values, adjoints)?;
            }
            Ok(())
        }
        "linalg:det:2x2" => {
            let cotangent_scalar = cotangent.scalar_value()?;
            // Cofactor adjoints for det = a*d - b*c: d/da = d, d/db = -c, d/dc = -b, d/dd = a.
            if effect.inputs.len() != 4 {
                return Err(format!(
                    "effect {} linalg:det:2x2 requires four operands",
                    effect.index
                ));
            }
            let a = operand_scalar_value(&effect.inputs[0], values)?;
            let b = operand_scalar_value(&effect.inputs[1], values)?;
            let c = operand_scalar_value(&effect.inputs[2], values)?;
            let d = operand_scalar_value(&effect.inputs[3], values)?;
            add_scalar_adjoint(&effect.inputs[0], cotangent_scalar * d, values, adjoints)?;
            add_scalar_adjoint(&effect.inputs[1], cotangent_scalar * (-c), values, adjoints)?;
            add_scalar_adjoint(&effect.inputs[2], cotangent_scalar * (-b), values, adjoints)?;
            add_scalar_adjoint(&effect.inputs[3], cotangent_scalar * a, values, adjoints)?;
            Ok(())
        }
        "linalg:det:3x3" => {
            let cotangent_scalar = cotangent.scalar_value()?;
            // d(det)/dA_{ij} is the (i,j) cofactor of the row-major 3x3 matrix.
            if effect.inputs.len() != 9 {
                return Err(format!(
                    "effect {} linalg:det:3x3 requires nine operands",
                    effect.index
                ));
            }
            let [a, b, c, d, e, f, g, h, i] = read_3x3_numeric(effect, values)?;
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
                add_scalar_adjoint(input, cotangent_scalar * cofactor, values, adjoints)?;
            }
            Ok(())
        }
        name if name.starts_with("linalg:det:") => {
            let cotangent_scalar = cotangent.scalar_value()?;
            // General determinant (4x4 and up): d(det)/dA_{ij} = det * (A^{-1})_{ji}.
            let n = parse_det_dim(name).ok_or_else(|| {
                format!(
                    "effect {} {name} has no determinant dimension",
                    effect.index
                )
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
                .map(|input| operand_scalar_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            let determinant = determinant_general(&matrix, n)?;
            let inverse = invert_square(&matrix, n)?;
            for i in 0..n {
                for j in 0..n {
                    let cofactor = determinant * inverse[j * n + i];
                    add_scalar_adjoint(
                        &effect.inputs[i * n + j],
                        cotangent_scalar * cofactor,
                        values,
                        adjoints,
                    )?;
                }
            }
            Ok(())
        }
        name if name.starts_with("linalg:inv:") => {
            let cotangent_scalar = cotangent.scalar_value()?;
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
                .map(|input| operand_scalar_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            let m = invert_square(&matrix, n)?;
            for k in 0..n {
                for l in 0..n {
                    let contribution = cotangent_scalar * (-m[row * n + k] * m[l * n + column]);
                    add_scalar_adjoint(&effect.inputs[k * n + l], contribution, values, adjoints)?;
                }
            }
            Ok(())
        }
        name if name.starts_with("linalg:solve:") => {
            let cotangent_scalar = cotangent.scalar_value()?;
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
                .map(|input| operand_scalar_value(input, values))
                .collect::<Result<Vec<f64>, String>>()?;
            let m = invert_square(&operands[..n * n], n)?;
            let rhs = &operands[n * n..];
            let x: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| m[i * n + j] * rhs[j]).sum())
                .collect();
            for j in 0..n {
                add_scalar_adjoint(
                    &effect.inputs[n * n + j],
                    cotangent_scalar * m[row * n + j],
                    values,
                    adjoints,
                )?;
            }
            for k in 0..n {
                for (l, x_l) in x.iter().enumerate().take(n) {
                    let contribution = cotangent_scalar * (-m[row * n + k] * *x_l);
                    add_scalar_adjoint(&effect.inputs[k * n + l], contribution, values, adjoints)?;
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

fn accumulate_sum(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} sum requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let contribution = if operation == "sum" {
        let scalar_cotangent = cotangent.scalar_value()?;
        ProgramADNumericValue::filled(&input.shape, scalar_cotangent)?
    } else {
        expand_axis_reduction_cotangent(effect.index, operation, "sum", &input, cotangent, 1.0)?
    };
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_mean(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} mean requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let contribution = if operation == "mean" {
        let scalar_cotangent = cotangent.scalar_value()?;
        let scale = scalar_cotangent / input.values.len() as f64;
        ProgramADNumericValue::filled(&input.shape, scale)?
    } else {
        let axis = parse_static_axis(operation, "mean", input.shape.len())?;
        let scale = 1.0 / input.shape[axis] as f64;
        expand_axis_reduction_cotangent(effect.index, operation, "mean", &input, cotangent, scale)?
    };
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_prod(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} prod requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let contribution_values = if operation == "prod" {
        let scalar_cotangent = cotangent.scalar_value()?;
        product_all_cotangent(effect.index, &input.values, scalar_cotangent)?
    } else {
        let axis = parse_static_axis(operation, "prod", input.shape.len())?;
        product_axis_cotangent(
            effect.index,
            &input.shape,
            axis,
            &cotangent.values,
            &input.values,
        )?
    };
    let contribution = ProgramADNumericValue::new(input.shape.clone(), contribution_values)?;
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_reshape_like(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} reshape/ravel requires one input",
            effect.index
        ));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let reshaped = ProgramADNumericValue::new(input.shape.clone(), cotangent.values.clone())?;
    add_numeric_adjoint(&effect.inputs[0], reshaped, values, adjoints)
}

fn accumulate_broadcast_to(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} broadcast_to requires one input",
            effect.index
        ));
    }
    add_numeric_adjoint(&effect.inputs[0], cotangent.clone(), values, adjoints)
}

fn accumulate_transpose(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} transpose requires one input",
            effect.index
        ));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let contribution = transpose_reversed_axes(cotangent, &input.shape)?;
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_concatenate(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let operands = numeric_operands(effect, values)?;
    let contributions = split_concatenate_cotangent(effect.index, operation, &operands, cotangent)?;
    for (input, contribution) in effect.inputs.iter().zip(contributions) {
        add_numeric_adjoint(input, contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_stack(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    let operands = numeric_operands(effect, values)?;
    let contributions = split_stack_cotangent(effect.index, operation, &operands, cotangent)?;
    for (input, contribution) in effect.inputs.iter().zip(contributions) {
        add_numeric_adjoint(input, contribution, values, adjoints)?;
    }
    Ok(())
}

fn accumulate_index_map(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!(
            "effect {} index_map requires one input",
            effect.index
        ));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let contribution_values = scatter_static_source_map_cotangent(
        effect.index,
        operation,
        input.values.len(),
        &cotangent.values,
    )?;
    let contribution = ProgramADNumericValue::new(input.shape.clone(), contribution_values)?;
    add_numeric_adjoint(&effect.inputs[0], contribution, values, adjoints)
}

fn accumulate_add_sub(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
    lhs_sign: f64,
    rhs_sign: f64,
) -> Result<(), String> {
    if effect.inputs.len() != 2 {
        return Err(format!("effect {} requires two inputs", effect.index));
    }
    add_numeric_adjoint(
        &effect.inputs[0],
        scale_value(cotangent, lhs_sign)?,
        values,
        adjoints,
    )?;
    add_numeric_adjoint(
        &effect.inputs[1],
        scale_value(cotangent, rhs_sign)?,
        values,
        adjoints,
    )
}

fn accumulate_unary(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
    derivative: impl Fn(f64) -> f64,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    let derivative_values = ProgramADNumericValue::new(
        input.shape.clone(),
        input
            .values
            .iter()
            .map(|value| derivative(*value))
            .collect(),
    )?;
    add_numeric_adjoint(
        &effect.inputs[0],
        elementwise_mul(cotangent, &derivative_values)?,
        values,
        adjoints,
    )
}

fn accumulate_unary_domain(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
    cotangent: &ProgramADNumericValue,
    predicate: impl Fn(f64) -> bool,
    derivative: impl Fn(f64) -> f64,
    domain_error: &str,
) -> Result<(), String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    if input.values.iter().any(|value| !predicate(*value)) {
        return Err(domain_error.to_owned());
    }
    let derivative_values = ProgramADNumericValue::new(
        input.shape.clone(),
        input
            .values
            .iter()
            .map(|value| derivative(*value))
            .collect(),
    )?;
    add_numeric_adjoint(
        &effect.inputs[0],
        elementwise_mul(cotangent, &derivative_values)?,
        values,
        adjoints,
    )
}

fn add_scalar_adjoint(
    input: &str,
    contribution: f64,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
) -> Result<(), String> {
    add_numeric_adjoint(
        input,
        ProgramADNumericValue::scalar(contribution),
        values,
        adjoints,
    )
}

fn add_numeric_adjoint(
    input: &str,
    contribution: ProgramADNumericValue,
    values: &HashMap<String, ProgramADNumericValue>,
    adjoints: &mut HashMap<String, ProgramADNumericValue>,
) -> Result<(), String> {
    if contribution.values.iter().any(|value| !value.is_finite()) {
        return Err(format!("adjoint contribution for {input} must be finite"));
    }
    let Some(target) = values.get(input) else {
        return Ok(());
    };
    let reduced = reduce_to_shape(&contribution, &target.shape)?;
    let entry = adjoints.entry(input.to_owned()).or_insert_with(|| {
        ProgramADNumericValue::filled(&target.shape, 0.0)
            .expect("zero adjoint shape is already validated")
    });
    if entry.shape != reduced.shape {
        return Err(format!(
            "adjoint shape {:?} does not match contribution shape {:?}",
            entry.shape, reduced.shape
        ));
    }
    for (slot, value) in entry.values.iter_mut().zip(reduced.values.iter()) {
        *slot += value;
    }
    Ok(())
}

fn evaluate_numeric_effect(
    effect: &ProgramADEffect,
    operation: &str,
    inputs: &[f64],
    input_index: &mut usize,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if operation == "parameter" {
        if effect.kind != "parameter" {
            return Err(format!(
                "effect {} operation parameter must have kind parameter",
                effect.index
            ));
        }
        let shape = target_shape(effect, shapes_by_target)?;
        let size = shape_size(&shape)?;
        let end = input_index
            .checked_add(size)
            .ok_or_else(|| "Program AD parameter input index overflowed".to_owned())?;
        let Some(slice) = inputs.get(*input_index..end) else {
            return Err(format!(
                "effect {} parameter input is missing flattened values",
                effect.index
            ));
        };
        *input_index = end;
        return ProgramADNumericValue::new(shape, slice.to_vec());
    }
    if operation.starts_with("branch:") {
        return evaluate_branch_effect(effect, operation).map(ProgramADNumericValue::scalar);
    }
    match operation {
        name if name == "sum" || name.starts_with("sum:") => {
            numeric_sum(effect, name, values, shapes_by_target)
        }
        name if name == "mean" || name.starts_with("mean:") => {
            numeric_mean(effect, name, values, shapes_by_target)
        }
        name if name == "prod" || name.starts_with("prod:") => {
            numeric_prod(effect, name, values, shapes_by_target)
        }
        "reshape" => numeric_reshape(effect, values, shapes_by_target),
        "ravel" => numeric_ravel(effect, values, shapes_by_target),
        "broadcast_to" => numeric_broadcast_to(effect, values, shapes_by_target),
        "transpose" => numeric_transpose(effect, values, shapes_by_target),
        name if name == "concatenate" || name.starts_with("concatenate:") => {
            numeric_concatenate(effect, name, values, shapes_by_target)
        }
        name if name == "stack" || name.starts_with("stack:") => {
            numeric_stack(effect, name, values, shapes_by_target)
        }
        name if name == "index_map" || name.starts_with("index_map:") => {
            numeric_index_map(effect, name, values, shapes_by_target)
        }
        "add" => numeric_binary(effect, values, |lhs, rhs| Ok(lhs + rhs)),
        "sub" => numeric_binary(effect, values, |lhs, rhs| Ok(lhs - rhs)),
        "mul" => numeric_binary(effect, values, |lhs, rhs| Ok(lhs * rhs)),
        "div" => numeric_binary(effect, values, |lhs, rhs| {
            if rhs == 0.0 {
                Err("division denominator must be non-zero".to_owned())
            } else {
                Ok(lhs / rhs)
            }
        }),
        "pow" => numeric_binary(effect, values, |lhs, rhs| {
            let value = lhs.powf(rhs);
            if value.is_finite() {
                Ok(value)
            } else {
                Err("power result must be finite".to_owned())
            }
        }),
        "sin" => numeric_unary(effect, values, f64::sin),
        "cos" => numeric_unary(effect, values, f64::cos),
        "exp" => numeric_unary_checked(effect, values, f64::exp, "exp result must be finite"),
        "expm1" => {
            numeric_unary_checked(effect, values, f64::exp_m1, "expm1 result must be finite")
        }
        "log" => numeric_unary_domain(
            effect,
            values,
            |value| value > 0.0,
            f64::ln,
            "log input must be positive",
        ),
        "log1p" => numeric_unary_domain(
            effect,
            values,
            |value| value > -1.0,
            f64::ln_1p,
            "log1p input must be greater than -1",
        ),
        "sqrt" => numeric_unary_domain(
            effect,
            values,
            |value| value > 0.0,
            f64::sqrt,
            "sqrt input must be positive",
        ),
        "tan" => numeric_unary_domain(
            effect,
            values,
            |value| value.cos().abs() > 1.0e-15,
            f64::tan,
            "tan input must have non-zero cosine",
        ),
        "tanh" => numeric_unary(effect, values, f64::tanh),
        "arcsin" => numeric_unary_domain(
            effect,
            values,
            |value| value.abs() < 1.0,
            f64::asin,
            "arcsin input must be strictly inside (-1, 1)",
        ),
        "arccos" => numeric_unary_domain(
            effect,
            values,
            |value| value.abs() < 1.0,
            f64::acos,
            "arccos input must be strictly inside (-1, 1)",
        ),
        "reciprocal" => numeric_unary_domain(
            effect,
            values,
            |value| value != 0.0,
            |value| 1.0 / value,
            "reciprocal input must be non-zero",
        ),
        "abs" => numeric_unary(effect, values, f64::abs),
        name if name.starts_with("linalg:trace:")
            || name.starts_with("linalg:det:")
            || name.starts_with("linalg:inv:")
            || name.starts_with("linalg:solve:") =>
        {
            evaluate_scalar_linalg_effect(effect, operation, values)
        }
        _ => Err(format!(
            "effect {} operation {operation} is outside bounded Rust elementwise/structural array value+gradient replay",
            effect.index
        )),
    }
}

fn evaluate_scalar_linalg_effect(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    let scalar_values = values
        .iter()
        .map(|(key, value)| value.scalar_value().map(|scalar| (key.clone(), scalar)))
        .collect::<Result<HashMap<String, f64>, String>>()?;
    let mut input_index = 0usize;
    evaluate_effect(effect, operation, &[], &mut input_index, &scalar_values)
        .map(ProgramADNumericValue::scalar)
}

fn ssa_shapes_by_target(ir: &ProgramADEffectIR) -> HashMap<String, Vec<usize>> {
    ir.ssa_values
        .iter()
        .map(|value| (value.name.clone(), value.shape.clone()))
        .collect()
}

fn target_shape(
    effect: &ProgramADEffect,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<Vec<usize>, String> {
    shapes_by_target
        .get(&effect.target)
        .cloned()
        .ok_or_else(|| {
            format!(
                "effect {} target {} is missing SSA shape metadata",
                effect.index, effect.target
            )
        })
}

fn parameter_targets_for_effect(
    effect: &ProgramADEffect,
    value: &ProgramADNumericValue,
) -> Vec<ScalarParameterTarget> {
    if value.shape.is_empty() {
        return vec![ScalarParameterTarget {
            label: effect.target.clone(),
            source: effect.target.clone(),
            flat_index: 0,
        }];
    }
    (0..value.values.len())
        .map(|flat_index| ScalarParameterTarget {
            label: format!("{}[{flat_index}]", effect.target),
            source: effect.target.clone(),
            flat_index,
        })
        .collect()
}

fn shape_size(shape: &[usize]) -> Result<usize, String> {
    let mut size = 1usize;
    for dimension in shape {
        if *dimension == 0 {
            return Err("Program AD shaped values must have non-zero dimensions".to_owned());
        }
        size = size
            .checked_mul(*dimension)
            .ok_or_else(|| "Program AD shaped value size overflowed".to_owned())?;
    }
    Ok(size)
}

fn numeric_operand(
    name: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<ProgramADNumericValue, String> {
    if let Some(value) = values.get(name) {
        return Ok(value.clone());
    }
    name.parse::<f64>()
        .map(ProgramADNumericValue::scalar)
        .map_err(|_| format!("operand {name} is neither an SSA value nor a scalar literal"))
}

fn numeric_operands(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<Vec<ProgramADNumericValue>, String> {
    if effect.inputs.is_empty() {
        return Err(format!(
            "effect {} requires at least one input",
            effect.index
        ));
    }
    effect
        .inputs
        .iter()
        .map(|input| numeric_operand(input, values))
        .collect()
}

fn operand_scalar_value(
    name: &str,
    values: &HashMap<String, ProgramADNumericValue>,
) -> Result<f64, String> {
    numeric_operand(name, values)?.scalar_value()
}

fn numeric_unary(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    function: fn(f64) -> f64,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    ProgramADNumericValue::new(
        input.shape,
        input.values.into_iter().map(function).collect::<Vec<f64>>(),
    )
}

fn numeric_unary_checked(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    function: fn(f64) -> f64,
    finite_error: &str,
) -> Result<ProgramADNumericValue, String> {
    let value = numeric_unary(effect, values, function)?;
    if value.values.iter().all(|item| item.is_finite()) {
        Ok(value)
    } else {
        Err(finite_error.to_owned())
    }
}

fn numeric_unary_domain(
    effect: &ProgramADEffect,
    values: &HashMap<String, ProgramADNumericValue>,
    predicate: fn(f64) -> bool,
    function: fn(f64) -> f64,
    domain_error: &str,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} requires one input", effect.index));
    }
    let input = numeric_operand(&effect.inputs[0], values)?;
    if input.values.iter().any(|value| !predicate(*value)) {
        return Err(domain_error.to_owned());
    }
    ProgramADNumericValue::new(
        input.shape,
        input.values.into_iter().map(function).collect::<Vec<f64>>(),
    )
}

fn numeric_sum(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} sum requires one input", effect.index));
    }
    let target = target_shape(effect, shapes_by_target)?;
    let source = numeric_operand(&effect.inputs[0], values)?;
    if operation == "sum" {
        if !target.is_empty() {
            return Err(format!(
                "effect {} sum non-scalar target requires static axis metadata sum:axis:<int>",
                effect.index
            ));
        }
        return Ok(ProgramADNumericValue::scalar(source.values.iter().sum()));
    }
    reduce_axis_values(effect.index, operation, "sum", &source, &target, 1.0)
}

fn numeric_mean(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} mean requires one input", effect.index));
    }
    let target = target_shape(effect, shapes_by_target)?;
    let source = numeric_operand(&effect.inputs[0], values)?;
    if operation == "mean" {
        if !target.is_empty() {
            return Err(format!(
                "effect {} mean non-scalar target requires static axis metadata mean:axis:<int>",
                effect.index
            ));
        }
        let total: f64 = source.values.iter().sum();
        return Ok(ProgramADNumericValue::scalar(
            total / source.values.len() as f64,
        ));
    }
    let axis = parse_static_axis(operation, "mean", source.shape.len())?;
    let scale = 1.0 / source.shape[axis] as f64;
    reduce_axis_values(effect.index, operation, "mean", &source, &target, scale)
}

fn numeric_prod(
    effect: &ProgramADEffect,
    operation: &str,
    values: &HashMap<String, ProgramADNumericValue>,
    shapes_by_target: &HashMap<String, Vec<usize>>,
) -> Result<ProgramADNumericValue, String> {
    if effect.inputs.len() != 1 {
        return Err(format!("effect {} prod requires one input", effect.index));
    }
    let target = target_shape(effect, shapes_by_target)?;
    let source = numeric_operand(&effect.inputs[0], values)?;
    if operation == "prod" {
        if !target.is_empty() {
            return Err(format!(
                "effect {} prod non-scalar target requires static axis metadata prod:axis:<int>",
                effect.index
            ));
        }
        return Ok(ProgramADNumericValue::scalar(product_all_value(
            effect.index,
            &source.values,
        )?));
    }
    let axis = parse_static_axis(operation, "prod", source.shape.len())?;
    let output = product_axis_values(effect.index, &source.shape, axis, &target, &source.values)?;
    ProgramADNumericValue::new(target, output)
}

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
                return Err(format!(
                    "effect {} linalg:det:3x3 requires nine operands",
                    effect.index
                ));
            }
            let m = read_3x3(effect, values)?;
            let [a, b, c, d, e, f, g, h, i] = m;
            Ok(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))
        }
        name if name.starts_with("linalg:det:") => {
            // General determinant (4x4 and up) via LU factorisation with partial pivoting.
            let n = parse_det_dim(name).ok_or_else(|| {
                format!(
                    "effect {} {name} has no determinant dimension",
                    effect.index
                )
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
