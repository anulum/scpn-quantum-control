// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD IR schema and parser

const PROGRAM_AD_EFFECT_IR_FORMAT: &str = "program_ad_effect_ir.v1";
const PROGRAM_AD_IR_CLAIM_BOUNDARY: &str = "metadata_only_no_program_execution";
const PROGRAM_AD_RUST_INTERPRETER_CLAIM_BOUNDARY: &str =
    "bounded_rust_program_ad_ir_scalar_static_signal_static_interpolation_static_stencil_static_cumulative_and_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit";
const PROGRAM_AD_RUST_VALUE_AND_GRADIENT_CLAIM_BOUNDARY: &str =
    "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit";

/// One SSA value record from Python-emitted Program AD metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProgramADSSAValue {
    /// Canonical SSA name this record defines.
    pub name: String,
    /// Index of the effect that produced the value.
    pub producer: usize,
    /// Monotonic version of this name, distinguishing repeated assignments.
    pub version: usize,
    /// Array shape as emitted by the tracer; empty for a scalar.
    pub shape: Vec<usize>,
    /// Element type name as emitted by the tracer.
    pub dtype: String,
    /// Index of the effect this value belongs to in the ordered effect list.
    pub effect: usize,
}

/// One ordered effect record from Python-emitted Program AD metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProgramADEffect {
    /// Position of this effect in the emitted list.
    pub index: usize,
    /// Effect category, which decides how the replay interprets the record.
    pub kind: String,
    /// SSA name this effect writes.
    pub target: String,
    /// SSA names this effect reads, in operand order.
    pub inputs: Vec<String>,
    /// Version of the written target, matching its SSA value record.
    pub version: usize,
    /// Total order of execution, which the replay follows rather than `index`.
    pub ordering: usize,
    /// Operation name for an op-effect; absent for effects that carry none.
    #[serde(default)]
    pub operation: Option<String>,
}

/// One alias edge record from Python-emitted Program AD metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProgramADAliasEdge {
    /// SSA name the view is taken from.
    pub source: String,
    /// SSA name that denotes the view.
    pub target: String,
    /// Kind of aliasing, such as a reshape, transpose or slice view.
    pub kind: String,
    /// Version of the target name at the point the edge was recorded.
    pub version: usize,
}

/// One control-flow region record from Python-emitted Program AD metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProgramADControlRegion {
    /// Position of this region in the emitted list.
    pub index: usize,
    /// Region category, such as a branch or a loop body.
    pub kind: String,
    /// SSA name of the controlling predicate, when the region has one.
    pub predicate: Option<String>,
    /// Whether the traced execution actually entered this region.
    pub entered: bool,
    /// Source line the region came from, when the tracer recorded one.
    pub source_line: Option<usize>,
}

/// One metadata-only phi record from Python-emitted Program AD metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProgramADPhiNode {
    /// Position of this phi record in the emitted list.
    pub index: usize,
    /// SSA name the phi defines.
    pub target: String,
    /// Candidate SSA names reaching the merge point.
    pub incoming: Vec<String>,
    /// Index of the control region this phi belongs to, when it has one.
    pub control_region: Option<usize>,
    /// Which incoming name the traced run selected, when it is recorded.
    pub selected: Option<String>,
    /// Source line the merge came from, when the tracer recorded one.
    pub source_line: Option<usize>,
}

/// Parsed Rust view of a `program_ad_effect_ir.v1` payload.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProgramADEffectIR {
    /// Schema identifier the payload declares; the parser fails closed on any
    /// value other than the one it implements.
    pub format: String,
    /// Every SSA value the trace defined.
    pub ssa_values: Vec<ProgramADSSAValue>,
    /// Every recorded effect, to be replayed in `ordering`, not list order.
    pub effects: Vec<ProgramADEffect>,
    /// View relationships between SSA names.
    pub alias_edges: Vec<ProgramADAliasEdge>,
    /// Control regions the trace passed through.
    pub control_regions: Vec<ProgramADControlRegion>,
    /// Metadata-only merge records; absent in payloads that emit none.
    #[serde(default)]
    pub phi_nodes: Vec<ProgramADPhiNode>,
    /// Bytecode offsets the trace was taken at, for source correlation.
    pub bytecode_offsets: Vec<usize>,
}

/// JSON-ready summary for Rust Program AD IR metadata inspection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProgramADEffectIRMetadataSummary {
    /// Schema identifier the summarised payload declared.
    pub format: String,
    /// Number of SSA value records.
    pub ssa_value_count: usize,
    /// Number of effect records.
    pub effect_count: usize,
    /// Number of alias edges.
    pub alias_edge_count: usize,
    /// Number of control regions.
    pub control_region_count: usize,
    /// Number of phi records.
    pub phi_node_count: usize,
    /// Number of recorded bytecode offsets.
    pub bytecode_offset_count: usize,
    /// What this summary may and may not be cited as evidence for.
    pub claim_boundary: String,
}

/// JSON-ready result for bounded Rust scalar Program AD IR interpretation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProgramADRustInterpreterResult {
    /// Whether the whole program lay inside the bounded set. When false, no
    /// value is produced rather than a partial one.
    pub supported: bool,
    /// The interpreted scalar, present only when `supported` holds.
    pub value: Option<f64>,
    /// Number of effects in the payload.
    pub effect_count: usize,
    /// Number of effects the bounded interpreter could execute.
    pub supported_effect_count: usize,
    /// Why the program was refused, one entry per distinct reason.
    pub blocked_reasons: Vec<String>,
    /// What this result may and may not be cited as evidence for.
    pub claim_boundary: String,
}

/// JSON-ready result for bounded Rust Program AD value and gradient replay.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProgramADRustValueAndGradientResult {
    /// Whether the whole program lay inside the bounded set. When false,
    /// neither a value nor a gradient is produced.
    pub supported: bool,
    /// The replayed scalar, present only when `supported` holds.
    pub value: Option<f64>,
    /// Reverse-mode gradient, ordered to match `parameter_targets`.
    pub gradient: Vec<f64>,
    /// SSA names the gradient is taken with respect to, in gradient order.
    pub parameter_targets: Vec<String>,
    /// Number of effects in the payload.
    pub effect_count: usize,
    /// Number of effects the bounded interpreter could execute.
    pub supported_effect_count: usize,
    /// Why the program was refused, one entry per distinct reason.
    pub blocked_reasons: Vec<String>,
    /// What this result may and may not be cited as evidence for.
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

/// Return true if the IR carries alias metadata that can change replay semantics.
///
/// `view_alias` edges record reshape, transpose and slice views. The forward-AD trace has
/// already resolved those views into canonical scalar SSA targets, so the scalar replay is
/// unaffected: an op-effect that still referenced a view name would fail closed in
/// [`operand_value`] rather than read a wrong value. Source-level
/// `alias_analysis:assignment_binding` and `expression_rebinding_alias` rows are deterministic
/// frontend evidence for ordinary local expression assignments and do not introduce replay
/// aliases. Mutation, control-path, local-name rebinding, list, and object aliases can change
/// value identity or content and stay outside the bounded replay.
fn has_replay_unsafe_alias(ir: &ProgramADEffectIR) -> bool {
    ir.alias_edges
        .iter()
        .any(|edge| !is_replay_inert_alias(edge))
}

fn is_replay_inert_alias(edge: &ProgramADAliasEdge) -> bool {
    edge.kind == "view_alias"
        || (edge.kind == "alias_analysis"
            && edge.source == "assignment_binding"
            && edge.target.starts_with("source:"))
        || (edge.kind == "expression_rebinding_alias"
            && edge.source.starts_with("expr:")
            && edge.target.starts_with("name:"))
}

/// Return true when the final effect is a raw element of a multi-output linalg op.
///
/// Inverse, linear solve, and pseudoinverse emit one effect per output element. The IR
/// does not record which element a program ultimately returns, so the last-ordered effect
/// is not a reliable proxy when the result is an indexed element (for example
/// `solve(A, b)[0]`); such programs fail closed rather than replaying the wrong component.
/// Single-output linalg ops (determinant, trace) are unaffected because their one effect is
/// the result.
fn final_effect_is_indexed_multi_output_linalg(effect: &ProgramADEffect) -> bool {
    effect.operation.as_deref().is_some_and(|op| {
        op.starts_with("linalg:inv:")
            || op.starts_with("linalg:solve:")
            || op.starts_with("linalg:pinv:")
    })
}
