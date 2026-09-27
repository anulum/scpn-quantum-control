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
    crate::program_ad_lifecycle::replay_checkpoint()?;
    if serialization.trim().is_empty() {
        return Err("program AD IR serialization must be non-empty".to_owned());
    }
    let ir = metadata_parser::parse(serialization)?;
    crate::program_ad_lifecycle::replay_checkpoint()?;
    validate_program_ad_effect_ir(&ir)?;
    crate::program_ad_lifecycle::replay_checkpoint()?;
    Ok(ir)
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
