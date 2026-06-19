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
//! tooling can inspect evidence metadata without promoting a Rust Program AD
//! interpreter, LLVM lowering, or executable whole-program AD.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const PROGRAM_AD_EFFECT_IR_FORMAT: &str = "program_ad_effect_ir.v1";
const PROGRAM_AD_IR_CLAIM_BOUNDARY: &str = "metadata_only_no_program_execution";

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
