// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — admitted Program AD metadata parsing

use serde_json::value::RawValue;
use super::{Context, FieldSlots, fields, optional, required, scalar, string, vector};
use super::super::{ProgramADEffectIR, ProgramADSSAValue, ProgramADEffect, ProgramADAliasEdge, ProgramADControlRegion, ProgramADPhiNode};

fn usize_vector(context: &Context, raw: &RawValue) -> Result<Vec<usize>, String> {
    vector(context, raw, scalar::<usize>)
}

fn string_vector(context: &Context, raw: &RawValue) -> Result<Vec<String>, String> {
    vector(context, raw, string)
}

fn ssa(context: &Context, raw: &RawValue) -> Result<ProgramADSSAValue, String> {
    let fields = fields(context, raw, &["name", "producer", "version", "shape", "dtype", "effect"])?;
    Ok(ProgramADSSAValue {
        name: string(context, required(fields[0], "name")?)?,
        producer: scalar(context, required(fields[1], "producer")?)?,
        version: scalar(context, required(fields[2], "version")?)?,
        shape: usize_vector(context, required(fields[3], "shape")?)?,
        dtype: string(context, required(fields[4], "dtype")?)?,
        effect: scalar(context, required(fields[5], "effect")?)?,
    })
}

fn effect(context: &Context, raw: &RawValue) -> Result<ProgramADEffect, String> {
    let fields = fields(context, raw, &["index", "kind", "target", "inputs", "version", "ordering", "operation"])?;
    Ok(ProgramADEffect {
        index: scalar(context, required(fields[0], "index")?)?,
        kind: string(context, required(fields[1], "kind")?)?,
        target: string(context, required(fields[2], "target")?)?,
        inputs: string_vector(context, required(fields[3], "inputs")?)?,
        version: scalar(context, required(fields[4], "version")?)?,
        ordering: scalar(context, required(fields[5], "ordering")?)?,
        operation: optional(context, fields[6], string)?,
    })
}

fn alias(context: &Context, raw: &RawValue) -> Result<ProgramADAliasEdge, String> {
    let fields = fields(context, raw, &["source", "target", "kind", "version"])?;
    Ok(ProgramADAliasEdge {
        source: string(context, required(fields[0], "source")?)?,
        target: string(context, required(fields[1], "target")?)?,
        kind: string(context, required(fields[2], "kind")?)?,
        version: scalar(context, required(fields[3], "version")?)?,
    })
}

fn region(context: &Context, raw: &RawValue) -> Result<ProgramADControlRegion, String> {
    let fields = fields(context, raw, &["index", "kind", "predicate", "entered", "source_line"])?;
    Ok(ProgramADControlRegion {
        index: scalar(context, required(fields[0], "index")?)?,
        kind: string(context, required(fields[1], "kind")?)?,
        predicate: optional(context, fields[2], string)?,
        entered: scalar(context, required(fields[3], "entered")?)?,
        source_line: optional(context, fields[4], scalar::<usize>)?,
    })
}

fn phi(context: &Context, raw: &RawValue) -> Result<ProgramADPhiNode, String> {
    let fields = fields(context, raw, &["index", "target", "incoming", "control_region", "selected", "source_line"])?;
    Ok(ProgramADPhiNode {
        index: scalar(context, required(fields[0], "index")?)?,
        target: string(context, required(fields[1], "target")?)?,
        incoming: string_vector(context, required(fields[2], "incoming")?)?,
        control_region: optional(context, fields[3], scalar::<usize>)?,
        selected: optional(context, fields[4], string)?,
        source_line: optional(context, fields[5], scalar::<usize>)?,
    })
}

pub(super) fn parse(context: &Context, serialization: &str) -> Result<ProgramADEffectIR, String> {
    if !serialization.trim_start().starts_with('{') {
        return Err("program AD IR serialization must decode to an object".to_owned());
    }
    let names = ["format", "ssa_values", "effects", "alias_edges", "control_regions", "phi_nodes", "bytecode_offsets"];
    let fields = context.decode(serialization, FieldSlots(context, &names, None))?;
    for index in 1..names.len() {
        let raw = fields[index].ok_or_else(|| format!("program AD IR {} must be present", names[index]))?;
        if !raw.get().starts_with('[') {
            return Err(format!("program AD IR {} must be a list", names[index]));
        }
    }
    Ok(ProgramADEffectIR {
        format: string(context, required(fields[0], "format")?)?,
        ssa_values: vector(context, required(fields[1], "ssa_values")?, ssa)?,
        effects: vector(context, required(fields[2], "effects")?, effect)?,
        alias_edges: vector(context, required(fields[3], "alias_edges")?, alias)?,
        control_regions: vector(context, required(fields[4], "control_regions")?, region)?,
        phi_nodes: vector(context, required(fields[5], "phi_nodes")?, phi)?,
        bytecode_offsets: usize_vector(context, required(fields[6], "bytecode_offsets")?)?,
    })
}
