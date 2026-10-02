// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD metadata and input validation

fn replay_values_are_finite(values: &[f64]) -> Result<bool, String> {
    for chunk in values.chunks(256) {
        crate::program_ad_lifecycle::replay_checkpoint()?;
        if chunk.iter().any(|value| !value.is_finite()) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn validate_replay_inputs(
    ir: &ProgramADEffectIR,
    inputs: &[f64],
    finite_reason: &str,
    alias_reason: &str,
) -> Result<(), String> {
    if !replay_values_are_finite(inputs)? {
        return Err(finite_reason.to_owned());
    }
    if has_replay_unsafe_alias(ir)? {
        return Err(alias_reason.to_owned());
    }
    validate_executed_branch_metadata(ir)
}

fn validate_program_ad_effect_ir(ir: &ProgramADEffectIR) -> Result<(), String> {
    if ir.format != PROGRAM_AD_EFFECT_IR_FORMAT {
        return Err("program AD IR format must be program_ad_effect_ir.v1".to_owned());
    }
    for (index, value) in ir.ssa_values.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        require_non_empty(&value.name, "ssa_values name")?;
        require_non_empty(&value.dtype, "ssa_values dtype")?;
    }
    for (index, effect) in ir.effects.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        require_non_empty(&effect.kind, "effects kind")?;
        require_non_empty(&effect.target, "effects target")?;
        for (index, input) in effect.inputs.iter().enumerate() {
            if index % 256 == 0 {
                crate::program_ad_lifecycle::replay_checkpoint()?;
            }
            require_non_empty(input, "effects inputs")?;
        }
        if let Some(operation) = &effect.operation {
            require_non_empty(operation, "effects operation")?;
        }
    }
    for (index, edge) in ir.alias_edges.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        require_non_empty(&edge.source, "alias_edges source")?;
        require_non_empty(&edge.target, "alias_edges target")?;
        require_non_empty(&edge.kind, "alias_edges kind")?;
    }
    for (index, region) in ir.control_regions.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        require_non_empty(&region.kind, "control_regions kind")?;
        if let Some(predicate) = &region.predicate {
            require_non_empty(predicate, "control_regions predicate")?;
        }
        require_positive_optional(region.source_line, "control_regions source_line")?;
    }
    for (index, phi) in ir.phi_nodes.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        require_non_empty(&phi.target, "phi_nodes target")?;
        if phi.incoming.len() < 2 {
            return Err(
                "program AD IR phi_nodes incoming must contain at least two entries".to_owned(),
            );
        }
        for (index, incoming) in phi.incoming.iter().enumerate() {
            if index % 256 == 0 {
                crate::program_ad_lifecycle::replay_checkpoint()?;
            }
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
fn has_replay_unsafe_alias(ir: &ProgramADEffectIR) -> Result<bool, String> {
    for (index, edge) in ir.alias_edges.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        if !is_replay_inert_alias(edge) {
            return Ok(true);
        }
    }
    Ok(false)
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

fn validate_executed_branch_metadata(ir: &ProgramADEffectIR) -> Result<(), String> {
    let mut branch_effects_by_operation: HashMap<&str, usize> = HashMap::new();
    let mut branch_count = 0usize;
    for (index, effect) in ir.effects.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        if !matches!(
            effect.kind.as_str(),
            "parameter" | "pure" | "primitive" | "control_branch" | "mutation"
        ) {
            return Err(format!(
                "program AD effect {} has unsupported kind {:?}",
                effect.index, effect.kind
            ));
        }
        if effect
            .operation
            .as_deref()
            .is_some_and(|op| op.starts_with("branch:"))
        {
            branch_count += 1;
        }
    }
    crate::program_ad_lifecycle::replay_checkpoint()?;
    admit_replay_table::<(&str, usize)>(branch_count)?;
    branch_effects_by_operation
        .try_reserve(branch_count)
        .map_err(|error| format!("Program AD branch-map allocation refused: {error}"))?;
    for (index, effect) in ir.effects.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
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
    crate::program_ad_lifecycle::replay_checkpoint()?;
    admit_replay_table::<(usize, bool)>(ir.control_regions.len())?;
    runtime_region_entered_by_index
        .try_reserve(ir.control_regions.len())
        .map_err(|error| format!("Program AD runtime-region map allocation refused: {error}"))?;
    crate::program_ad_lifecycle::replay_checkpoint()?;
    admit_replay_table::<usize>(ir.control_regions.len())?;
    source_region_indices
        .try_reserve(ir.control_regions.len())
        .map_err(|error| format!("Program AD source-region set allocation refused: {error}"))?;
    for (index, region) in ir.control_regions.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
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
    crate::program_ad_lifecycle::replay_checkpoint()?;
    admit_replay_table::<(usize, usize)>(ir.phi_nodes.len())?;
    phi_count_by_region
        .try_reserve(ir.phi_nodes.len())
        .map_err(|error| format!("Program AD phi-count map allocation refused: {error}"))?;
    for (index, phi) in ir.phi_nodes.iter().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
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
        let mut has_true = false;
        let mut has_false = false;
        for (index, incoming) in phi.incoming.iter().enumerate() {
            if index % 256 == 0 {
                crate::program_ad_lifecycle::replay_checkpoint()?;
            }
            has_true |= incoming == "executed_true";
            has_false |= incoming == "executed_false";
            if has_true && has_false {
                break;
            }
        }
        if !has_true || !has_false {
            return Err(
                "runtime branch phi incoming paths must include executed_true and executed_false"
                    .to_owned(),
            );
        }
        *phi_count_by_region.entry(region_index).or_insert(0) += 1;
    }
    for (index, region_index) in runtime_region_entered_by_index.keys().enumerate() {
        if index % 256 == 0 {
            crate::program_ad_lifecycle::replay_checkpoint()?;
        }
        if phi_count_by_region.get(region_index) != Some(&1) {
            return Err("each runtime branch region must have exactly one phi node".to_owned());
        }
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
