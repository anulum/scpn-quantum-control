// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD registry metadata mirror

//! Metadata-only mirror for Python Program AD registry coverage snapshots.
//!
//! The mirror validates the JSON shape emitted by Python's
//! `program_ad_registry_dispatch_coverage_report().to_dict()` and reports
//! deterministic family, facet and current Rust-executable primitive overlap.
//! It does not execute Program AD, promote registry primitives to Rust, or
//! claim LLVM/JIT, provider, hardware, or performance evidence.

use std::collections::{BTreeMap, HashSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const REGISTRY_METADATA_MIRROR_CLAIM_BOUNDARY: &str =
    "rust_program_ad_registry_metadata_mirror_only_no_execution_promotion";

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ProgramADRegistryDispatchCoverageSnapshot {
    supported: bool,
    covered_primitives: usize,
    total_primitives: usize,
    blocked_identities: Vec<String>,
    family_counts: BTreeMap<String, usize>,
    rows: Vec<ProgramADRegistryDispatchCoverageRowSnapshot>,
    claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ProgramADRegistryDispatchCoverageRowSnapshot {
    family: String,
    primitive: String,
    identity: String,
    derivative_rule: Option<String>,
    has_batching_rule: bool,
    has_lowering_rule: bool,
    has_lowering_metadata: bool,
    has_shape_rule: bool,
    has_dtype_rule: bool,
    has_static_argument_rule: bool,
    nondifferentiable_policy: Option<String>,
    effect: Option<String>,
    lowering_metadata_keys: Vec<String>,
    complete: bool,
    blocked_reasons: Vec<String>,
    claim_boundary: String,
}

/// JSON-ready metadata mirror of Python Program AD registry coverage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProgramADRegistryMetadataMirrorSummary {
    pub supported: bool,
    pub primitive_count: usize,
    pub covered_primitives: usize,
    pub family_counts: BTreeMap<String, usize>,
    pub facet_counts: BTreeMap<String, usize>,
    pub executable_operation_count: usize,
    pub executable_operations: Vec<String>,
    pub blocked_reasons: Vec<String>,
    pub claim_boundary: String,
}

/// Validate a Python Program AD registry coverage snapshot and mirror its metadata.
pub fn mirror_program_ad_registry_metadata(
    snapshot: &str,
) -> Result<ProgramADRegistryMetadataMirrorSummary, String> {
    if snapshot.trim().is_empty() {
        return Err("Program AD registry metadata snapshot must be non-empty JSON".to_owned());
    }
    let payload: Value = serde_json::from_str(snapshot).map_err(|error| {
        format!("Program AD registry metadata snapshot is invalid JSON: {error}")
    })?;
    if !payload.is_object() {
        return Err("Program AD registry metadata snapshot must decode to an object".to_owned());
    }
    let snapshot: ProgramADRegistryDispatchCoverageSnapshot = serde_json::from_value(payload)
        .map_err(|error| {
            format!("Program AD registry metadata snapshot does not match schema: {error}")
        })?;
    validate_registry_snapshot(&snapshot)?;

    let facet_counts = registry_facet_counts(&snapshot.rows);
    let executable_operations = executable_registry_operations(&snapshot.rows);
    let blocked_reasons = registry_blocked_reasons(&snapshot);

    Ok(ProgramADRegistryMetadataMirrorSummary {
        supported: snapshot.supported && blocked_reasons.is_empty(),
        primitive_count: snapshot.total_primitives,
        covered_primitives: snapshot.covered_primitives,
        family_counts: snapshot.family_counts,
        facet_counts,
        executable_operation_count: executable_operations.len(),
        executable_operations,
        blocked_reasons,
        claim_boundary: REGISTRY_METADATA_MIRROR_CLAIM_BOUNDARY.to_owned(),
    })
}

fn validate_registry_snapshot(
    snapshot: &ProgramADRegistryDispatchCoverageSnapshot,
) -> Result<(), String> {
    let mut errors = Vec::new();
    require_non_empty(
        &snapshot.claim_boundary,
        "Program AD registry claim_boundary",
        &mut errors,
    );
    if snapshot.total_primitives == 0 {
        errors.push("Program AD registry total_primitives must be positive".to_owned());
    }
    if snapshot.rows.len() != snapshot.total_primitives {
        errors.push(format!(
            "Program AD registry rows length {} does not match total_primitives {}",
            snapshot.rows.len(),
            snapshot.total_primitives
        ));
    }
    if snapshot.covered_primitives > snapshot.total_primitives {
        errors.push("Program AD registry covered_primitives exceeds total_primitives".to_owned());
    }

    let mut observed_family_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut complete_count = 0usize;
    for row in &snapshot.rows {
        validate_registry_row(row, &snapshot.claim_boundary, &mut errors);
        *observed_family_counts
            .entry(row.family.clone())
            .or_insert(0) += 1;
        if row.complete {
            complete_count += 1;
        }
    }

    if snapshot.covered_primitives != complete_count {
        errors.push(format!(
            "Program AD registry covered_primitives {} does not match complete row count {}",
            snapshot.covered_primitives, complete_count
        ));
    }
    let family_total: usize = snapshot.family_counts.values().sum();
    if family_total != snapshot.total_primitives {
        errors.push(format!(
            "Program AD registry family_counts sum {family_total} does not match total_primitives {}",
            snapshot.total_primitives
        ));
    }
    if snapshot.family_counts != observed_family_counts {
        errors.push(format!(
            "Program AD registry family_counts mismatch: expected {:?}, observed {:?}",
            snapshot.family_counts, observed_family_counts
        ));
    }
    if snapshot
        .blocked_identities
        .iter()
        .any(|identity| identity.is_empty())
    {
        errors.push("Program AD registry blocked_identities must be non-empty".to_owned());
    }
    let expected_supported = snapshot.blocked_identities.is_empty()
        && snapshot.covered_primitives == snapshot.total_primitives;
    if snapshot.supported != expected_supported {
        errors.push(format!(
            "Program AD registry supported flag {} does not match coverage state {}",
            snapshot.supported, expected_supported
        ));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn validate_registry_row(
    row: &ProgramADRegistryDispatchCoverageRowSnapshot,
    snapshot_claim_boundary: &str,
    errors: &mut Vec<String>,
) {
    require_non_empty(&row.family, "Program AD registry row family", errors);
    require_non_empty(&row.primitive, "Program AD registry row primitive", errors);
    require_non_empty(&row.identity, "Program AD registry row identity", errors);
    if !row.identity.contains(':') || !row.identity.contains('@') {
        errors.push(format!(
            "Program AD registry identity {} must include family and version",
            row.identity
        ));
    }
    require_non_empty(
        &row.claim_boundary,
        "Program AD registry row claim_boundary",
        errors,
    );
    if row.claim_boundary != snapshot_claim_boundary {
        errors.push(format!(
            "Program AD registry row {} claim_boundary diverges from snapshot",
            row.identity
        ));
    }
    if row.blocked_reasons.iter().any(|reason| reason.is_empty()) {
        errors.push(format!(
            "Program AD registry row {} blocked_reasons must be non-empty",
            row.identity
        ));
    }
    if row.complete && !row.blocked_reasons.is_empty() {
        errors.push(format!(
            "Program AD registry complete row {} must not carry blocked reasons",
            row.identity
        ));
    }
    if !row.complete && row.blocked_reasons.is_empty() {
        errors.push(format!(
            "Program AD registry incomplete row {} must carry blocked reasons",
            row.identity
        ));
    }
    if row.complete {
        validate_complete_registry_row(row, errors);
    }
}

fn validate_complete_registry_row(
    row: &ProgramADRegistryDispatchCoverageRowSnapshot,
    errors: &mut Vec<String>,
) {
    for (field, present) in [
        (
            "derivative_rule",
            has_present_optional(row.derivative_rule.as_deref()),
        ),
        ("batching_rule", row.has_batching_rule),
        ("lowering_metadata", row.has_lowering_metadata),
        ("shape_rule", row.has_shape_rule),
        ("dtype_rule", row.has_dtype_rule),
        ("static_argument_rule", row.has_static_argument_rule),
        (
            "nondifferentiable_policy",
            has_present_optional(row.nondifferentiable_policy.as_deref()),
        ),
        ("effect", has_present_optional(row.effect.as_deref())),
    ] {
        if !present {
            errors.push(format!(
                "Program AD registry complete row {} is missing {field}",
                row.identity
            ));
        }
    }
    let metadata_keys = row
        .lowering_metadata_keys
        .iter()
        .map(String::as_str)
        .collect::<HashSet<&str>>();
    for key in [
        "mlir_op",
        "nondifferentiable_boundary",
        "nondifferentiable_boundary_policy",
    ] {
        if !metadata_keys.contains(key) {
            errors.push(format!(
                "Program AD registry complete row {} is missing lowering metadata key {key}",
                row.identity
            ));
        }
    }
}

fn registry_facet_counts(
    rows: &[ProgramADRegistryDispatchCoverageRowSnapshot],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    counts.insert(
        "derivative_rule".to_owned(),
        rows.iter()
            .filter(|row| has_present_optional(row.derivative_rule.as_deref()))
            .count(),
    );
    counts.insert(
        "batching_rule".to_owned(),
        rows.iter().filter(|row| row.has_batching_rule).count(),
    );
    counts.insert(
        "lowering_rule".to_owned(),
        rows.iter().filter(|row| row.has_lowering_rule).count(),
    );
    counts.insert(
        "lowering_metadata".to_owned(),
        rows.iter().filter(|row| row.has_lowering_metadata).count(),
    );
    counts.insert(
        "shape_rule".to_owned(),
        rows.iter().filter(|row| row.has_shape_rule).count(),
    );
    counts.insert(
        "dtype_rule".to_owned(),
        rows.iter().filter(|row| row.has_dtype_rule).count(),
    );
    counts.insert(
        "static_argument_rule".to_owned(),
        rows.iter()
            .filter(|row| row.has_static_argument_rule)
            .count(),
    );
    counts.insert(
        "nondifferentiable_policy".to_owned(),
        rows.iter()
            .filter(|row| has_present_optional(row.nondifferentiable_policy.as_deref()))
            .count(),
    );
    counts.insert(
        "effect".to_owned(),
        rows.iter()
            .filter(|row| has_present_optional(row.effect.as_deref()))
            .count(),
    );
    counts
}

fn executable_registry_operations(
    rows: &[ProgramADRegistryDispatchCoverageRowSnapshot],
) -> Vec<String> {
    let supported = rust_supported_registry_primitives();
    let mut seen = HashSet::new();
    let mut operations = Vec::new();
    for row in rows {
        if row.complete
            && supported.contains(row.primitive.as_str())
            && seen.insert(row.primitive.clone())
        {
            operations.push(row.primitive.clone());
        }
    }
    operations.sort();
    operations
}

fn rust_supported_registry_primitives() -> HashSet<&'static str> {
    [
        "abs",
        "add",
        "arccos",
        "arcsin",
        "cos",
        "det",
        "divide",
        "eigh",
        "exp",
        "expm1",
        "eigvals",
        "eigvalsh",
        "log",
        "log1p",
        "multi_dot",
        "multiply",
        "pinv",
        "power",
        "reciprocal",
        "sin",
        "sqrt",
        "subtract",
        "svd",
        "tan",
        "tanh",
        "trace",
    ]
    .into_iter()
    .collect()
}

fn registry_blocked_reasons(snapshot: &ProgramADRegistryDispatchCoverageSnapshot) -> Vec<String> {
    let mut blocked = Vec::new();
    for identity in &snapshot.blocked_identities {
        blocked.push(format!("blocked primitive identity: {identity}"));
    }
    for row in &snapshot.rows {
        if row.complete {
            continue;
        }
        if row.blocked_reasons.is_empty() {
            blocked.push(format!("{} is incomplete", row.identity));
        } else {
            for reason in &row.blocked_reasons {
                blocked.push(format!("{}: {reason}", row.identity));
            }
        }
    }
    blocked
}

fn has_present_optional(value: Option<&str>) -> bool {
    value.is_some_and(|text| !text.is_empty() && text != "not_declared")
}

fn require_non_empty(value: &str, name: &str, errors: &mut Vec<String>) {
    if value.is_empty() {
        errors.push(format!("{name} must be non-empty"));
    }
}

/// PyO3 wrapper returning JSON for a metadata-only Program AD registry mirror.
#[pyfunction]
pub fn program_ad_registry_metadata_mirror(snapshot: &str) -> PyResult<String> {
    let result = mirror_program_ad_registry_metadata(snapshot).map_err(PyValueError::new_err)?;
    serde_json::to_string(&result).map_err(|error| {
        PyValueError::new_err(format!(
            "failed to encode Program AD registry metadata mirror: {error}"
        ))
    })
}
