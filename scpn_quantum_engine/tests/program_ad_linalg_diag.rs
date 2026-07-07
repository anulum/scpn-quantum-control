// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD diag replay tests

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use serde_json::json;

#[test]
fn value_and_gradient_replays_static_diag_vector_construct_nodes() {
    let inputs = [1.5, -2.0, 0.75];
    let terms = vec![
        (0, "linalg:diag:3:offset:1:construct:0", 0.5),
        (1, "linalg:diag:3:offset:1:construct:1", -0.25),
        (2, "linalg:diag:3:offset:1:construct:2", 1.5),
    ];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &weighted_diag_objective_ir(inputs.len(), &terms),
        &inputs,
    )
    .expect("static diag construct replay should serialize");

    assert!(result.supported, "{result:?}");
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        2.375,
    );
    for (actual, expected) in result.gradient.iter().zip([0.5, -0.25, 1.5]) {
        assert_close(*actual, expected);
    }
}

#[test]
fn value_and_gradient_replays_static_diag_matrix_extract_nodes() {
    let inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let terms = vec![
        (2, "linalg:diag:3x2:offset:-1:extract:0", 2.0),
        (5, "linalg:diag:3x2:offset:-1:extract:1", -0.5),
    ];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &weighted_diag_objective_ir(inputs.len(), &terms),
        &inputs,
    )
    .expect("static diag extract replay should serialize");

    assert!(result.supported, "{result:?}");
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        3.0,
    );
    for (actual, expected) in result.gradient.iter().zip([0.0, 0.0, 2.0, 0.0, 0.0, -0.5]) {
        assert_close(*actual, expected);
    }
}

#[test]
fn value_and_gradient_fails_closed_on_malformed_diag_mode() {
    let terms = vec![(0, "linalg:diag:2:offset:0:rotate:0", 1.0)];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &weighted_diag_objective_ir(2, &terms),
        &[1.0, 2.0],
    )
    .expect("malformed diag mode should serialize");

    assert!(!result.supported, "{result:?}");
    let error = result.blocked_reasons.join("; ");
    assert!(error.contains("diag mode metadata is malformed"), "{error}");
}

#[test]
fn value_and_gradient_fails_closed_on_out_of_range_diag_construct_index() {
    let terms = vec![(0, "linalg:diag:2:offset:0:construct:2", 1.0)];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &weighted_diag_objective_ir(2, &terms),
        &[1.0, 2.0],
    )
    .expect("out-of-range diag construct should serialize");

    assert!(!result.supported, "{result:?}");
    let error = result.blocked_reasons.join("; ");
    assert!(
        error.contains("diag construct source index is outside vector shape"),
        "{error}"
    );
}

#[test]
fn value_and_gradient_fails_closed_on_non_matrix_diag_extract() {
    let terms = vec![(0, "linalg:diag:2:offset:0:extract:0", 1.0)];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &weighted_diag_objective_ir(2, &terms),
        &[1.0, 2.0],
    )
    .expect("non-matrix diag extract should serialize");

    assert!(!result.supported, "{result:?}");
    let error = result.blocked_reasons.join("; ");
    assert!(
        error.contains("diag extract requires rank-2 source metadata"),
        "{error}"
    );
}

#[test]
fn value_and_gradient_fails_closed_on_out_of_range_diag_extract_index() {
    let terms = vec![(0, "linalg:diag:3x2:offset:-1:extract:2", 1.0)];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &weighted_diag_objective_ir(6, &terms),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .expect("out-of-range diag extract should serialize");

    assert!(!result.supported, "{result:?}");
    let error = result.blocked_reasons.join("; ");
    assert!(
        error.contains("diag extract source index is outside diagonal shape"),
        "{error}"
    );
}

fn weighted_diag_objective_ir(input_count: usize, terms: &[(usize, &str, f64)]) -> String {
    let mut ssa_values = Vec::new();
    let mut effects = Vec::new();
    for index in 0..input_count {
        ssa_values.push(json!({
            "name": format!("%{index}"),
            "producer": index,
            "version": 0,
            "shape": [],
            "dtype": "float64",
            "effect": index,
        }));
        effects.push(json!({
            "index": index,
            "kind": "parameter",
            "target": format!("%{index}"),
            "inputs": [format!("p{index}")],
            "version": 0,
            "ordering": index,
            "operation": "parameter",
        }));
    }

    let mut next_index = input_count;
    let mut weighted_targets = Vec::new();
    for (source_index, operation, weight) in terms {
        let diag_index = next_index;
        ssa_values.push(json!({
            "name": format!("%{diag_index}"),
            "producer": diag_index,
            "version": 0,
            "shape": [],
            "dtype": "float64",
            "effect": diag_index,
        }));
        effects.push(json!({
            "index": diag_index,
            "kind": "primitive",
            "target": format!("%{diag_index}"),
            "inputs": [format!("%{source_index}")],
            "version": 0,
            "ordering": diag_index,
            "operation": operation,
        }));
        next_index += 1;

        let mul_index = next_index;
        ssa_values.push(json!({
            "name": format!("%{mul_index}"),
            "producer": mul_index,
            "version": 0,
            "shape": [],
            "dtype": "float64",
            "effect": mul_index,
        }));
        effects.push(json!({
            "index": mul_index,
            "kind": "pure",
            "target": format!("%{mul_index}"),
            "inputs": [format!("%{diag_index}"), weight.to_string()],
            "version": 0,
            "ordering": mul_index,
            "operation": "mul",
        }));
        weighted_targets.push(mul_index);
        next_index += 1;
    }

    let final_target = sum_targets(&mut ssa_values, &mut effects, weighted_targets, next_index);
    json!({
        "format": "program_ad_effect_ir.v1",
        "ssa_values": ssa_values,
        "effects": effects,
        "alias_edges": [],
        "control_regions": [],
        "phi_nodes": [],
        "bytecode_offsets": [0],
        "return_value": format!("%{final_target}"),
    })
    .to_string()
}

fn sum_targets(
    ssa_values: &mut Vec<serde_json::Value>,
    effects: &mut Vec<serde_json::Value>,
    targets: Vec<usize>,
    mut next_index: usize,
) -> usize {
    let mut accumulated = targets[0];
    for target in targets.into_iter().skip(1) {
        let add_index = next_index;
        ssa_values.push(json!({
            "name": format!("%{add_index}"),
            "producer": add_index,
            "version": 0,
            "shape": [],
            "dtype": "float64",
            "effect": add_index,
        }));
        effects.push(json!({
            "index": add_index,
            "kind": "pure",
            "target": format!("%{add_index}"),
            "inputs": [format!("%{accumulated}"), format!("%{target}")],
            "version": 0,
            "ordering": add_index,
            "operation": "add",
        }));
        accumulated = add_index;
        next_index += 1;
    }
    accumulated
}

fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= 1.0e-12,
        "expected {expected}, got {actual}, delta {delta}"
    );
}
