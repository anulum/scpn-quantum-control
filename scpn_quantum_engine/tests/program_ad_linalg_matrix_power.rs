// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD matrix-power replay tests

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use serde_json::json;

#[test]
fn value_and_gradient_replays_static_matrix_power_positive_nodes() {
    let inputs = [1.5, 0.4, 0.2, 1.1];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &matrix_power_weighted_objective_ir(3, [0.5, -1.0, 0.75, 1.25]),
        &inputs,
    )
    .expect("static matrix_power replay should serialize");

    assert!(result.supported, "{result:?}");
    assert!(result
        .claim_boundary
        .contains("value_and_gradient_static_linalg_primitives_executed_branch_view_assignment"));
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        2.58775,
    );
    assert_eq!(result.gradient.len(), 4);
    for (actual, expected) in result.gradient.iter().zip([2.53, -3.905, 6.4625, 3.8525]) {
        assert_close(*actual, expected);
    }
}

#[test]
fn value_and_gradient_replays_static_matrix_power_negative_nodes() {
    let inputs = [1.5, 0.4, 0.2, 1.1];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &matrix_power_weighted_objective_ir(-2, [1.25, -0.5, 0.2, 0.75]),
        &inputs,
    )
    .expect("negative static matrix_power replay should serialize");

    assert!(result.supported, "{result:?}");
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        1.531907988153677,
    );
    assert_eq!(result.gradient.len(), 4);
    for (actual, expected) in result.gradient.iter().zip([
        -1.1688953673912945,
        1.0799781802752682,
        0.8941849296608454,
        -1.7466374393297168,
    ]) {
        assert_close(*actual, expected);
    }
}

#[test]
fn value_and_gradient_replays_static_matrix_power_zero_nodes() {
    let inputs = [1.5, 0.4, 0.2, 1.1];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &matrix_power_weighted_objective_ir(0, [1.0, 2.0, 3.0, 4.0]),
        &inputs,
    )
    .expect("zero static matrix_power replay should serialize");

    assert!(result.supported, "{result:?}");
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        5.0,
    );
    assert_eq!(result.gradient, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn value_and_gradient_fails_closed_on_singular_negative_matrix_power() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &matrix_power_weighted_objective_ir(-1, [1.0, 0.0, 0.0, 1.0]),
        &[1.0, 2.0, 2.0, 4.0],
    )
    .expect("unsupported negative matrix_power should serialize");

    assert!(!result.supported, "{result:?}");
    assert!(
        result
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("matrix_power requires a nonsingular matrix")),
        "{result:?}"
    );
}

fn matrix_power_weighted_objective_ir(exponent: i64, weights: [f64; 4]) -> String {
    let mut ssa_values = Vec::new();
    let mut effects = Vec::new();
    for index in 0..15 {
        ssa_values.push(json!({
            "name": format!("%{index}"),
            "producer": index,
            "version": 0,
            "shape": [],
            "dtype": "float64",
            "effect": index,
        }));
    }
    for index in 0..4 {
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
    for (offset, (row, column)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
        let index = 4 + offset;
        effects.push(json!({
            "index": index,
            "kind": "primitive",
            "target": format!("%{index}"),
            "inputs": ["%0", "%1", "%2", "%3"],
            "version": 0,
            "ordering": index,
            "operation": format!("linalg:matrix_power:2x2:power:{exponent}:{row}:{column}"),
        }));
    }
    for (offset, weight) in weights.into_iter().enumerate() {
        let index = 8 + offset;
        effects.push(json!({
            "index": index,
            "kind": "pure",
            "target": format!("%{index}"),
            "inputs": [format!("%{}", 4 + offset), weight.to_string()],
            "version": 0,
            "ordering": index,
            "operation": "mul",
        }));
    }
    effects.push(json!({
        "index": 12,
        "kind": "pure",
        "target": "%12",
        "inputs": ["%8", "%9"],
        "version": 0,
        "ordering": 12,
        "operation": "add",
    }));
    effects.push(json!({
        "index": 13,
        "kind": "pure",
        "target": "%13",
        "inputs": ["%10", "%11"],
        "version": 0,
        "ordering": 13,
        "operation": "add",
    }));
    effects.push(json!({
        "index": 14,
        "kind": "pure",
        "target": "%14",
        "inputs": ["%12", "%13"],
        "version": 0,
        "ordering": 14,
        "operation": "add",
    }));
    json!({
        "format": "program_ad_effect_ir.v1",
        "ssa_values": ssa_values,
        "effects": effects,
        "alias_edges": [],
        "control_regions": [],
        "phi_nodes": [],
        "bytecode_offsets": [0],
    })
    .to_string()
}

fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= 1.0e-12,
        "expected {expected}, got {actual}, delta {delta}"
    );
}
