// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD solve replay tests

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use serde_json::json;

#[test]
fn value_and_gradient_replays_static_solve_matrix_rhs_nodes() {
    let inputs = [
        4.0, 0.5, -0.25, 0.2, 3.5, 0.75, -0.1, 0.4, 2.8, 1.0, -0.5, 0.25, 2.0, -1.0, 0.3, 1.5,
        -0.75, 0.5, 2.25, -1.2, 0.4, 1.75, -0.6, 0.8,
    ];
    let weights = [
        0.2, -0.7, 1.1, 0.4, -0.3, 0.5, 0.9, -0.2, 1.3, -0.8, -1.0, 0.6, 0.75, -0.45, 0.95,
    ];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &solve_matrix_rhs_weighted_objective_ir(3, 5, &weights),
        &inputs,
    )
    .expect("static solve matrix-RHS replay should serialize");

    assert!(result.supported, "{result:?}");
    assert!(result
        .claim_boundary
        .contains("value_and_gradient_static_linalg_primitives_executed_branch_view_assignment"));
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        2.1429973386560213,
    );
    let expected_gradient = [
        -0.132860271915694,
        0.20000141876371452,
        -0.14443084113432042,
        -0.2363074914708175,
        -0.09296061901479809,
        0.29084352917584655,
        0.3017964957200259,
        -0.07259103438720252,
        -0.5478419962027598,
        0.030671989354624084,
        -0.1853226879574185,
        0.29007318695941453,
        0.0740119760479042,
        -0.05145708582834331,
        0.18463073852295409,
        0.2692614770459082,
        -0.13632734530938123,
        0.3904191616766467,
        -0.26766467065868266,
        -0.4038589487691284,
        0.12561543579507653,
        0.3302727877578177,
        -0.25868263473053893,
        0.40638722554890216,
    ];
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert_close(*actual, expected);
    }
}

#[test]
fn value_and_gradient_replays_static_solve_vector_rhs_nodes() {
    let inputs = [2.0, 1.0, 1.0, 3.0, 1.0, 2.0];
    let weights = [0.5, -1.25];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &solve_vector_rhs_weighted_objective_ir(2, &weights),
        &inputs,
    )
    .expect("static solve vector-RHS replay should serialize");

    assert!(result.supported, "{result:?}");
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        -0.65,
    );
    for (actual, expected) in result
        .gradient
        .iter()
        .zip([-0.11, -0.33, 0.12, 0.36, 0.55, -0.6])
    {
        assert_close(*actual, expected);
    }
}

#[test]
fn value_and_gradient_fails_closed_on_solve_matrix_rhs_column_out_of_range() {
    let inputs = [
        4.0, 0.5, -0.25, 0.2, 3.5, 0.75, -0.1, 0.4, 2.8, 1.0, -0.5, 0.25, 2.0, -1.0, 0.3, 1.5,
        -0.75, 0.5, 2.25, -1.2, 0.4, 1.75, -0.6, 0.8,
    ];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &single_solve_output_ir(24, "linalg:solve:3x3:rhs:3x5:0:5"),
        &inputs,
    )
    .expect("out-of-range solve metadata should serialize");

    assert!(!result.supported, "{result:?}");
    let error = result.blocked_reasons.join("; ");
    assert!(error.contains("has no solution index"), "{error}");
}

#[test]
fn value_and_gradient_fails_closed_on_solve_matrix_rhs_shape_mismatch() {
    let inputs = [
        4.0, 0.5, -0.25, 0.2, 3.5, 0.75, -0.1, 0.4, 2.8, 1.0, -0.5, 0.25,
    ];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &single_solve_output_ir(12, "linalg:solve:3x3:rhs:2x1:0:0"),
        &inputs,
    )
    .expect("mismatched solve metadata should serialize");

    assert!(!result.supported, "{result:?}");
    let error = result.blocked_reasons.join("; ");
    assert!(error.contains("has no solution index"), "{error}");
}

fn solve_matrix_rhs_weighted_objective_ir(n: usize, rhs_columns: usize, weights: &[f64]) -> String {
    let input_count = n * n + n * rhs_columns;
    let operations = (0..n)
        .flat_map(|row| {
            (0..rhs_columns).map(move |column| {
                format!("linalg:solve:{n}x{n}:rhs:{n}x{rhs_columns}:{row}:{column}")
            })
        })
        .collect::<Vec<String>>();
    weighted_solve_objective_ir(input_count, &operations, weights)
}

fn solve_vector_rhs_weighted_objective_ir(n: usize, weights: &[f64]) -> String {
    let input_count = n * n + n;
    let operations = (0..n)
        .map(|row| format!("linalg:solve:{n}x{n}:rhs:{n}:{row}"))
        .collect::<Vec<String>>();
    weighted_solve_objective_ir(input_count, &operations, weights)
}

fn weighted_solve_objective_ir(
    input_count: usize,
    operations: &[String],
    weights: &[f64],
) -> String {
    assert_eq!(operations.len(), weights.len());
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
    let input_targets = (0..input_count)
        .map(|index| format!("%{index}"))
        .collect::<Vec<String>>();
    for (operation, weight) in operations.iter().zip(weights.iter()) {
        let solve_index = next_index;
        ssa_values.push(json!({
            "name": format!("%{solve_index}"),
            "producer": solve_index,
            "version": 0,
            "shape": [],
            "dtype": "float64",
            "effect": solve_index,
        }));
        effects.push(json!({
            "index": solve_index,
            "kind": "primitive",
            "target": format!("%{solve_index}"),
            "inputs": input_targets,
            "version": 0,
            "ordering": solve_index,
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
            "inputs": [format!("%{solve_index}"), weight.to_string()],
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

fn single_solve_output_ir(input_count: usize, operation: &str) -> String {
    weighted_solve_objective_ir(input_count, &[operation.to_owned()], &[1.0])
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
