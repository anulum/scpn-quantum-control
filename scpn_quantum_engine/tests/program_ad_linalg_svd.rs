// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD SVD replay tests

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use serde_json::json;

#[test]
fn value_and_gradient_replays_static_svdvals_3x2_nodes() {
    let inputs = [3.0, 0.5, -1.0, 2.0, 0.25, -0.75];
    let weights = [0.7, -1.2];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &svdvals_weighted_objective_ir(3, 2, &weights),
        &inputs,
    )
    .expect("static 3x2 svdvals replay should serialize");

    assert!(result.supported, "{result:?}");
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        -0.37791324163266937,
    );
    assert_gradient_close(
        &result.gradient,
        &[
            0.5724583434451499,
            -0.5630823936091282,
            -0.4026628062062343,
            -0.9812694240894069,
            0.12510916598131058,
            0.3801977662484035,
        ],
    );
}

#[test]
fn value_and_gradient_replays_static_svdvals_2x3_nodes() {
    let inputs = [2.0, -0.5, 1.25, 0.75, 3.0, -1.5];
    let weights = [-0.4, 1.1];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &svdvals_weighted_objective_ir(2, 3, &weights),
        &inputs,
    )
    .expect("static 2x3 svdvals replay should serialize");

    assert!(result.supported, "{result:?}");
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        1.1206358269939978,
    );
    assert_gradient_close(
        &result.gradient,
        &[
            0.9856136813083707,
            0.2543029864165801,
            0.3067487113584044,
            0.2625914082650306,
            -0.2848514515896717,
            0.2995100886948116,
        ],
    );
}

#[test]
fn value_and_gradient_rejects_svdvals_output_index_out_of_range() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &single_svdvals_output_ir(3, 2, "linalg:svdvals:3x2:2"),
        &[3.0, 0.5, -1.0, 2.0, 0.25, -0.75],
    )
    .expect("out-of-range svdvals metadata should serialize");

    assert!(!result.supported, "{result:?}");
    let error = result.blocked_reasons.join("; ");
    assert!(
        error.contains("svdvals output index is outside the singular-value spectrum"),
        "{error}"
    );
}

#[test]
fn value_and_gradient_rejects_rank_deficient_rectangular_svdvals() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &single_svdvals_output_ir(3, 2, "linalg:svdvals:3x2:0"),
        &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0],
    )
    .expect("rank-deficient svdvals metadata should serialize");

    assert!(!result.supported, "{result:?}");
    let error = result.blocked_reasons.join("; ");
    assert!(
        error.contains("svdvals gradient requires positive singular values"),
        "{error}"
    );
}

fn svdvals_weighted_objective_ir(rows: usize, cols: usize, weights: &[f64]) -> String {
    let operations = (0..weights.len())
        .map(|index| format!("linalg:svdvals:{rows}x{cols}:{index}"))
        .collect::<Vec<String>>();
    weighted_svdvals_objective_ir(rows * cols, &operations, weights)
}

fn single_svdvals_output_ir(rows: usize, cols: usize, operation: &str) -> String {
    weighted_svdvals_objective_ir(rows * cols, &[operation.to_owned()], &[1.0])
}

fn weighted_svdvals_objective_ir(
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

    let input_targets = (0..input_count)
        .map(|index| format!("%{index}"))
        .collect::<Vec<String>>();
    let mut weighted_targets = Vec::new();
    let mut next_index = input_count;
    for (operation, weight) in operations.iter().zip(weights.iter()) {
        let svd_index = next_index;
        ssa_values.push(json!({
            "name": format!("%{svd_index}"),
            "producer": svd_index,
            "version": 0,
            "shape": [],
            "dtype": "float64",
            "effect": svd_index,
        }));
        effects.push(json!({
            "index": svd_index,
            "kind": "primitive",
            "target": format!("%{svd_index}"),
            "inputs": input_targets,
            "version": 0,
            "ordering": svd_index,
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
            "inputs": [format!("%{svd_index}"), weight.to_string()],
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

fn assert_gradient_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (observed, reference) in actual.iter().zip(expected.iter()) {
        assert_close(*observed, *reference);
    }
}

fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= 1.0e-12,
        "expected {expected}, got {actual}, delta {delta}"
    );
}
