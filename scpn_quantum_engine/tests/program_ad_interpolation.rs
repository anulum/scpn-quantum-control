// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD interpolation replay tests

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;

const INTERPOLATION_WEIGHTED_OBJECTIVE_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4},
    {"name": "%5", "producer": 5, "version": 0, "shape": [], "dtype": "float64", "effect": 5},
    {"name": "%6", "producer": 6, "version": 0, "shape": [], "dtype": "float64", "effect": 6},
    {"name": "%7", "producer": 7, "version": 0, "shape": [], "dtype": "float64", "effect": 7},
    {"name": "%8", "producer": 8, "version": 0, "shape": [], "dtype": "float64", "effect": 8},
    {"name": "%9", "producer": 9, "version": 0, "shape": [], "dtype": "float64", "effect": 9},
    {"name": "%10", "producer": 10, "version": 0, "shape": [], "dtype": "float64", "effect": 10},
    {"name": "%11", "producer": 11, "version": 0, "shape": [], "dtype": "float64", "effect": 11}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["sample0"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["sample1"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["fp0"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["fp1"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "parameter", "target": "%4", "inputs": ["fp2"], "version": 0, "ordering": 4, "operation": "parameter"},
    {"index": 5, "kind": "parameter", "target": "%5", "inputs": ["weight0"], "version": 0, "ordering": 5, "operation": "parameter"},
    {"index": 6, "kind": "parameter", "target": "%6", "inputs": ["weight1"], "version": 0, "ordering": 6, "operation": "parameter"},
    {"index": 7, "kind": "primitive", "target": "%7", "inputs": ["%0", "%1", "%2", "%3", "%4"], "version": 0, "ordering": 7, "operation": "interpolation:interp:samples:2:grid:0,2,4:left:none:right:none:out:0"},
    {"index": 8, "kind": "primitive", "target": "%8", "inputs": ["%0", "%1", "%2", "%3", "%4"], "version": 0, "ordering": 8, "operation": "interpolation:interp:samples:2:grid:0,2,4:left:-1.25:right:8.5:out:1"},
    {"index": 9, "kind": "pure", "target": "%9", "inputs": ["%7", "%5"], "version": 0, "ordering": 9, "operation": "mul"},
    {"index": 10, "kind": "pure", "target": "%10", "inputs": ["%8", "%6"], "version": 0, "ordering": 10, "operation": "mul"},
    {"index": 11, "kind": "pure", "target": "%11", "inputs": ["%9", "%10"], "version": 0, "ordering": 11, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

#[test]
fn value_and_gradient_replays_static_grid_interpolation_nodes() {
    let inputs = [1.0, 5.0, 2.0, 6.0, 10.0, 3.0, -0.5];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        INTERPOLATION_WEIGHTED_OBJECTIVE_IR,
        &inputs,
    )
    .expect("static interpolation replay should serialize");

    assert!(result.supported, "{result:?}");
    assert!(result
        .claim_boundary
        .contains("static_interpolation_primitives"));
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        7.75,
    );
    assert_eq!(
        result.parameter_targets,
        vec!["%0", "%1", "%2", "%3", "%4", "%5", "%6"]
    );
    assert_eq!(result.gradient.len(), 7);
    for (actual, expected) in result
        .gradient
        .iter()
        .zip([6.0, 0.0, 1.5, 1.5, 0.0, 4.0, 8.5])
    {
        assert_close(*actual, expected);
    }
}

#[test]
fn value_and_gradient_rejects_non_increasing_static_grid() {
    let malformed_grid = INTERPOLATION_WEIGHTED_OBJECTIVE_IR.replace("grid:0,2,4", "grid:0,0,4");
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &malformed_grid,
        &[1.0, 5.0, 2.0, 6.0, 10.0, 3.0, -0.5],
    )
    .expect("malformed interpolation replay should serialize");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("strictly increasing")));
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "expected {expected}, got {actual}",
    );
}
