// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD linalg-array replay tests

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;

const MULTI_DOT_WEIGHTED_OBJECTIVE_IR: &str = r#"{
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
    {"name": "%11", "producer": 11, "version": 0, "shape": [], "dtype": "float64", "effect": 11},
    {"name": "%12", "producer": 12, "version": 0, "shape": [], "dtype": "float64", "effect": 12},
    {"name": "%13", "producer": 13, "version": 0, "shape": [], "dtype": "float64", "effect": 13},
    {"name": "%14", "producer": 14, "version": 0, "shape": [], "dtype": "float64", "effect": 14},
    {"name": "%15", "producer": 15, "version": 0, "shape": [], "dtype": "float64", "effect": 15},
    {"name": "%16", "producer": 16, "version": 0, "shape": [], "dtype": "float64", "effect": 16},
    {"name": "%17", "producer": 17, "version": 0, "shape": [], "dtype": "float64", "effect": 17},
    {"name": "%18", "producer": 18, "version": 0, "shape": [], "dtype": "float64", "effect": 18},
    {"name": "%19", "producer": 19, "version": 0, "shape": [], "dtype": "float64", "effect": 19},
    {"name": "%20", "producer": 20, "version": 0, "shape": [], "dtype": "float64", "effect": 20},
    {"name": "%21", "producer": 21, "version": 0, "shape": [], "dtype": "float64", "effect": 21},
    {"name": "%22", "producer": 22, "version": 0, "shape": [], "dtype": "float64", "effect": 22},
    {"name": "%23", "producer": 23, "version": 0, "shape": [], "dtype": "float64", "effect": 23},
    {"name": "%24", "producer": 24, "version": 0, "shape": [], "dtype": "float64", "effect": 24},
    {"name": "%25", "producer": 25, "version": 0, "shape": [], "dtype": "float64", "effect": 25},
    {"name": "%26", "producer": 26, "version": 0, "shape": [], "dtype": "float64", "effect": 26},
    {"name": "%27", "producer": 27, "version": 0, "shape": [], "dtype": "float64", "effect": 27},
    {"name": "%28", "producer": 28, "version": 0, "shape": [], "dtype": "float64", "effect": 28},
    {"name": "%29", "producer": 29, "version": 0, "shape": [], "dtype": "float64", "effect": 29}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["p0"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["p1"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["p2"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["p3"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "parameter", "target": "%4", "inputs": ["p4"], "version": 0, "ordering": 4, "operation": "parameter"},
    {"index": 5, "kind": "parameter", "target": "%5", "inputs": ["p5"], "version": 0, "ordering": 5, "operation": "parameter"},
    {"index": 6, "kind": "parameter", "target": "%6", "inputs": ["p6"], "version": 0, "ordering": 6, "operation": "parameter"},
    {"index": 7, "kind": "parameter", "target": "%7", "inputs": ["p7"], "version": 0, "ordering": 7, "operation": "parameter"},
    {"index": 8, "kind": "parameter", "target": "%8", "inputs": ["p8"], "version": 0, "ordering": 8, "operation": "parameter"},
    {"index": 9, "kind": "parameter", "target": "%9", "inputs": ["p9"], "version": 0, "ordering": 9, "operation": "parameter"},
    {"index": 10, "kind": "parameter", "target": "%10", "inputs": ["p10"], "version": 0, "ordering": 10, "operation": "parameter"},
    {"index": 11, "kind": "parameter", "target": "%11", "inputs": ["p11"], "version": 0, "ordering": 11, "operation": "parameter"},
    {"index": 12, "kind": "parameter", "target": "%12", "inputs": ["p12"], "version": 0, "ordering": 12, "operation": "parameter"},
    {"index": 13, "kind": "parameter", "target": "%13", "inputs": ["p13"], "version": 0, "ordering": 13, "operation": "parameter"},
    {"index": 14, "kind": "parameter", "target": "%14", "inputs": ["p14"], "version": 0, "ordering": 14, "operation": "parameter"},
    {"index": 15, "kind": "parameter", "target": "%15", "inputs": ["p15"], "version": 0, "ordering": 15, "operation": "parameter"},
    {"index": 16, "kind": "primitive", "target": "%16", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5", "%6", "%7", "%8", "%9", "%10", "%11"], "version": 0, "ordering": 16, "operation": "linalg:multi_dot:2x2__2x2__2x2:out:2x2:0"},
    {"index": 17, "kind": "primitive", "target": "%17", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5", "%6", "%7", "%8", "%9", "%10", "%11"], "version": 0, "ordering": 17, "operation": "linalg:multi_dot:2x2__2x2__2x2:out:2x2:1"},
    {"index": 18, "kind": "primitive", "target": "%18", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5", "%6", "%7", "%8", "%9", "%10", "%11"], "version": 0, "ordering": 18, "operation": "linalg:multi_dot:2x2__2x2__2x2:out:2x2:2"},
    {"index": 19, "kind": "primitive", "target": "%19", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5", "%6", "%7", "%8", "%9", "%10", "%11"], "version": 0, "ordering": 19, "operation": "linalg:multi_dot:2x2__2x2__2x2:out:2x2:3"},
    {"index": 20, "kind": "primitive", "target": "%20", "inputs": ["%12", "%13", "%4", "%5", "%6", "%7", "%14", "%15"], "version": 0, "ordering": 20, "operation": "linalg:multi_dot:2__2x2__2:out:scalar"},
    {"index": 21, "kind": "pure", "target": "%21", "inputs": ["%16", "1.0"], "version": 0, "ordering": 21, "operation": "mul"},
    {"index": 22, "kind": "pure", "target": "%22", "inputs": ["%17", "-2.0"], "version": 0, "ordering": 22, "operation": "mul"},
    {"index": 23, "kind": "pure", "target": "%23", "inputs": ["%18", "0.5"], "version": 0, "ordering": 23, "operation": "mul"},
    {"index": 24, "kind": "pure", "target": "%24", "inputs": ["%19", "1.5"], "version": 0, "ordering": 24, "operation": "mul"},
    {"index": 25, "kind": "pure", "target": "%25", "inputs": ["%21", "%22"], "version": 0, "ordering": 25, "operation": "add"},
    {"index": 26, "kind": "pure", "target": "%26", "inputs": ["%23", "%24"], "version": 0, "ordering": 26, "operation": "add"},
    {"index": 27, "kind": "pure", "target": "%27", "inputs": ["%25", "%26"], "version": 0, "ordering": 27, "operation": "add"},
    {"index": 28, "kind": "pure", "target": "%28", "inputs": ["%20", "0.75"], "version": 0, "ordering": 28, "operation": "mul"},
    {"index": 29, "kind": "pure", "target": "%29", "inputs": ["%27", "%28"], "version": 0, "ordering": 29, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

const MULTI_DOT_INPUT_COUNT_MISMATCH_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["a"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["b"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "primitive", "target": "%2", "inputs": ["%0", "%1"], "version": 0, "ordering": 2, "operation": "linalg:multi_dot:2x2__2x2:out:2x2:0"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0]
}"#;

#[test]
fn value_and_gradient_replays_static_multi_dot_array_nodes() {
    let inputs = [
        1.0, 2.0, 3.0, 5.0, 0.5, -1.0, 2.0, 1.5, 2.0, 0.25, -0.5, 3.0, 1.25, -0.75, 0.5, 2.5,
    ];
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(MULTI_DOT_WEIGHTED_OBJECTIVE_IR, &inputs)
            .expect("static multi_dot replay should serialize");
    assert!(result.supported, "{result:?}");
    assert!(result
        .claim_boundary
        .contains("value_and_gradient_static_linalg_primitives"));
    assert!(result
        .claim_boundary
        .contains("executed_branch_view_assignment"));
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        23.90625,
    );
    assert_eq!(result.gradient.len(), 16);
    for (actual, expected) in result.gradient.iter().zip([
        7.25, -6.75, -3.5625, 9.125, 6.09375, 8.59375, 9.59375, 6.84375, 10.25, 8.25, 4.25, 2.75,
        -1.6875, 3.5625, -0.65625, -1.78125,
    ]) {
        assert_close(*actual, expected);
    }
}

#[test]
fn value_and_gradient_fails_closed_on_multi_dot_input_count_mismatch() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        MULTI_DOT_INPUT_COUNT_MISMATCH_IR,
        &[1.0, 2.0],
    )
    .expect("unsupported static multi_dot should serialize");
    assert!(!result.supported, "{result:?}");
    assert!(
        result
            .blocked_reasons
            .iter()
            .any(|reason| reason
                .contains("multi_dot input count must match flattened operand shapes")),
        "{result:?}"
    );
}

fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= 1.0e-12,
        "expected {expected}, got {actual}, delta {delta}"
    );
}
