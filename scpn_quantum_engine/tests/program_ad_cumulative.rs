// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD cumulative replay tests

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;

const CUMULATIVE_WEIGHTED_OBJECTIVE_IR: &str = r#"{
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
    {"name": "%20", "producer": 20, "version": 0, "shape": [], "dtype": "float64", "effect": 20}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x0"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["x1"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["x2"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["x3"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "parameter", "target": "%4", "inputs": ["x4"], "version": 0, "ordering": 4, "operation": "parameter"},
    {"index": 5, "kind": "parameter", "target": "%5", "inputs": ["x5"], "version": 0, "ordering": 5, "operation": "parameter"},
    {"index": 6, "kind": "parameter", "target": "%6", "inputs": ["w0"], "version": 0, "ordering": 6, "operation": "parameter"},
    {"index": 7, "kind": "parameter", "target": "%7", "inputs": ["w1"], "version": 0, "ordering": 7, "operation": "parameter"},
    {"index": 8, "kind": "parameter", "target": "%8", "inputs": ["w2"], "version": 0, "ordering": 8, "operation": "parameter"},
    {"index": 9, "kind": "parameter", "target": "%9", "inputs": ["w3"], "version": 0, "ordering": 9, "operation": "parameter"},
    {"index": 10, "kind": "primitive", "target": "%10", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5"], "version": 0, "ordering": 10, "operation": "cumsum:shape:2x3:axis:1:out:4"},
    {"index": 11, "kind": "primitive", "target": "%11", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5"], "version": 0, "ordering": 11, "operation": "cumprod:shape:2x3:axis:1:out:5"},
    {"index": 12, "kind": "primitive", "target": "%12", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5"], "version": 0, "ordering": 12, "operation": "diff:shape:2x3:n:2:axis:1:out:1"},
    {"index": 13, "kind": "primitive", "target": "%13", "inputs": ["%0", "%1", "%2", "%3", "%4", "%5"], "version": 0, "ordering": 13, "operation": "cumsum:shape:2x3:axis:flat:out:3"},
    {"index": 14, "kind": "pure", "target": "%14", "inputs": ["%10", "%6"], "version": 0, "ordering": 14, "operation": "mul"},
    {"index": 15, "kind": "pure", "target": "%15", "inputs": ["%11", "%7"], "version": 0, "ordering": 15, "operation": "mul"},
    {"index": 16, "kind": "pure", "target": "%16", "inputs": ["%12", "%8"], "version": 0, "ordering": 16, "operation": "mul"},
    {"index": 17, "kind": "pure", "target": "%17", "inputs": ["%13", "%9"], "version": 0, "ordering": 17, "operation": "mul"},
    {"index": 18, "kind": "pure", "target": "%18", "inputs": ["%14", "%15"], "version": 0, "ordering": 18, "operation": "add"},
    {"index": 19, "kind": "pure", "target": "%19", "inputs": ["%18", "%16"], "version": 0, "ordering": 19, "operation": "add"},
    {"index": 20, "kind": "pure", "target": "%20", "inputs": ["%19", "%17"], "version": 0, "ordering": 20, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

#[test]
fn value_and_gradient_replays_static_cumulative_nodes() {
    let inputs = [1.25, -0.75, 2.0, 0.5, 1.5, -1.25, 0.2, -0.4, 0.6, -0.8];
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        CUMULATIVE_WEIGHTED_OBJECTIVE_IR,
        &inputs,
    )
    .expect("static cumulative replay should serialize");

    assert!(result.supported, "{result:?}");
    assert!(result
        .claim_boundary
        .contains("static_cumulative_primitives"));
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        -3.875,
    );
    assert_eq!(result.gradient.len(), 10);
    for (actual, expected) in result
        .gradient
        .iter()
        .zip([-0.8, -0.8, -0.8, 0.75, -0.75, 0.3, 2.0, -0.9375, -3.75, 3.0])
    {
        assert_close(*actual, expected);
    }
}

#[test]
fn value_and_gradient_rejects_static_cumulative_diff_order_outside_axis() {
    let malformed = CUMULATIVE_WEIGHTED_OBJECTIVE_IR.replace(
        "diff:shape:2x3:n:2:axis:1:out:1",
        "diff:shape:2x3:n:4:axis:1:out:1",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &malformed,
        &[1.25, -0.75, 2.0, 0.5, 1.5, -1.25, 0.2, -0.4, 0.6, -0.8],
    )
    .expect("malformed static cumulative replay should serialize");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("exceeds axis length")));
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "expected {expected}, got {actual}",
    );
}
