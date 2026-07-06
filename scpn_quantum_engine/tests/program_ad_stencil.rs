// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD stencil replay tests

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;

const STENCIL_WEIGHTED_OBJECTIVE_IR: &str = r#"{
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
    {"name": "%23", "producer": 23, "version": 0, "shape": [], "dtype": "float64", "effect": 23}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x0"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["x1"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "parameter", "target": "%2", "inputs": ["x2"], "version": 0, "ordering": 2, "operation": "parameter"},
    {"index": 3, "kind": "parameter", "target": "%3", "inputs": ["x3"], "version": 0, "ordering": 3, "operation": "parameter"},
    {"index": 4, "kind": "parameter", "target": "%4", "inputs": ["x4"], "version": 0, "ordering": 4, "operation": "parameter"},
    {"index": 5, "kind": "parameter", "target": "%5", "inputs": ["w0"], "version": 0, "ordering": 5, "operation": "parameter"},
    {"index": 6, "kind": "parameter", "target": "%6", "inputs": ["w1"], "version": 0, "ordering": 6, "operation": "parameter"},
    {"index": 7, "kind": "parameter", "target": "%7", "inputs": ["w2"], "version": 0, "ordering": 7, "operation": "parameter"},
    {"index": 8, "kind": "parameter", "target": "%8", "inputs": ["w3"], "version": 0, "ordering": 8, "operation": "parameter"},
    {"index": 9, "kind": "parameter", "target": "%9", "inputs": ["w4"], "version": 0, "ordering": 9, "operation": "parameter"},
    {"index": 10, "kind": "primitive", "target": "%10", "inputs": ["%0", "%1", "%2", "%3", "%4"], "version": 0, "ordering": 10, "operation": "stencil:gradient:shape:5:axis:0:edge:2:spacing:coordinates=0,0.5,1.5,3,5:out:0"},
    {"index": 11, "kind": "primitive", "target": "%11", "inputs": ["%0", "%1", "%2", "%3", "%4"], "version": 0, "ordering": 11, "operation": "stencil:gradient:shape:5:axis:0:edge:2:spacing:coordinates=0,0.5,1.5,3,5:out:1"},
    {"index": 12, "kind": "primitive", "target": "%12", "inputs": ["%0", "%1", "%2", "%3", "%4"], "version": 0, "ordering": 12, "operation": "stencil:gradient:shape:5:axis:0:edge:2:spacing:coordinates=0,0.5,1.5,3,5:out:2"},
    {"index": 13, "kind": "primitive", "target": "%13", "inputs": ["%0", "%1", "%2", "%3", "%4"], "version": 0, "ordering": 13, "operation": "stencil:gradient:shape:5:axis:0:edge:2:spacing:coordinates=0,0.5,1.5,3,5:out:3"},
    {"index": 14, "kind": "primitive", "target": "%14", "inputs": ["%0", "%1", "%2", "%3", "%4"], "version": 0, "ordering": 14, "operation": "stencil:gradient:shape:5:axis:0:edge:2:spacing:coordinates=0,0.5,1.5,3,5:out:4"},
    {"index": 15, "kind": "pure", "target": "%15", "inputs": ["%10", "%5"], "version": 0, "ordering": 15, "operation": "mul"},
    {"index": 16, "kind": "pure", "target": "%16", "inputs": ["%11", "%6"], "version": 0, "ordering": 16, "operation": "mul"},
    {"index": 17, "kind": "pure", "target": "%17", "inputs": ["%12", "%7"], "version": 0, "ordering": 17, "operation": "mul"},
    {"index": 18, "kind": "pure", "target": "%18", "inputs": ["%13", "%8"], "version": 0, "ordering": 18, "operation": "mul"},
    {"index": 19, "kind": "pure", "target": "%19", "inputs": ["%14", "%9"], "version": 0, "ordering": 19, "operation": "mul"},
    {"index": 20, "kind": "pure", "target": "%20", "inputs": ["%15", "%16"], "version": 0, "ordering": 20, "operation": "add"},
    {"index": 21, "kind": "pure", "target": "%21", "inputs": ["%20", "%17"], "version": 0, "ordering": 21, "operation": "add"},
    {"index": 22, "kind": "pure", "target": "%22", "inputs": ["%21", "%18"], "version": 0, "ordering": 22, "operation": "add"},
    {"index": 23, "kind": "pure", "target": "%23", "inputs": ["%22", "%19"], "version": 0, "ordering": 23, "operation": "add"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;

#[test]
fn value_and_gradient_replays_static_gradient_nodes() {
    let inputs = [1.0, -2.0, 0.5, 3.0, -1.5, 0.5, -1.0, 0.25, 2.0, -0.75];
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(STENCIL_WEIGHTED_OBJECTIVE_IR, &inputs)
            .expect("static stencil replay should serialize");

    assert!(result.supported, "{result:?}");
    assert!(result.claim_boundary.contains("static_stencil_primitives"));
    assert_close(
        result
            .value
            .expect("supported replay should return a value"),
        2.633928571428572,
    );
    assert_eq!(result.gradient.len(), 10);
    for (actual, expected) in result.gradient.iter().zip([
        0.0,
        0.35,
        -1.4642857142857142,
        1.275,
        -0.16071428571428573,
        -8.833333333333332,
        -3.1666666666666665,
        2.1666666666666665,
        -0.011904761904761862,
        -4.488095238095238,
    ]) {
        assert_close(*actual, expected);
    }
}

#[test]
fn value_and_gradient_rejects_non_monotonic_static_gradient_spacing() {
    let malformed = STENCIL_WEIGHTED_OBJECTIVE_IR.replace(
        "spacing:coordinates=0,0.5,1.5,3,5",
        "spacing:coordinates=0,0.5,0.5,3,5",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &malformed,
        &[1.0, -2.0, 0.5, 3.0, -1.5, 0.5, -1.0, 0.25, 2.0, -0.75],
    )
    .expect("malformed static stencil replay should serialize");

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("strictly monotonic")));
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "expected {expected}, got {actual}",
    );
}
