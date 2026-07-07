// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD dynamic-boundary tests

use scpn_quantum_engine::program_ad_ir::{
    interpret_program_ad_effect_ir_forward, interpret_program_ad_effect_ir_value_and_gradient,
};

const DYNAMIC_BOUNDARY_CLAIM_FRAGMENT: &str = "dynamic_boundary_fail_closed_audit";

struct DynamicBoundaryCase {
    name: &'static str,
    operation: &'static str,
    source_shape: &'static str,
    target_shape: &'static str,
    inputs: &'static [f64],
    reason_fragment: &'static str,
}

#[test]
fn program_ad_rust_claim_boundary_reports_dynamic_boundary_audit() {
    let ir = single_source_program_ad_ir("sin", "[]", "[]");
    let forward = interpret_program_ad_effect_ir_forward(&ir, &[0.25]).unwrap();
    let reverse = interpret_program_ad_effect_ir_value_and_gradient(&ir, &[0.25]).unwrap();

    assert!(forward.supported, "{:?}", forward.blocked_reasons);
    assert!(reverse.supported, "{:?}", reverse.blocked_reasons);
    assert!(
        forward
            .claim_boundary
            .contains(DYNAMIC_BOUNDARY_CLAIM_FRAGMENT),
        "{}",
        forward.claim_boundary
    );
    assert!(
        reverse
            .claim_boundary
            .contains(DYNAMIC_BOUNDARY_CLAIM_FRAGMENT),
        "{}",
        reverse.claim_boundary
    );
}

#[test]
fn program_ad_value_and_gradient_rejects_dynamic_boundary_metadata() {
    let cases = [
        DynamicBoundaryCase {
            name: "dynamic source-map index",
            operation: "index_map:s%0,c1.0",
            source_shape: "[2]",
            target_shape: "[2]",
            inputs: &[1.0, 2.0],
            reason_fragment: "source token",
        },
        DynamicBoundaryCase {
            name: "dynamic reduction axis",
            operation: "sum:axis:%0",
            source_shape: "[2,2]",
            target_shape: "[2]",
            inputs: &[1.0, 2.0, 3.0, 4.0],
            reason_fragment: "sum axis metadata must be an integer",
        },
        DynamicBoundaryCase {
            name: "dynamic variance ddof",
            operation: "var:ddof:%1",
            source_shape: "[3]",
            target_shape: "[]",
            inputs: &[1.0, 2.0, 4.0],
            reason_fragment: "var correction metadata must be a finite non-negative scalar",
        },
        DynamicBoundaryCase {
            name: "dynamic standard-deviation correction",
            operation: "std:axis:0:correction:%2",
            source_shape: "[2,2]",
            target_shape: "[2]",
            inputs: &[1.0, 3.0, 2.0, 5.0],
            reason_fragment: "std correction metadata must be a finite non-negative scalar",
        },
        DynamicBoundaryCase {
            name: "dynamic quantile q",
            operation: "quantile:q:%0",
            source_shape: "[3]",
            target_shape: "[]",
            inputs: &[1.0, 2.0, 4.0],
            reason_fragment: "quantile q metadata must be a finite float",
        },
        DynamicBoundaryCase {
            name: "unsupported quantile method",
            operation: "quantile:q:0.5:method:nearest",
            source_shape: "[3]",
            target_shape: "[]",
            inputs: &[1.0, 2.0, 4.0],
            reason_fragment: "quantile operation metadata field method is unsupported",
        },
        DynamicBoundaryCase {
            name: "dynamic percentile axis",
            operation: "percentile:axis:%0:q:50.0",
            source_shape: "[2,2]",
            target_shape: "[2]",
            inputs: &[1.0, 2.0, 3.0, 4.0],
            reason_fragment: "percentile axis metadata must be an integer",
        },
        DynamicBoundaryCase {
            name: "dynamic trapezoid axis",
            operation: "trapezoid:axis:%0:dx:1.0",
            source_shape: "[3]",
            target_shape: "[]",
            inputs: &[1.0, 2.0, 4.0],
            reason_fragment: "trapezoid axis metadata must be an integer",
        },
        DynamicBoundaryCase {
            name: "dynamic trapezoid dx",
            operation: "trapezoid:axis:0:dx:%0",
            source_shape: "[3]",
            target_shape: "[]",
            inputs: &[1.0, 2.0, 4.0],
            reason_fragment: "trapezoid dx metadata must be a finite float",
        },
        DynamicBoundaryCase {
            name: "dynamic trapezoid axis grid",
            operation: "trapezoid:axis:0:x:%0",
            source_shape: "[3]",
            target_shape: "[]",
            inputs: &[1.0, 2.0, 4.0],
            reason_fragment: "trapezoid x metadata must be a finite float",
        },
        DynamicBoundaryCase {
            name: "dynamic trapezoid full grid",
            operation: "trapezoid:axis:0:xfull:%0",
            source_shape: "[3]",
            target_shape: "[]",
            inputs: &[1.0, 2.0, 4.0],
            reason_fragment: "trapezoid xfull metadata must be a finite float",
        },
        DynamicBoundaryCase {
            name: "zero-variance std gradient",
            operation: "std",
            source_shape: "[3]",
            target_shape: "[]",
            inputs: &[2.0, 2.0, 2.0],
            reason_fragment: "std gradient requires positive variance",
        },
    ];

    for case in cases {
        assert_dynamic_boundary_fail_closed(&case);
    }
}

fn assert_dynamic_boundary_fail_closed(case: &DynamicBoundaryCase) {
    let ir = single_source_program_ad_ir(case.operation, case.source_shape, case.target_shape);
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, case.inputs)
        .unwrap_or_else(|reason| panic!("{} parse failed: {reason}", case.name));

    assert!(!result.supported, "{} unexpectedly supported", case.name);
    assert_eq!(result.value, None, "{} returned a value", case.name);
    assert!(
        result.gradient.is_empty(),
        "{} returned a gradient {:?}",
        case.name,
        result.gradient
    );
    assert_eq!(result.effect_count, 2, "{}", case.name);
    assert!(
        result.supported_effect_count <= result.effect_count,
        "{} reported impossible effect counts",
        case.name
    );
    assert_eq!(
        result.blocked_reasons.len(),
        1,
        "{} should return one typed fail-closed reason",
        case.name
    );
    assert!(
        result.blocked_reasons[0].contains(case.reason_fragment),
        "{} reason {:?} did not contain {:?}",
        case.name,
        result.blocked_reasons[0],
        case.reason_fragment
    );
}

fn single_source_program_ad_ir(operation: &str, source_shape: &str, target_shape: &str) -> String {
    format!(
        r#"{{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {{"name": "%0", "producer": 0, "version": 0, "shape": {source_shape}, "dtype": "float64", "effect": 0}},
    {{"name": "%1", "producer": 1, "version": 0, "shape": {target_shape}, "dtype": "float64", "effect": 1}}
  ],
  "effects": [
    {{"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"}},
    {{"index": 1, "kind": "primitive", "target": "%1", "inputs": ["%0"], "version": 0, "ordering": 1, "operation": "{operation}"}}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2]
}}"#
    )
}
