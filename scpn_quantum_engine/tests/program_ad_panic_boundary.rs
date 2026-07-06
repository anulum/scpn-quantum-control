// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD panic-boundary tests

use std::panic::{catch_unwind, AssertUnwindSafe};

use scpn_quantum_engine::program_ad_ir::{
    interpret_program_ad_effect_ir_forward, interpret_program_ad_effect_ir_value_and_gradient,
};

const EMPTY_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [],
  "effects": [],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": []
}"#;

const SINGLE_PARAMETER_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0]
}"#;

const MISSING_OPCODE_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "pure", "target": "%1", "inputs": ["%0"], "version": 0, "ordering": 1}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2]
}"#;

const UNSAFE_ALIAS_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"}
  ],
  "alias_edges": [
    {"source": "mutable:list", "target": "%0", "kind": "mutation_alias", "version": 0}
  ],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0]
}"#;

const UNKNOWN_OPCODE_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "primitive", "target": "%1", "inputs": ["%0"], "version": 0, "ordering": 1, "operation": "python:opaque_call"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2]
}"#;

const MALFORMED_SIGNAL_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "primitive", "target": "%1", "inputs": ["%0"], "version": 0, "ordering": 1, "operation": "signal:convolve:left:1:right:0:mode:full:out:0"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2]
}"#;

const MALFORMED_CUMULATIVE_PROGRAM_AD_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "primitive", "target": "%1", "inputs": ["%0"], "version": 0, "ordering": 1, "operation": "cumsum:shape:0:axis:flat:out:0"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2]
}"#;

struct PanicBoundaryCase {
    name: &'static str,
    ir: &'static str,
    inputs: &'static [f64],
    parse_error_allowed: bool,
    expected_reason_fragment: Option<&'static str>,
}

#[test]
fn program_ad_forward_and_reverse_fail_closed_without_panicking() {
    let cases = [
        PanicBoundaryCase {
            name: "invalid JSON",
            ir: "{not-json",
            inputs: &[],
            parse_error_allowed: true,
            expected_reason_fragment: Some("invalid JSON"),
        },
        PanicBoundaryCase {
            name: "missing required payload lists",
            ir: r#"{"format": "program_ad_effect_ir.v1"}"#,
            inputs: &[],
            parse_error_allowed: true,
            expected_reason_fragment: Some("must be present"),
        },
        PanicBoundaryCase {
            name: "empty effects",
            ir: EMPTY_PROGRAM_AD_IR,
            inputs: &[],
            parse_error_allowed: false,
            expected_reason_fragment: Some("contains no effects"),
        },
        PanicBoundaryCase {
            name: "parameter count mismatch",
            ir: SINGLE_PARAMETER_PROGRAM_AD_IR,
            inputs: &[],
            parse_error_allowed: false,
            expected_reason_fragment: Some("parameter count"),
        },
        PanicBoundaryCase {
            name: "non-finite input",
            ir: SINGLE_PARAMETER_PROGRAM_AD_IR,
            inputs: &[f64::NAN],
            parse_error_allowed: false,
            expected_reason_fragment: Some("inputs must be finite"),
        },
        PanicBoundaryCase {
            name: "missing opcode metadata",
            ir: MISSING_OPCODE_PROGRAM_AD_IR,
            inputs: &[1.0],
            parse_error_allowed: false,
            expected_reason_fragment: Some("no opcode-bearing operation metadata"),
        },
        PanicBoundaryCase {
            name: "unsafe alias metadata",
            ir: UNSAFE_ALIAS_PROGRAM_AD_IR,
            inputs: &[1.0],
            parse_error_allowed: false,
            expected_reason_fragment: Some("non-view alias-bearing Program AD IR"),
        },
        PanicBoundaryCase {
            name: "unknown opcode",
            ir: UNKNOWN_OPCODE_PROGRAM_AD_IR,
            inputs: &[1.0],
            parse_error_allowed: false,
            expected_reason_fragment: None,
        },
        PanicBoundaryCase {
            name: "malformed signal metadata",
            ir: MALFORMED_SIGNAL_PROGRAM_AD_IR,
            inputs: &[1.0],
            parse_error_allowed: false,
            expected_reason_fragment: Some("size must be positive"),
        },
        PanicBoundaryCase {
            name: "malformed cumulative metadata",
            ir: MALFORMED_CUMULATIVE_PROGRAM_AD_IR,
            inputs: &[1.0],
            parse_error_allowed: false,
            expected_reason_fragment: Some("dimensions must be positive"),
        },
    ];

    for case in cases {
        assert_forward_fail_closed(case.name, case.ir, case.inputs, &case);
        assert_value_and_gradient_fail_closed(case.name, case.ir, case.inputs, &case);
    }
}

fn assert_forward_fail_closed(name: &str, ir: &str, inputs: &[f64], case: &PanicBoundaryCase) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        interpret_program_ad_effect_ir_forward(ir, inputs)
    }))
    .unwrap_or_else(|_| panic!("{name}: forward interpreter panicked"));

    match result {
        Ok(output) => {
            assert!(!output.supported, "{name}: forward unexpectedly supported");
            assert_fail_closed_reason(name, &output.blocked_reasons, case.expected_reason_fragment);
        }
        Err(reason) => {
            assert!(
                case.parse_error_allowed,
                "{name}: forward returned parse error after parsing should have succeeded: {reason}"
            );
            assert_error_reason(name, &reason, case.expected_reason_fragment);
        }
    }
}

fn assert_value_and_gradient_fail_closed(
    name: &str,
    ir: &str,
    inputs: &[f64],
    case: &PanicBoundaryCase,
) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        interpret_program_ad_effect_ir_value_and_gradient(ir, inputs)
    }))
    .unwrap_or_else(|_| panic!("{name}: value+gradient replay panicked"));

    match result {
        Ok(output) => {
            assert!(
                !output.supported,
                "{name}: value+gradient unexpectedly supported"
            );
            assert_fail_closed_reason(name, &output.blocked_reasons, case.expected_reason_fragment);
        }
        Err(reason) => {
            assert!(
                case.parse_error_allowed,
                "{name}: value+gradient returned parse error after parsing should have succeeded: {reason}"
            );
            assert_error_reason(name, &reason, case.expected_reason_fragment);
        }
    }
}

fn assert_fail_closed_reason(
    name: &str,
    reasons: &[String],
    expected_reason_fragment: Option<&str>,
) {
    assert!(!reasons.is_empty(), "{name}: expected fail-closed reason");
    if let Some(fragment) = expected_reason_fragment {
        assert!(
            reasons.iter().any(|reason| reason.contains(fragment)),
            "{name}: expected reason containing {fragment:?}, got {reasons:?}"
        );
    }
}

fn assert_error_reason(name: &str, reason: &str, expected_reason_fragment: Option<&str>) {
    assert!(!reason.is_empty(), "{name}: expected parse error reason");
    if let Some(fragment) = expected_reason_fragment {
        assert!(
            reason.contains(fragment),
            "{name}: expected parse error containing {fragment:?}, got {reason:?}"
        );
    }
}
