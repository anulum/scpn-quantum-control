// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Public static source-map replay contracts

use scpn_quantum_engine::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
    replay_checkpoint, with_replay_checkpoint,
};
use std::cell::Cell;
use std::rc::Rc;

const SOURCE_MAP_IR: &str = r#"{
  "format": "program_ad_effect_ir.v1",
  "ssa_values": [
    {"name": "%0", "producer": 0, "version": 0, "shape": [4], "dtype": "float64", "effect": 0},
    {"name": "%1", "producer": 1, "version": 0, "shape": [6], "dtype": "float64", "effect": 1},
    {"name": "%2", "producer": 2, "version": 0, "shape": [6], "dtype": "float64", "effect": 2},
    {"name": "%3", "producer": 3, "version": 0, "shape": [6], "dtype": "float64", "effect": 3},
    {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4}
  ],
  "effects": [
    {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["source"], "version": 0, "ordering": 0, "operation": "parameter"},
    {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["weights"], "version": 0, "ordering": 1, "operation": "parameter"},
    {"index": 2, "kind": "pure", "target": "%2", "inputs": ["%0"], "version": 0, "ordering": 2, "operation": "index_map:s2,s0,s2,c-1.5,s3,s1"},
    {"index": 3, "kind": "pure", "target": "%3", "inputs": ["%2", "%1"], "version": 0, "ordering": 3, "operation": "mul"},
    {"index": 4, "kind": "primitive", "target": "%4", "inputs": ["%3"], "version": 0, "ordering": 4, "operation": "sum"}
  ],
  "alias_edges": [],
  "control_regions": [],
  "phi_nodes": [],
  "bytecode_offsets": [0, 2, 4]
}"#;
const ORIGINAL_MAP: &str = "index_map:s2,s0,s2,c-1.5,s3,s1";
const PARAMETERS: [f64; 10] = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

#[test]
fn source_map_constants_and_repeated_sources_preserve_weighted_gradients() {
    for (operation, expected_value, expected_gradient) in [
        (
            "index_map:c2,c-3,c0,c4,c1,c-2",
            50.0,
            [0.0, 0.0, 0.0, 0.0, 2.0, -3.0, 0.0, 4.0, 1.0, -2.0],
        ),
        (
            "index_map:s1,s1,s0,s3,s1,s2",
            530.0,
            [30.0, 80.0, 60.0, 40.0, 2.0, 2.0, 1.0, 4.0, 2.0, 3.0],
        ),
    ] {
        let ir = SOURCE_MAP_IR.replace(ORIGINAL_MAP, operation);
        let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &PARAMETERS).unwrap();
        assert!(result.supported, "{:?}", result.blocked_reasons);
        assert_eq!(result.value, Some(expected_value));
        assert_eq!(result.gradient, expected_gradient);
        assert_eq!(result.effect_count, 5);
        assert_eq!(result.supported_effect_count, 5);
    }
}

#[test]
fn malformed_source_maps_refuse_without_poisoning_later_public_replay() {
    for (operation, expected_reason) in [
        ("index_map:", "must not be empty"),
        ("index_map:s0,s1", "target size"),
        ("index_map:s4,s0,s2,c-1.5,s3,s1", "outside source size"),
        (
            "index_map:s,s0,s2,c-1.5,s3,s1",
            "must include a flattened source index",
        ),
        (
            "index_map:s184467440737095516160,s0,s2,c-1.5,s3,s1",
            "not a usize",
        ),
        (
            "index_map:c,s0,s2,c-1.5,s3,s1",
            "must include a finite value",
        ),
        ("index_map:cNaN,s0,s2,c-1.5,s3,s1", "must be finite"),
        ("index_map:cInf,s0,s2,c-1.5,s3,s1", "must be finite"),
        ("index_map:cbad,s0,s2,c-1.5,s3,s1", "not finite f64"),
        ("index_map:x0,s0,s2,c-1.5,s3,s1", "must start with"),
    ] {
        let ir = SOURCE_MAP_IR.replace(ORIGINAL_MAP, operation);
        let refused = interpret_program_ad_effect_ir_value_and_gradient(&ir, &PARAMETERS).unwrap();
        assert!(!refused.supported, "{operation}");
        assert!(
            refused
                .blocked_reasons
                .iter()
                .any(|reason| { reason.contains("index_map") && reason.contains(expected_reason) }),
            "{operation}: {:?}",
            refused.blocked_reasons
        );

        let recovered =
            interpret_program_ad_effect_ir_value_and_gradient(SOURCE_MAP_IR, &PARAMETERS).unwrap();
        assert!(recovered.supported, "{:?}", recovered.blocked_reasons);
        assert_eq!(recovered.value, Some(400.0));
        assert_eq!(
            recovered.gradient,
            [20.0, 60.0, 40.0, 50.0, 3.0, 1.0, 3.0, -1.5, 4.0, 2.0]
        );
    }
}

#[test]
fn public_source_map_owned_boundaries_refuse_and_restore_independent_replay() {
    for operation in [
        ORIGINAL_MAP,
        "index_map:c2,c-3,c0,c4,c1,c-2",
        "index_map:s1,s1,s0,s3,s1,s2",
    ] {
        let ir = SOURCE_MAP_IR.replace(ORIGINAL_MAP, operation);
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let baseline = with_replay_checkpoint(
            move || {
                recorded.set(recorded.get() + 1);
                Ok(())
            },
            || interpret_program_ad_effect_ir_value_and_gradient(&ir, &PARAMETERS),
        )
        .unwrap();
        assert!(baseline.supported, "{:?}", baseline.blocked_reasons);
        assert!(calls.get() > 1);
        for boundary in 1..=calls.get() {
            assert_cancelled_source_map_and_retry(
                &ir,
                &PARAMETERS,
                boundary,
                baseline.value,
                &baseline.gradient,
            );
        }
    }
}

fn assert_cancelled_source_map_and_retry(
    ir: &str,
    parameters: &[f64],
    boundary: usize,
    expected_value: Option<f64>,
    expected_gradient: &[f64],
) {
    let calls = Rc::new(Cell::new(0usize));
    let recorded = Rc::clone(&calls);
    let result = with_replay_checkpoint(
        move || {
            recorded.set(recorded.get() + 1);
            if recorded.get() >= boundary {
                Err("source-map owner cancelled".to_owned())
            } else {
                Ok(())
            }
        },
        || interpret_program_ad_effect_ir_value_and_gradient(ir, parameters),
    );
    match result {
        Err(reason) => assert!(reason.contains("source-map owner cancelled"), "{reason}"),
        Ok(result) => {
            assert!(!result.supported);
            assert!(
                result
                    .blocked_reasons
                    .iter()
                    .any(|reason| reason.contains("source-map owner cancelled")),
                "{:?}",
                result.blocked_reasons
            );
        }
    }
    replay_checkpoint().unwrap();
    let retry = interpret_program_ad_effect_ir_value_and_gradient(ir, parameters).unwrap();
    assert!(retry.supported, "{:?}", retry.blocked_reasons);
    assert_eq!(retry.value, expected_value);
    assert_eq!(retry.gradient, expected_gradient);
}

#[test]
fn public_large_source_map_crosses_metadata_blocks_and_preserves_repeated_scatter() {
    let mut entries = vec!["s0"; 257];
    entries[256] = "c-0";
    let ir = serde_json::json!({
        "format":"program_ad_effect_ir.v1",
        "ssa_values":[
            {"name":"%0","producer":0,"version":0,"shape":[1],"dtype":"float64","effect":0},
            {"name":"%1","producer":1,"version":0,"shape":[257],"dtype":"float64","effect":1},
            {"name":"%2","producer":2,"version":0,"shape":[],"dtype":"float64","effect":2}
        ],
        "effects":[
            {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
            {"index":1,"kind":"pure","target":"%1","inputs":["%0"],"version":0,"ordering":1,"operation":format!("index_map:{}", entries.join(","))},
            {"index":2,"kind":"primitive","target":"%2","inputs":["%1"],"version":0,"ordering":2,"operation":"sum"}
        ],
        "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
    });
    let source = ir.to_string();
    let calls = Rc::new(Cell::new(0usize));
    let recorded = Rc::clone(&calls);
    let baseline = with_replay_checkpoint(
        move || {
            recorded.set(recorded.get() + 1);
            Ok(())
        },
        || interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0]),
    )
    .unwrap();
    assert!(baseline.supported, "{:?}", baseline.blocked_reasons);
    assert_eq!(baseline.value, Some(512.0));
    assert_eq!(baseline.gradient, [256.0]);
    assert!(calls.get() > 1);
    for boundary in [1, calls.get().div_ceil(2), calls.get()] {
        assert_cancelled_source_map_and_retry(&source, &[2.0], boundary, Some(512.0), &[256.0]);
    }
    let mut constant_only = ir.clone();
    constant_only["ssa_values"][1]["shape"] = serde_json::json!([1]);
    constant_only["effects"][1]["operation"] = serde_json::json!("index_map:c-0");
    let constant_result = with_replay_checkpoint(
        || Ok(()),
        || interpret_program_ad_effect_ir_value_and_gradient(&constant_only.to_string(), &[2.0]),
    )
    .unwrap();
    assert!(
        constant_result.supported,
        "{:?}",
        constant_result.blocked_reasons
    );
    assert_eq!(
        constant_result.value.unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(constant_result.gradient, [0.0]);
    let mut malformed = ir;
    malformed["ssa_values"][1]["shape"] = serde_json::json!([256]);
    let refused =
        interpret_program_ad_effect_ir_value_and_gradient(&malformed.to_string(), &[2.0]).unwrap();
    assert!(!refused.supported);
    assert!(
        refused
            .blocked_reasons
            .iter()
            .any(|reason| reason.contains("index_map target size")),
        "{:?}",
        refused.blocked_reasons
    );
    let retry = interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0]).unwrap();
    assert!(retry.supported);
    assert_eq!(retry.value, Some(512.0));
    assert_eq!(retry.gradient, [256.0]);
}
