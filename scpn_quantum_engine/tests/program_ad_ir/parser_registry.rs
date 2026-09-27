// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD IR parser and registry tests

#[test]
fn program_ad_effect_ir_parser_round_trips_python_payload_shape() {
    let ir = parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR).unwrap();

    assert_eq!(ir.format, "program_ad_effect_ir.v1");
    assert_eq!(ir.ssa_values.len(), 2);
    assert_eq!(ir.ssa_values[1].shape, vec![2]);
    assert_eq!(ir.effects[1].kind, "control_branch");
    assert_eq!(ir.alias_edges[0].kind, "view_alias");
    assert_eq!(ir.control_regions[0].kind, "runtime_branch");
    assert!(ir.control_regions[0].entered);
    assert_eq!(
        ir.phi_nodes[0].incoming,
        vec!["executed_true", "executed_false"]
    );
    assert_eq!(ir.bytecode_offsets, vec![0, 2, 4]);

    let summary = ir.metadata_summary();
    assert_eq!(summary.format, "program_ad_effect_ir.v1");
    assert_eq!(summary.ssa_value_count, 2);
    assert_eq!(summary.effect_count, 2);
    assert_eq!(summary.alias_edge_count, 1);
    assert_eq!(summary.control_region_count, 1);
    assert_eq!(summary.phi_node_count, 1);
    assert_eq!(summary.claim_boundary, "metadata_only_no_program_execution");
}

#[test]
fn program_ad_effect_ir_parser_fails_closed_on_malformed_payloads() {
    let wrong_format =
        VALID_PROGRAM_AD_IR.replace("program_ad_effect_ir.v1", "program_ad_effect_ir.v2");
    assert!(parse_program_ad_effect_ir(&wrong_format)
        .unwrap_err()
        .contains("format must be program_ad_effect_ir.v1"));

    let wrong_effect_shape = r#"{
      "format": "program_ad_effect_ir.v1",
      "ssa_values": [],
      "effects": {},
      "alias_edges": [],
      "control_regions": [],
      "phi_nodes": [],
      "bytecode_offsets": []
    }"#;
    assert!(parse_program_ad_effect_ir(wrong_effect_shape)
        .unwrap_err()
        .contains("effects"));

    let bad_phi = VALID_PROGRAM_AD_IR.replace(
        "\"incoming\": [\"executed_true\", \"executed_false\"]",
        "\"incoming\": [\"executed_true\"]",
    );
    assert!(parse_program_ad_effect_ir(&bad_phi)
        .unwrap_err()
        .contains("phi_nodes incoming"));

    let bad_source_line = VALID_PROGRAM_AD_IR.replace(
        "\"kind\": \"runtime_branch\", \"predicate\": \"%0 > 0\", \"entered\": true, \"source_line\": null",
        "\"kind\": \"runtime_branch\", \"predicate\": \"%0 > 0\", \"entered\": true, \"source_line\": 0",
    );
    assert!(parse_program_ad_effect_ir(&bad_source_line)
        .unwrap_err()
        .contains("source_line"));
}

#[test]
fn program_ad_registry_metadata_mirror_validates_coverage_snapshot() {
    let result = mirror_program_ad_registry_metadata(VALID_REGISTRY_COVERAGE_SNAPSHOT).unwrap();

    assert!(result.supported);
    assert_eq!(result.primitive_count, 3);
    assert_eq!(result.covered_primitives, 3);
    assert_eq!(result.family_counts.get("elementwise"), Some(&2));
    assert_eq!(result.family_counts.get("linalg"), Some(&1));
    assert_eq!(result.facet_counts.get("derivative_rule"), Some(&3));
    assert_eq!(result.facet_counts.get("lowering_metadata"), Some(&3));
    assert_eq!(result.executable_operations, vec!["det", "sin", "sqrt"]);
    assert_eq!(result.executable_operation_count, 3);
    assert_eq!(
        result.claim_boundary,
        "rust_program_ad_registry_metadata_mirror_only_no_execution_promotion"
    );
}

#[test]
fn program_ad_registry_metadata_mirror_fails_closed_on_snapshot_drift() {
    let drifted =
        VALID_REGISTRY_COVERAGE_SNAPSHOT.replace("\"elementwise\": 2", "\"elementwise\": 3");

    assert!(mirror_program_ad_registry_metadata(&drifted)
        .unwrap_err()
        .contains("family_counts"));
    assert!(mirror_program_ad_registry_metadata("")
        .unwrap_err()
        .contains("non-empty JSON"));
}


#[test]
fn public_parser_observes_owner_before_json_and_recovers() {
    let refused = scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_checkpoint(
        || Err("parser owner cancelled".to_owned()),
        || parse_program_ad_effect_ir("{"),
    );
    assert_eq!(refused.unwrap_err(), "parser owner cancelled");
    let retry = parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR).unwrap();
    assert_eq!(retry.format, "program_ad_effect_ir.v1");
    assert_eq!(retry.ssa_values.len(), 2);
    assert_eq!(retry.effects[1].kind, "control_branch");
}

#[test]
fn public_parser_admits_metadata_separately_and_refuses_before_json() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_memory_admission, with_replay_metadata_admission,
    };
    let refused = with_replay_metadata_admission(
        |_| Err("parser metadata capacity refused".to_owned()),
        || parse_program_ad_effect_ir("{"),
    );
    assert_eq!(refused.unwrap_err(), "parser metadata capacity refused");
    let retry = with_replay_memory_admission(
        |_| Err("numeric policy must not receive parser metadata".to_owned()),
        || parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR),
    ).unwrap();
    assert_eq!(retry.ssa_values.len(), 2);
    assert_eq!(retry.bytecode_offsets, vec![0, 2, 4]);
}

#[test]
fn public_parser_metadata_capacity_boundary_and_parent_ownership() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_metadata_admission;
    use std::{cell::Cell, rc::Rc};
    let used = Rc::new(Cell::new(0usize));
    let observed = Rc::clone(&used);
    let reference = with_replay_metadata_admission(
        move |bytes| { observed.set(observed.get().checked_add(bytes).unwrap()); Ok(()) },
        || parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR),
    ).unwrap();
    let required = used.get();
    assert!(required > VALID_PROGRAM_AD_IR.len());
    for limit in [required, required - 1] {
        let charged = Cell::new(0usize);
        let result = with_replay_metadata_admission(
            move |bytes| {
                let total = charged.get().checked_add(bytes).unwrap();
                if total > limit { return Err("metadata limit".to_owned()); }
                charged.set(total); Ok(())
            },
            || parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR),
        );
        if limit == required { assert_eq!(result.unwrap(), reference); }
        else { assert_eq!(result.unwrap_err(), "metadata limit"); }
    }
    let refused = with_replay_metadata_admission(
        |_| Err("parent metadata refused".to_owned()),
        || with_replay_metadata_admission(|_| Ok(()), || parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR)),
    );
    assert_eq!(refused.unwrap_err(), "parent metadata refused");
    assert_eq!(parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR).unwrap(), reference);
}

#[test]
fn public_parser_retains_duplicate_and_escaped_key_semantics() {
    let expected = parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR).unwrap();
    let duplicate = VALID_PROGRAM_AD_IR.replacen(
        "\"format\": \"program_ad_effect_ir.v1\"",
        "\"format\": false, \"\\u0066ormat\": \"program_ad_effect_ir.v1\"",
        1,
    );
    assert_ne!(duplicate, VALID_PROGRAM_AD_IR);
    assert_eq!(parse_program_ad_effect_ir(&duplicate).unwrap(), expected);
    let invalid = VALID_PROGRAM_AD_IR.replacen("{", "{\"unknown\":1e9999,", 1);
    assert!(parse_program_ad_effect_ir(&invalid).unwrap_err().contains("invalid JSON"));
}

#[test]
fn public_parser_preserves_positional_records_and_escaped_owned_strings() {
    let original = parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR).unwrap();
    let positional = VALID_PROGRAM_AD_IR.replace(
        r#"{"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0}"#,
        r#"["\u00250", 0, 0, [], "float64", 0]"#,
    );
    assert_ne!(positional, VALID_PROGRAM_AD_IR);
    assert_eq!(parse_program_ad_effect_ir(&positional).unwrap(), original);
    let duplicate = VALID_PROGRAM_AD_IR.replace(
        r#""name": "%0""#, r#""name": false, "name": "%0""#,
    );
    assert_ne!(duplicate, VALID_PROGRAM_AD_IR);
    assert_eq!(parse_program_ad_effect_ir(&duplicate).unwrap(), original);
}

#[test]
fn public_parser_metadata_policy_restores_after_unwind() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_metadata_admission;
    let unwound = std::panic::catch_unwind(|| {
        with_replay_metadata_admission(
            |_| panic!("metadata owner unwind"),
            || parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR),
        )
    });
    assert!(unwound.is_err());
    assert_eq!(parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR).unwrap().bytecode_offsets, vec![0, 2, 4]);
}

#[test]
fn public_parser_validates_unknown_numeric_tokens_without_typed_float_storage() {
    let reference = parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR).unwrap();
    for token in ["1.2345678901234567890123456789", "1e-300", "-0.0", "18446744073709551616", "1e-9999"] {
        let source = VALID_PROGRAM_AD_IR.replacen("{", &format!("{{\"unknown_number\":{token},"), 1);
        assert_eq!(parse_program_ad_effect_ir(&source).unwrap(), reference);
    }
    for token in ["1e9999", "1e+", "01", "NaN"] {
        let source = VALID_PROGRAM_AD_IR.replacen("{", &format!("{{\"unknown_number\":{token},"), 1);
        assert!(parse_program_ad_effect_ir(&source).unwrap_err().contains("invalid JSON"));
    }
    assert_eq!(parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR).unwrap(), reference);
}

#[test]
fn public_parser_preserves_json_depth_and_typed_integer_refusal() {
    let nested = format!("{}0{}", "[".repeat(128), "]".repeat(128));
    let source = VALID_PROGRAM_AD_IR.replacen("{", &format!("{{\"unknown_nested\":{nested},"), 1);
    assert!(parse_program_ad_effect_ir(&source).unwrap_err().contains("invalid JSON"));
    let malformed = VALID_PROGRAM_AD_IR.replacen("\"producer\": 0", "\"producer\": 1.0", 1);
    assert_ne!(malformed, VALID_PROGRAM_AD_IR);
    assert!(parse_program_ad_effect_ir(&malformed).unwrap_err().contains("schema"));
    assert_eq!(parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR).unwrap().bytecode_offsets, vec![0, 2, 4]);
}

#[test]
fn public_parser_depth_scan_ignores_quoted_braces_and_rejects_invalid_unicode() {
    let reference = parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR).unwrap();
    let quoted = serde_json::to_string(&"{[\"".repeat(128)).unwrap();
    let source = VALID_PROGRAM_AD_IR.replacen("{", &format!("{{\"unknown_text\":{quoted},"), 1);
    assert_eq!(parse_program_ad_effect_ir(&source).unwrap(), reference);
    let malformed = VALID_PROGRAM_AD_IR.replacen("{", r#"{"unknown_text":"\uD800","#, 1);
    assert!(parse_program_ad_effect_ir(&malformed).unwrap_err().contains("invalid JSON"));
    let wrong_string = VALID_PROGRAM_AD_IR.replacen("\"name\": \"%0\"", "\"name\": 1.234567890123456789e100", 1);
    assert_ne!(wrong_string, VALID_PROGRAM_AD_IR);
    assert!(parse_program_ad_effect_ir(&wrong_string).unwrap_err().contains("schema"));
    assert_eq!(parse_program_ad_effect_ir(VALID_PROGRAM_AD_IR).unwrap(), reference);
}
