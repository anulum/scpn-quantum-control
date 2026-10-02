// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — supported source through the public browser compiler

//! Source-bound program semantics, exact refusal locations and bounded imports.

use scpn_quantum_studio_wasm_kernel::program_source::compile_program_source;
use scpn_quantum_studio_wasm_kernel::program_source_abi::{
    scpn_program_source_compile, MAX_OUTPUT_BYTES,
};
use scpn_quantum_studio_wasm_kernel::{scpn_alloc, scpn_free};

const HEADER: &str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncreg c[2];\n";

#[test]
/// Use the same independently authored source cases as native Python and browser bindings.
fn shared_source_corpus() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/data/program_authoring/corpus.json"
    ))
    .unwrap();
    for case in corpus["cases"].as_array().unwrap() {
        let source = case["source"].as_str().unwrap();
        let result = compile_program_source(source);
        if case["ok"] == true {
            let record = result.unwrap_or_else(|error| panic!("{}: {error:?}", case["name"]));
            assert_eq!(
                serde_json::json!(record
                    .operations
                    .iter()
                    .map(|op| &op.name)
                    .collect::<Vec<_>>()),
                case["operations"],
                "{}",
                case["name"]
            );
            assert_eq!(serde_json::json!(record.measurements), case["measurements"]);
            if case.get("parameters").is_some() {
                assert_eq!(
                    serde_json::json!(record
                        .operations
                        .iter()
                        .map(|op| &op.parameters)
                        .collect::<Vec<_>>()),
                    case["parameters"]
                );
            }
            if case.get("conditions").is_some() {
                assert_eq!(
                    serde_json::json!(record
                        .operations
                        .iter()
                        .map(|op| &op.condition)
                        .collect::<Vec<_>>()),
                    case["conditions"]
                );
            }
        } else {
            let error = result.expect_err("unsupported original source");
            assert_eq!(
                error.code,
                case["code"].as_str().unwrap(),
                "{}",
                case["name"]
            );
            let token = case["token"].as_str().unwrap();
            let suffix: String = source
                .chars()
                .skip(case["token_after"].as_u64().unwrap() as usize)
                .collect();
            let scalar_start = case["token_after"].as_u64().unwrap() as usize
                + suffix[..suffix.find(token).unwrap()].chars().count();
            assert_eq!(
                (error.source_span.start, error.source_span.end),
                (scalar_start, scalar_start + token.chars().count()),
                "{}",
                case["name"]
            );
        }
    }
}

#[test]
/// Retain ordered measurement mapping and exact source on a supported roundtrip.
fn test_program_authoring_01() {
    let source =
        format!("{HEADER}h q[0];\ncx q[0],q[1];\nmeasure q[1] -> c[0];\nmeasure q[0] -> c[1];");
    let first = compile_program_source(&source).expect("supported native source");
    let restored = compile_program_source(&first.source).expect("source export restores");
    assert_eq!(first, restored);
    assert_eq!(first.measurements, vec![(1, 0), (0, 1)]);
    assert_eq!(first.operations[1].name, "cx");
    assert_eq!(first.operations[1].qubits, vec![0, 1]);
    assert_eq!(first.execution_status, "emitted_not_executed");
}

#[test]
/// Report the exact invalid token without admitting any executable source.
fn test_program_authoring_02() {
    let source = format!("{HEADER}  unexpected q[0];");
    let refusal = compile_program_source(&source).expect_err("unknown instruction");
    assert_eq!(refusal.code, "unsupported_operation");
    assert_eq!(
        (refusal.source_span.line, refusal.source_span.column),
        (5, 3)
    );
    assert_eq!(
        source
            .chars()
            .skip(refusal.source_span.start)
            .take(refusal.source_span.end - refusal.source_span.start)
            .collect::<String>(),
        "unexpected"
    );
}

#[test]
/// Imported Python and external includes have no filesystem or evaluation path.
fn test_program_authoring_03() {
    for source in [
        "import os; os.remove('file');",
        "OPENQASM 2.0; include \"/etc/passwd\"; qreg q[1];",
    ] {
        assert!(compile_program_source(source).is_err());
    }
}

#[test]
/// Preserve conditional controls and the exact IEEE rotation parameter.
fn conditional_readout_and_phase_are_not_discarded() {
    let source = format!("{HEADER}h q[0];measure q[0] -> c[0];if(c==1) rz(-0.7853981633974492) q[1];measure q[1] -> c[1];");
    let record = compile_program_source(&source).expect("supported classical control");
    let rotation = &record.operations[2];
    assert_eq!(rotation.name, "rz");
    assert_eq!(
        rotation.parameters,
        vec![format!("{:016x}", (-0.7853981633974492_f64).to_bits())]
    );
    assert_eq!(
        rotation
            .condition
            .as_ref()
            .expect("original condition")
            .value,
        "1"
    );
    assert_eq!(record.measurements, vec![(0, 0), (1, 1)]);
}

#[test]
/// Unicode comments do not corrupt the original scalar source coordinates.
fn unicode_comment_offsets_remain_original() {
    let source = format!("{HEADER}// α😀 preserved\n  x q[1];");
    let record = compile_program_source(&source).expect("comment is inert");
    let span = &record.operations[0].source_span;
    assert_eq!((span.line, span.column), (6, 3));
    assert_eq!(
        source
            .chars()
            .skip(span.start)
            .take(span.end - span.start)
            .collect::<String>(),
        "x q[1];"
    );
}

#[test]
/// Width and operation budgets refuse before a source plan can be produced.
fn source_budgets_are_enforced() {
    for source in [
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[9];".to_owned(),
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[65];".to_owned(),
        format!("{HEADER}{}", "x q[0];".repeat(4097)),
    ] {
        assert!(compile_program_source(&source).is_err());
    }
}

/// Exercise the real allocator and public ABI with independently owned buffers.
fn source_abi(input: &[u8], output_length: usize) -> (i32, Vec<u8>) {
    let input_size = input.len().max(1);
    let output_size = output_length.max(1);
    let input_pointer = scpn_alloc(input_size);
    let output_pointer = scpn_alloc(output_size);
    assert!(!input_pointer.is_null() && !output_pointer.is_null());
    // SAFETY: both original guest allocations have their declared lengths and do not overlap.
    let (status, output) = unsafe {
        core::ptr::copy_nonoverlapping(input.as_ptr(), input_pointer, input.len());
        core::ptr::write_bytes(output_pointer, 0xa5, output_size);
        let status =
            scpn_program_source_compile(input_pointer, input.len(), output_pointer, output_length);
        let output = core::slice::from_raw_parts(output_pointer, output_size).to_vec();
        scpn_free(input_pointer, input_size);
        scpn_free(output_pointer, output_size);
        (status, output)
    };
    (status, output)
}

#[test]
/// Positive ABI records contain the actual exact source; syntax refusals remain diagnostics.
fn source_abi_emits_original_records_and_authored_diagnostics() {
    for (source, admitted) in [
        (format!("{HEADER}if(c==2) rz(-0.0) q[1];"), true),
        ("import os".to_owned(), false),
        (String::new(), false),
    ] {
        let (status, bytes) = source_abi(source.as_bytes(), 8192);
        assert!(status > 0);
        let wire: serde_json::Value = serde_json::from_slice(&bytes[..status as usize]).unwrap();
        assert_eq!(wire["ok"], admitted);
        if admitted {
            assert_eq!(wire["value"]["source"], source);
            assert_eq!(
                wire["value"]["operations"][0]["parameters"],
                serde_json::json!(["8000000000000000"])
            );
        } else {
            assert_eq!(wire["diagnostic"]["code"], "invalid_source");
        }
        assert!(bytes[status as usize..].iter().all(|byte| *byte == 0xa5));
    }
}

#[test]
/// Negative statuses preserve all original output bytes without a partial success record.
fn source_abi_transport_refusals_leave_output_unchanged() {
    for (input, length, expected) in [
        (vec![0xff], 4096, -3),
        (vec![b'x'; 1_048_577], 4096, -2),
        (HEADER.as_bytes().to_vec(), 0, -2),
        (HEADER.as_bytes().to_vec(), MAX_OUTPUT_BYTES + 1, -2),
        (HEADER.as_bytes().to_vec(), 1, -4),
    ] {
        let (status, output) = source_abi(&input, length);
        assert_eq!(status, expected);
        assert!(output.iter().all(|byte| *byte == 0xa5));
    }
    // SAFETY: null pointers are explicitly refused before either allocation is read.
    assert_eq!(
        unsafe { scpn_program_source_compile(core::ptr::null(), 0, core::ptr::null_mut(), 0) },
        -1
    );
}

#[test]
/// Empty, lexical and source transport limits retain located refusal categories.
fn source_parser_transport_and_missing_tokens_are_bounded() {
    for (source, code) in [
        ("".to_owned(), "invalid_source"),
        (" \n".to_owned(), "invalid_source"),
        ("x".repeat(1_048_577), "source_budget"),
        (";".repeat(65_537), "source_budget"),
        (format!("{HEADER}h q[0]"), "invalid_source"),
        (format!("{HEADER}if(c=="), "invalid_source"),
        (format!("{HEADER}barrier q"), "invalid_source"),
    ] {
        assert_eq!(compile_program_source(&source).unwrap_err().code, code);
    }
}
