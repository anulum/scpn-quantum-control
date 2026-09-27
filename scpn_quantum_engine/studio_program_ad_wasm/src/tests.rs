// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public WASM replay boundary tests

    use super::*;

    // f(x, y) = x*x + y*2 — a rational scalar program, so value+gradient are
    // bit-exact reproducible (no transcendentals). Gradient: [2x, 2].
    const RATIONAL_IR: &str = r#"{
      "format": "program_ad_effect_ir.v1",
      "ssa_values": [
        {"name": "%0", "producer": 0, "version": 0, "shape": [], "dtype": "float64", "effect": 0},
        {"name": "%1", "producer": 1, "version": 0, "shape": [], "dtype": "float64", "effect": 1},
        {"name": "%2", "producer": 2, "version": 0, "shape": [], "dtype": "float64", "effect": 2},
        {"name": "%3", "producer": 3, "version": 0, "shape": [], "dtype": "float64", "effect": 3},
        {"name": "%4", "producer": 4, "version": 0, "shape": [], "dtype": "float64", "effect": 4}
      ],
      "effects": [
        {"index": 0, "kind": "parameter", "target": "%0", "inputs": ["x"], "version": 0, "ordering": 0, "operation": "parameter"},
        {"index": 1, "kind": "parameter", "target": "%1", "inputs": ["y"], "version": 0, "ordering": 1, "operation": "parameter"},
        {"index": 2, "kind": "pure", "target": "%2", "inputs": ["%0", "%0"], "version": 0, "ordering": 2, "operation": "mul"},
        {"index": 3, "kind": "pure", "target": "%3", "inputs": ["%1", "2.0"], "version": 0, "ordering": 3, "operation": "mul"},
        {"index": 4, "kind": "pure", "target": "%4", "inputs": ["%2", "%3"], "version": 0, "ordering": 4, "operation": "add"}
      ],
      "alias_edges": [],
      "control_regions": [],
      "phi_nodes": [],
      "bytecode_offsets": [0, 2, 4]
    }"#;

    fn encode(ir: &str, inputs: &[f64]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend((ir.len() as u32).to_le_bytes());
        bytes.extend(ir.as_bytes());
        bytes.extend((inputs.len() as u32).to_le_bytes());
        for value in inputs {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn replays_the_rational_program_value_and_gradient() {
        // x = 3, y = 5 → f = 9 + 10 = 19, grad = [2*3, 2] = [6, 2]
        let values = replay_value_and_gradient(&encode(RATIONAL_IR, &[3.0, 5.0])).expect("replay");
        assert_eq!(values.len(), 3);
        assert_eq!(values[0], 19.0);
        assert_eq!(values[1], 6.0);
        assert_eq!(values[2], 2.0);
    }

    #[test]
    /// Preserve value and gradient bytes through the allocated public FFI.
    fn ffi_round_trip_matches_the_reference() {
        let payload = encode(RATIONAL_IR, &[3.0, 5.0]);
        let expected = replay_value_and_gradient(&payload).expect("replay");
        let output_bytes = expected.len() * 8;
        let input_ptr = scpn_alloc(payload.len());
        let output_ptr = scpn_alloc(output_bytes);
        assert!(!input_ptr.is_null() && !output_ptr.is_null());
        let mut produced = vec![0.0_f64; expected.len()];
        unsafe {
            core::ptr::copy_nonoverlapping(payload.as_ptr(), input_ptr, payload.len());
            let status = scpn_program_ad_replay(input_ptr, payload.len(), output_ptr, output_bytes);
            assert_eq!(status, i32::from(ProgramAdStatus::Ok));
            let raw = core::slice::from_raw_parts(output_ptr, output_bytes);
            let (chunks, remainder) = raw.as_chunks::<8>();
            assert!(remainder.is_empty());
            assert_eq!(chunks.len(), produced.len());
            for (slot, chunk) in produced.iter_mut().zip(chunks) {
                *slot = f64::from_le_bytes(*chunk);
            }
            scpn_free(input_ptr, payload.len());
            scpn_free(output_ptr, output_bytes);
        }
        assert_eq!(produced, expected);
    }

    #[test]
    fn ffi_fails_closed() {
        let payload = encode(RATIONAL_IR, &[3.0, 5.0]);
        let mut sink = [0_u8; 8];
        unsafe {
            assert_eq!(
                scpn_program_ad_replay(core::ptr::null(), 0, sink.as_mut_ptr(), sink.len()),
                i32::from(ProgramAdStatus::NullPointer)
            );
            // wrong output length (value+grad needs 24 bytes, we give 8)
            assert_eq!(
                scpn_program_ad_replay(payload.as_ptr(), payload.len(), sink.as_mut_ptr(), 8),
                i32::from(ProgramAdStatus::OutputMismatch)
            );
        }
    }

    #[test]
    fn parse_fails_closed_on_malformed_input() {
        assert_eq!(
            parse_replay_input(&[0_u8; 2]).expect_err("short"),
            ProgramAdStatus::InvalidLength
        );
        // ir_len claims more bytes than present
        let mut bad = Vec::new();
        bad.extend(1000_u32.to_le_bytes());
        bad.extend(b"{}");
        assert_eq!(
            parse_replay_input(&bad).expect_err("truncated"),
            ProgramAdStatus::InvalidLength
        );
        // invalid UTF-8 in the IR region
        let mut bad_utf8 = Vec::new();
        bad_utf8.extend(2_u32.to_le_bytes());
        bad_utf8.extend([0xff, 0xfe]);
        bad_utf8.extend(0_u32.to_le_bytes());
        assert_eq!(
            parse_replay_input(&bad_utf8).expect_err("utf8"),
            ProgramAdStatus::InvalidUtf8
        );
        assert_eq!(
            parse_replay_input(&encode("", &[])).expect_err("empty IR"),
            ProgramAdStatus::InvalidLength
        );
        let oversized_ir = "x".repeat(MAX_PROGRAM_AD_REPLAY_IR_BYTES + 1);
        assert_eq!(
            parse_replay_input(&encode(&oversized_ir, &[])).expect_err("oversized IR"),
            ProgramAdStatus::InvalidLength
        );
        let oversized_inputs = vec![0.0; MAX_PROGRAM_AD_REPLAY_INPUTS + 1];
        assert_eq!(
            parse_replay_input(&encode("{}", &oversized_inputs))
                .expect_err("oversized input arity"),
            ProgramAdStatus::InvalidLength
        );
        assert_eq!(
            parse_replay_input(&encode("{}", &[f64::NAN])).expect_err("non-finite input"),
            ProgramAdStatus::NonFiniteInput
        );
    }

    #[test]
    fn replay_rejects_a_non_bounded_program() {
        // A structurally valid but empty-effects IR is not a supported scalar
        // program; the replay must fail closed, not fabricate a gradient.
        let empty = r#"{"format":"program_ad_effect_ir.v1","ssa_values":[],"effects":[],"alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]}"#;
        let status = replay_value_and_gradient(&encode(empty, &[])).expect_err("unsupported");
        assert!(matches!(
            status,
            ProgramAdStatus::Unsupported | ProgramAdStatus::ReplayError
        ));
    }


#[test]
fn ffi_refuses_wrong_output_before_ir_replay_and_preserves_sink() {
    let payload = encode("{", &[]);
    for length in [0, 16, usize::MAX] {
        let mut sink = [0x5a_u8; 16];
        let status = unsafe {
            scpn_program_ad_replay(payload.as_ptr(), payload.len(), sink.as_mut_ptr(), length)
        };
        assert_eq!(status, i32::from(ProgramAdStatus::OutputMismatch));
        assert_eq!(sink, [0x5a;16]);
    }
    let payload = encode(RATIONAL_IR, &[3.0,5.0]);
    let mut sink = [0_u8;24];
    let status = unsafe {
        scpn_program_ad_replay(payload.as_ptr(), payload.len(), sink.as_mut_ptr(), sink.len())
    };
    assert_eq!(status, i32::from(ProgramAdStatus::Ok));
    let values:Vec<_> = sink.as_chunks::<8>().0.iter().map(|bytes|f64::from_le_bytes(*bytes)).collect();
    assert_eq!(values,[19.0,6.0,2.0]);
}

#[test]
fn ffi_refuses_oversized_readable_payload_before_slice_decode_and_recovers() {
    let payload = vec![0_u8;MAX_PROGRAM_AD_REPLAY_IR_BYTES+8+MAX_PROGRAM_AD_REPLAY_INPUTS*8+1];
    let mut sink = [0x5a_u8;24];
    let status = unsafe {
        scpn_program_ad_replay(payload.as_ptr(),payload.len(),sink.as_mut_ptr(),sink.len())
    };
    assert_eq!(status,i32::from(ProgramAdStatus::InvalidLength));
    assert_eq!(sink,[0x5a;24]);
    assert_eq!(replay_value_and_gradient(&encode(RATIONAL_IR,&[3.0,5.0])).unwrap(),[19.0,6.0,2.0]);
}

#[test]
fn public_decode_refuses_complete_layout_and_nonfinite_errors_before_recovery() {
    let good = encode(RATIONAL_IR,&[3.0,5.0]);
    let mut extra = good.clone();extra.push(0);
    let truncated = good[..good.len()-1].to_vec();
    for invalid in [extra,truncated] {
        assert_eq!(parse_replay_input(&invalid).unwrap_err(),ProgramAdStatus::InvalidLength);
        assert_eq!(replay_value_and_gradient(&good).unwrap(),[19.0,6.0,2.0]);
    }
    for invalid in [f64::NAN,f64::INFINITY,f64::NEG_INFINITY] {
        assert_eq!(parse_replay_input(&encode(RATIONAL_IR,&[3.0,invalid])).unwrap_err(),ProgramAdStatus::NonFiniteInput);
        assert_eq!(replay_value_and_gradient(&good).unwrap(),[19.0,6.0,2.0]);
    }
}

#[test]
fn declared_memory_budget_is_inclusive_and_recovers_after_refusal() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_memory_admission, with_replay_metadata_admission,
    };
    use std::{cell::Cell, rc::Rc};
    let payload = encode(RATIONAL_IR, &[3.0, 5.0]);
    let requests = Rc::new(Cell::new(0usize));
    let metadata = Rc::clone(&requests);
    let numeric = Rc::clone(&requests);
    let actual = with_replay_metadata_admission(
        move |bytes| { metadata.set(metadata.get().checked_add(bytes).unwrap()); Ok(()) },
        || with_replay_memory_admission(
            move |request| {
                numeric.set(numeric.get().checked_add(request.total_bytes()?).unwrap()); Ok(())
            },
            || replay_value_and_gradient(&payload),
        ),
    ).unwrap();
    assert_eq!(actual, [19.0, 6.0, 2.0]);
    // The envelope contains eight header bytes, two input f64s and borrowed IR.
    // The owned copies omit the headers and retain an additional three outputs.
    let required = requests.get() + payload.len() - 8 + 3 * 8;
    assert_eq!(replay_value_and_gradient_with_memory_budget(&payload, required).unwrap(), actual);
    for limit in [0, required - 1, MAX_PROGRAM_AD_REPLAY_MEMORY_BYTES + 1] {
        assert_eq!(replay_value_and_gradient_with_memory_budget(&payload, limit), Err(ProgramAdStatus::ReplayError));
        assert_eq!(replay_value_and_gradient(&payload).unwrap(), actual);
    }
}

#[test]
fn ffi_refuses_oversized_intermediate_without_touching_output() {
    let oversized = r#"{
        "format":"program_ad_effect_ir.v1",
        "ssa_values":[
            {"name":"%0","producer":0,"version":0,"shape":[],"dtype":"float64","effect":0},
            {"name":"%1","producer":1,"version":0,"shape":[100000000],"dtype":"float64","effect":1},
            {"name":"%2","producer":2,"version":0,"shape":[],"dtype":"float64","effect":2}
        ],
        "effects":[
            {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
            {"index":1,"kind":"pure","target":"%1","inputs":["%0"],"version":0,"ordering":1,"operation":"broadcast_to"},
            {"index":2,"kind":"pure","target":"%2","inputs":["%1"],"version":0,"ordering":2,"operation":"sum"}
        ],
        "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
    }"#;
    let payload = encode(oversized, &[3.0]);
    let mut output = [0x5a_u8; 16];
    let status = unsafe { scpn_program_ad_replay(payload.as_ptr(), payload.len(), output.as_mut_ptr(), output.len()) };
    assert_eq!(status, i32::from(ProgramAdStatus::ReplayError));
    assert_eq!(output, [0x5a; 16]);
    assert_eq!(replay_value_and_gradient(&encode(RATIONAL_IR, &[3.0, 5.0])).unwrap(), [19.0, 6.0, 2.0]);
}
