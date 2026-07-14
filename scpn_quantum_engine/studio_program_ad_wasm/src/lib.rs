// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Studio WASM program-AD gradient replay kernel

//! Standalone WASM kernel that replays the bounded program-AD value+gradient.
//!
//! It runs the SAME bounded replay the engine does — the
//! `scpn-quantum-program-ad-replay` crate with no Python bindings — so a visitor
//! recomputes a displayed gradient bit-exactly in their browser. This kernel is
//! deliberately separate from the compile/simulate kernel: the replay pulls
//! serde_json + nalgebra, so bundling it would bloat the lightweight recompute
//! path; the panel loads this module only when the gradient card is used.
//!
//! Input freezes a serialised effect-IR plus its input bindings; output is the
//! scalar value followed by the reverse-mode gradient. Any program outside the
//! bounded set, or a malformed payload, fails closed with a negative status
//! rather than a fabricated gradient.

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;

/// Maximum UTF-8 effect-IR size shared with the Python artifact packer.
pub const MAX_PROGRAM_AD_REPLAY_IR_BYTES: usize = 1_048_576;
/// Maximum scalar-input arity shared with the Python artifact packer.
pub const MAX_PROGRAM_AD_REPLAY_INPUTS: usize = 4_096;

/// Fail-closed status codes for the program-AD replay FFI.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[repr(i32)]
pub enum ProgramAdStatus {
    Ok = 0,
    NullPointer = -1,
    InvalidLength = -2,
    InvalidUtf8 = -3,
    ReplayError = -4,
    Unsupported = -5,
    OutputMismatch = -6,
    NonFiniteInput = -7,
}

impl From<ProgramAdStatus> for i32 {
    fn from(value: ProgramAdStatus) -> Self {
        value as i32
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut raw = [0_u8; 4];
    raw.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(raw)
}

fn read_f64(bytes: &[u8], offset: usize) -> f64 {
    let mut raw = [0_u8; 8];
    raw.copy_from_slice(&bytes[offset..offset + 8]);
    f64::from_le_bytes(raw)
}

/// Decode the canonical little-endian replay input.
///
/// Layout: `u32 ir_len | ir_bytes (UTF-8 effect-IR JSON) | u32 n_inputs |
/// n_inputs * f64`.
pub fn parse_replay_input(bytes: &[u8]) -> Result<(String, Vec<f64>), ProgramAdStatus> {
    if bytes.len() < 4 {
        return Err(ProgramAdStatus::InvalidLength);
    }
    let ir_len = read_u32(bytes, 0) as usize;
    if ir_len == 0 || ir_len > MAX_PROGRAM_AD_REPLAY_IR_BYTES {
        return Err(ProgramAdStatus::InvalidLength);
    }
    let after_ir = 4usize
        .checked_add(ir_len)
        .ok_or(ProgramAdStatus::InvalidLength)?;
    let inputs_header_end = after_ir
        .checked_add(4)
        .ok_or(ProgramAdStatus::InvalidLength)?;
    if bytes.len() < inputs_header_end {
        return Err(ProgramAdStatus::InvalidLength);
    }
    let ir = core::str::from_utf8(&bytes[4..after_ir])
        .map_err(|_| ProgramAdStatus::InvalidUtf8)?
        .to_owned();
    let n_inputs = read_u32(bytes, after_ir) as usize;
    if n_inputs > MAX_PROGRAM_AD_REPLAY_INPUTS {
        return Err(ProgramAdStatus::InvalidLength);
    }
    let expected_len = inputs_header_end
        .checked_add(
            n_inputs
                .checked_mul(8)
                .ok_or(ProgramAdStatus::InvalidLength)?,
        )
        .ok_or(ProgramAdStatus::InvalidLength)?;
    if bytes.len() != expected_len {
        return Err(ProgramAdStatus::InvalidLength);
    }
    let mut inputs = Vec::with_capacity(n_inputs);
    for index in 0..n_inputs {
        let value = read_f64(bytes, inputs_header_end + index * 8);
        if !value.is_finite() {
            return Err(ProgramAdStatus::NonFiniteInput);
        }
        inputs.push(value);
    }
    Ok((ir, inputs))
}

/// Replay a bounded program-AD value+gradient, returning `[value, gradient…]`.
pub fn replay_value_and_gradient(bytes: &[u8]) -> Result<Vec<f64>, ProgramAdStatus> {
    let (ir, inputs) = parse_replay_input(bytes)?;
    let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs)
        .map_err(|_| ProgramAdStatus::ReplayError)?;
    if !result.supported {
        return Err(ProgramAdStatus::Unsupported);
    }
    let value = result.value.ok_or(ProgramAdStatus::Unsupported)?;
    let mut out = Vec::with_capacity(1 + result.gradient.len());
    out.push(value);
    out.extend_from_slice(&result.gradient);
    Ok(out)
}

/// Allocate `len` bytes inside the module's linear memory for host input.
///
/// Returns a null pointer for zero-length or failed allocations; the host must
/// treat null as fail-closed and release every successful allocation with
/// [`scpn_free`] using the same length.
#[no_mangle]
pub extern "C" fn scpn_alloc(len: usize) -> *mut u8 {
    let Ok(layout) = core::alloc::Layout::array::<u8>(len) else {
        return core::ptr::null_mut();
    };
    if layout.size() == 0 {
        return core::ptr::null_mut();
    }
    // SAFETY: the layout is non-zero-sized and well-formed.
    unsafe { std::alloc::alloc(layout) }
}

/// Release a buffer previously returned by [`scpn_alloc`].
///
/// # Safety
///
/// `ptr` must come from `scpn_alloc(len)` with the identical `len` and must not
/// be used afterwards. Null pointers and zero lengths are ignored.
#[no_mangle]
pub unsafe extern "C" fn scpn_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let Ok(layout) = core::alloc::Layout::array::<u8>(len) else {
        return;
    };
    // SAFETY: caller contract guarantees ptr/layout came from scpn_alloc.
    unsafe { std::alloc::dealloc(ptr, layout) };
}

/// Replay a bounded program-AD gradient and write `[value ; gradient]`.
///
/// The output is `(1 + k)` little-endian `f64` — the scalar value followed by
/// the `k` reverse-mode gradient components. `output_len` is the byte length the
/// host allocated (from the committed unit's gradient arity) and must equal
/// `(1 + k) * 8`, else the call fails closed.
///
/// # Safety
///
/// `input_ptr` must point to `input_len` readable bytes and `output_ptr` to
/// `output_len` writable bytes. Null pointers, malformed payloads, and any
/// program outside the bounded replay set return a negative status code without
/// writing output.
#[no_mangle]
pub unsafe extern "C" fn scpn_program_ad_replay(
    input_ptr: *const u8,
    input_len: usize,
    output_ptr: *mut u8,
    output_len: usize,
) -> i32 {
    if input_ptr.is_null() || output_ptr.is_null() {
        return ProgramAdStatus::NullPointer.into();
    }
    let input = unsafe { core::slice::from_raw_parts(input_ptr, input_len) };
    let values = match replay_value_and_gradient(input) {
        Ok(values) => values,
        Err(status) => return status.into(),
    };
    let Some(needed) = values.len().checked_mul(8) else {
        return ProgramAdStatus::OutputMismatch.into();
    };
    if output_len != needed {
        return ProgramAdStatus::OutputMismatch.into();
    }
    let output = unsafe { core::slice::from_raw_parts_mut(output_ptr, output_len) };
    for (index, value) in values.iter().enumerate() {
        output[index * 8..index * 8 + 8].copy_from_slice(&value.to_le_bytes());
    }
    ProgramAdStatus::Ok.into()
}

#[cfg(test)]
mod tests {
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
            for (slot, chunk) in produced.iter_mut().zip(raw.chunks_exact(8)) {
                let mut buf = [0_u8; 8];
                buf.copy_from_slice(chunk);
                *slot = f64::from_le_bytes(buf);
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
}
