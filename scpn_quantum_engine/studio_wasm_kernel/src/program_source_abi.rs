// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — emitted-only source compilation host ABI

//! Versioned source records and diagnostics through the original WASM allocator.

use crate::program_source::{compile_program_source, MAX_SOURCE_BYTES};

/// Largest response buffer; covers bounded4096-node IR and exact source text.
pub const MAX_OUTPUT_BYTES: usize = 8 * 1_048_576;

/// Compile source to a JSON success record or a located authored refusal.
///
/// Returns the positive number of bytes written. Negative statuses are
/// -1 for null pointers, -2 for invalid source/buffer lengths, -3 for invalid
/// UTF8 and -4 for an insufficient response buffer.
/// No bytes are written on a negative status. A positive response can still be
/// a diagnostic; consumers must inspect its ok discriminant.
///
/// # Safety
///
/// Input and output point to non-overlapping valid allocations of their
/// declared lengths, acquired through scpn_alloc and owned by the host.
#[no_mangle]
pub unsafe extern "C" fn scpn_program_source_compile(
    input_ptr: *const u8,
    input_len: usize,
    output_ptr: *mut u8,
    output_len: usize,
) -> i32 {
    if input_ptr.is_null() || output_ptr.is_null() {
        return -1;
    }
    if input_len > MAX_SOURCE_BYTES || output_len == 0 || output_len > MAX_OUTPUT_BYTES {
        return -2;
    }
    let bytes = unsafe { core::slice::from_raw_parts(input_ptr, input_len) };
    let Ok(source) = core::str::from_utf8(bytes) else {
        return -3;
    };
    let value = match compile_program_source(source) {
        Ok(record) => serde_json::json!({ "ok": true, "value": record }),
        Err(diagnostic) => serde_json::json!({ "ok": false, "diagnostic": diagnostic }),
    };
    // JSON Value contains only representable strings, integers, arrays and objects.
    let response = serde_json::to_vec(&value).expect("JSON value serialization is infallible");
    if response.len() > output_len {
        return -4;
    }
    let output = unsafe { core::slice::from_raw_parts_mut(output_ptr, output_len) };
    output[..response.len()].copy_from_slice(&response);
    response.len() as i32
}
