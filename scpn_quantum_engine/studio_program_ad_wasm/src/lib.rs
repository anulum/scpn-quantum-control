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

#![deny(missing_docs)]

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;

/// Maximum UTF-8 effect-IR size shared with the Python artifact packer.
pub const MAX_PROGRAM_AD_REPLAY_IR_BYTES: usize = 1_048_576;
/// Maximum scalar-input arity shared with the Python artifact packer.
pub const MAX_PROGRAM_AD_REPLAY_INPUTS: usize = 4_096;

/// Product ceiling for cumulative declared replay storage, including input/output copies.
///
/// This 64 MiB policy is a requested allocation ceiling, not observed browser
/// capacity. Fallible allocation and parent admission policies still apply.
pub const MAX_PROGRAM_AD_REPLAY_MEMORY_BYTES: usize = 64 * 1024 * 1024;

/// Fail-closed status codes for the program-AD replay FFI.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[repr(i32)]
pub enum ProgramAdStatus {
    /// The replay completed; the output buffer holds the scalar value followed
    /// by the reverse-mode gradient.
    Ok = 0,
    /// The input or the output pointer was null.
    NullPointer = -1,
    /// The payload does not match the declared layout: a length header of zero
    /// or above its bound, an overflow while computing the section bounds, or a
    /// total size other than exactly the one the headers describe.
    InvalidLength = -2,
    /// The effect-IR section is not valid UTF-8.
    InvalidUtf8 = -3,
    /// The bounded interpreter rejected the effect-IR.
    ReplayError = -4,
    /// The interpreter ran and reported the program lies outside the bounded
    /// set, so no gradient is produced rather than an unsupported one.
    Unsupported = -5,
    /// The caller's output buffer is not exactly eight bytes per returned
    /// scalar, including the case where that size overflows.
    OutputMismatch = -6,
    /// A scalar input was NaN or infinite; the replay fails closed rather than
    /// propagating it into the gradient.
    NonFiniteInput = -7,
}

impl From<ProgramAdStatus> for i32 {
    fn from(value: ProgramAdStatus) -> Self {
        value as i32
    }
}

include!("input.rs");

/// Replay with the product's declared-memory ceiling, returning `[value, gradient…]`.
pub fn replay_value_and_gradient(bytes: &[u8]) -> Result<Vec<f64>, ProgramAdStatus> {
    replay_value_and_gradient_with_memory_budget(bytes, MAX_PROGRAM_AD_REPLAY_MEMORY_BYTES)
}

/// Replay under a positive byte ceiling no larger than the product policy.
///
/// Input copies, returned values and additional parser/numerical declarations
/// share one cumulative charge for this invocation. Nested policies cannot waive
/// a parent refusal. The cap does not reserve browser pages or qualify allocator
/// overhead; owned state is discarded on every return. A budget refusal yields
/// `ReplayError` without returning partial numerical output.
pub fn replay_value_and_gradient_with_memory_budget(
    bytes: &[u8],
    max_bytes: usize,
) -> Result<Vec<f64>, ProgramAdStatus> {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_memory_admission, with_replay_metadata_admission,
    };
    use std::{cell::Cell, rc::Rc};

    if max_bytes == 0 || max_bytes > MAX_PROGRAM_AD_REPLAY_MEMORY_BYTES {
        return Err(ProgramAdStatus::ReplayError);
    }
    let layout = replay_input_layout(bytes)?;
    let output_count = layout.count.checked_add(1).ok_or(ProgramAdStatus::OutputMismatch)?;
    let output_bytes = output_count.checked_mul(8).ok_or(ProgramAdStatus::OutputMismatch)?;
    let initial = layout.count.checked_mul(8)
        .and_then(|size| size.checked_add(layout.ir.len()))
        .and_then(|size| size.checked_add(output_bytes))
        .ok_or(ProgramAdStatus::ReplayError)?;
    if initial > max_bytes { return Err(ProgramAdStatus::ReplayError); }
    let charged = Rc::new(Cell::new(initial));
    let refused = Rc::new(Cell::new(false));
    let numeric_charge = Rc::clone(&charged);
    let metadata_charge = Rc::clone(&charged);
    let numeric_refused = Rc::clone(&refused);
    let metadata_refused = Rc::clone(&refused);
    let admit = move |charge: &Cell<usize>, refused: &Cell<bool>, bytes: usize| {
        match charge.get().checked_add(bytes).filter(|size| *size <= max_bytes) {
            Some(total) => { charge.set(total); Ok(()) }
            None => { refused.set(true); Err("WASM replay memory budget exceeded".to_owned()) }
        }
    };
    let result = with_replay_metadata_admission(
        move |bytes| admit(&metadata_charge, &metadata_refused, bytes),
        || with_replay_memory_admission(
            move |request| {
                let bytes = request.total_bytes()?;
                admit(&numeric_charge, &numeric_refused, bytes)
            },
            || {
                let (ir, inputs) = parse_replay_input(bytes)?;
                let result = interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs)
                    .map_err(|_| ProgramAdStatus::ReplayError)?;
                if !result.supported { return Err(ProgramAdStatus::Unsupported); }
                let value = result.value.ok_or(ProgramAdStatus::Unsupported)?;
                if result.gradient.len() != inputs.len() {
                    return Err(ProgramAdStatus::OutputMismatch);
                }
                let mut out = Vec::new();
                out.try_reserve_exact(output_count).map_err(|_| ProgramAdStatus::ReplayError)?;
                out.push(value);
                out.extend_from_slice(&result.gradient);
                Ok(out)
            },
        ),
    );
    if refused.get() { Err(ProgramAdStatus::ReplayError) } else { result }
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
/// writing output. The bounded input envelope and exact output byte count are
/// checked before IR copies or replay. The output pointer must provide writable
/// storage for an accepted byte count; rejected lengths do not access it.
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
    const MAX_INPUT_BYTES: usize = 8 + MAX_PROGRAM_AD_REPLAY_IR_BYTES + MAX_PROGRAM_AD_REPLAY_INPUTS * 8;
    if input_len > MAX_INPUT_BYTES || input_len > isize::MAX as usize {
        return ProgramAdStatus::InvalidLength.into();
    }
    let input = unsafe { core::slice::from_raw_parts(input_ptr, input_len) };
    let layout = match replay_input_layout(input) {
        Ok(layout) => layout,
        Err(status) => return status.into(),
    };
    let Some(expected_output) = layout.count.checked_add(1).and_then(|count|count.checked_mul(8)) else {
        return ProgramAdStatus::OutputMismatch.into();
    };
    if output_len != expected_output || output_len > isize::MAX as usize {
        return ProgramAdStatus::OutputMismatch.into();
    }
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
mod tests;
