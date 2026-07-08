// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Studio WASM verifier kernel

//! WASM-safe recompute kernel for QUANTUM Studio compile claims.
//!
//! The exported ABI consumes the canonical binary payload produced by
//! `scpn_quantum_control.studio.recompute_kernel` and writes a 32-byte SHA-256
//! digest. The digest covers the structural XY compile terms, not QPU execution
//! or floating measurement results.

use sha2::{Digest, Sha256};

pub mod kuramoto;

const SCHEMA_TAG: &[u8] = b"scpn.quantum.xy_compile.v1\0";
const INPUT_VERSION: u32 = 1;
const HEADER_LEN: usize = 24;
const DIGEST_LEN: usize = 32;
const MAX_QUBITS: u32 = 64;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[repr(i32)]
pub enum KernelStatus {
    Ok = 0,
    NullPointer = -1,
    InvalidLength = -2,
    InvalidVersion = -3,
    InvalidQubitCount = -4,
    InvalidFloat = -5,
    InvalidTrotter = -6,
}

impl From<KernelStatus> for i32 {
    fn from(value: KernelStatus) -> Self {
        value as i32
    }
}

#[derive(Debug, Clone)]
pub struct CompileInput {
    pub n_qubits: u32,
    pub time: f64,
    pub trotter_steps: u32,
    pub trotter_order: u32,
    pub k_nm: Vec<f64>,
    pub omega: Vec<f64>,
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

fn push_u32(hasher: &mut Sha256, value: u32) {
    hasher.update(value.to_le_bytes());
}

fn push_f64(hasher: &mut Sha256, value: f64) {
    hasher.update(value.to_le_bytes());
}

pub fn parse_compile_input(bytes: &[u8]) -> Result<CompileInput, KernelStatus> {
    if bytes.len() < HEADER_LEN {
        return Err(KernelStatus::InvalidLength);
    }
    let version = read_u32(bytes, 0);
    if version != INPUT_VERSION {
        return Err(KernelStatus::InvalidVersion);
    }
    let n_qubits = read_u32(bytes, 4);
    if n_qubits == 0 || n_qubits > MAX_QUBITS {
        return Err(KernelStatus::InvalidQubitCount);
    }
    let time = read_f64(bytes, 8);
    let trotter_steps = read_u32(bytes, 16);
    let trotter_order = read_u32(bytes, 20);
    if !time.is_finite() {
        return Err(KernelStatus::InvalidFloat);
    }
    if trotter_steps == 0 || trotter_order == 0 {
        return Err(KernelStatus::InvalidTrotter);
    }

    let n = n_qubits as usize;
    let values_len = n
        .checked_mul(n)
        .and_then(|value| value.checked_add(n))
        .ok_or(KernelStatus::InvalidLength)?;
    let expected_len = HEADER_LEN
        .checked_add(
            values_len
                .checked_mul(8)
                .ok_or(KernelStatus::InvalidLength)?,
        )
        .ok_or(KernelStatus::InvalidLength)?;
    if bytes.len() != expected_len {
        return Err(KernelStatus::InvalidLength);
    }

    let mut offset = HEADER_LEN;
    let mut k_nm = Vec::with_capacity(n * n);
    for _ in 0..(n * n) {
        let value = read_f64(bytes, offset);
        if !value.is_finite() {
            return Err(KernelStatus::InvalidFloat);
        }
        k_nm.push(value);
        offset += 8;
    }
    let mut omega = Vec::with_capacity(n);
    for _ in 0..n {
        let value = read_f64(bytes, offset);
        if !value.is_finite() {
            return Err(KernelStatus::InvalidFloat);
        }
        omega.push(value);
        offset += 8;
    }

    Ok(CompileInput {
        n_qubits,
        time,
        trotter_steps,
        trotter_order,
        k_nm,
        omega,
    })
}

pub fn xy_compile_digest(input: &CompileInput) -> Result<[u8; DIGEST_LEN], KernelStatus> {
    let n = input.n_qubits as usize;
    if n == 0 || input.n_qubits > MAX_QUBITS {
        return Err(KernelStatus::InvalidQubitCount);
    }
    if input.k_nm.len() != n * n || input.omega.len() != n {
        return Err(KernelStatus::InvalidLength);
    }
    if !input.time.is_finite() {
        return Err(KernelStatus::InvalidFloat);
    }
    if input.trotter_steps == 0 || input.trotter_order == 0 {
        return Err(KernelStatus::InvalidTrotter);
    }
    if input
        .k_nm
        .iter()
        .chain(input.omega.iter())
        .any(|value| !value.is_finite())
    {
        return Err(KernelStatus::InvalidFloat);
    }

    let dt = input.time / f64::from(input.trotter_steps);
    let mut hasher = Sha256::new();
    hasher.update(SCHEMA_TAG);
    push_u32(&mut hasher, input.n_qubits);
    push_f64(&mut hasher, input.time);
    push_u32(&mut hasher, input.trotter_steps);
    push_u32(&mut hasher, input.trotter_order);
    push_f64(&mut hasher, dt);

    for (qubit, omega) in input.omega.iter().enumerate() {
        push_u32(&mut hasher, qubit as u32);
        push_f64(&mut hasher, *omega);
        push_f64(&mut hasher, omega * dt);
    }

    for source in 0..n {
        for target in (source + 1)..n {
            let forward = input.k_nm[source * n + target];
            let reverse = input.k_nm[target * n + source];
            if forward == 0.0 && reverse == 0.0 {
                continue;
            }
            let coupling = 0.5 * (forward + reverse);
            if coupling == 0.0 {
                continue;
            }
            let angle = coupling * dt;
            push_u32(&mut hasher, source as u32);
            push_u32(&mut hasher, target as u32);
            push_f64(&mut hasher, forward);
            push_f64(&mut hasher, reverse);
            push_f64(&mut hasher, coupling);
            push_f64(&mut hasher, angle);
            push_f64(&mut hasher, angle);
        }
    }

    Ok(hasher.finalize().into())
}

/// Allocate `len` bytes inside the module's linear memory for host input.
///
/// Returns a null pointer for zero-length or failed allocations; the host
/// must treat null as fail-closed and must release every successful
/// allocation with [`scpn_free`] using the same length.
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
/// `ptr` must come from `scpn_alloc(len)` with the identical `len` and must
/// not be used afterwards. Null pointers and zero lengths are ignored.
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

/// Compute the XY compile digest for a canonical byte payload.
///
/// # Safety
///
/// `input_ptr` must point to `input_len` readable bytes, and `output_ptr` must
/// point to at least 32 writable bytes. The function checks null pointers and
/// returns a negative status code on invalid payloads.
#[no_mangle]
pub unsafe extern "C" fn scpn_xy_compile_digest(
    input_ptr: *const u8,
    input_len: usize,
    output_ptr: *mut u8,
) -> i32 {
    if input_ptr.is_null() || output_ptr.is_null() {
        return KernelStatus::NullPointer.into();
    }
    let input = unsafe { core::slice::from_raw_parts(input_ptr, input_len) };
    let output = unsafe { core::slice::from_raw_parts_mut(output_ptr, DIGEST_LEN) };
    match parse_compile_input(input).and_then(|parsed| xy_compile_digest(&parsed)) {
        Ok(digest) => {
            output.copy_from_slice(&digest);
            KernelStatus::Ok.into()
        }
        Err(status) => status.into(),
    }
}

/// Report the fail-closed oscillator-count boundary for the Play panel.
///
/// The panel reads this to display the declared `N` limit and refuse larger
/// requests before they reach the kernel.
#[no_mangle]
pub extern "C" fn scpn_kuramoto_max_oscillators() -> u32 {
    kuramoto::MAX_OSCILLATORS as u32
}

/// Report the fail-closed step-count boundary for the Play panel.
#[no_mangle]
pub extern "C" fn scpn_kuramoto_max_steps() -> u32 {
    kuramoto::MAX_STEPS as u32
}

/// Integrate a Kuramoto request and write the `R(t)` trajectory + final phases.
///
/// The output is `(steps + 1 + n)` little-endian `f64` values: `steps + 1`
/// order-parameter samples followed by the `n` final phases. `output_len` is
/// the byte length the host allocated and must equal that element count times
/// eight, else the call fails closed.
///
/// # Safety
///
/// `input_ptr` must point to `input_len` readable bytes and `output_ptr` to
/// `output_len` writable bytes. Null pointers and any malformed payload return
/// a negative status code without writing output.
#[no_mangle]
pub unsafe extern "C" fn scpn_kuramoto_simulate(
    input_ptr: *const u8,
    input_len: usize,
    output_ptr: *mut u8,
    output_len: usize,
) -> i32 {
    if input_ptr.is_null() || output_ptr.is_null() {
        return kuramoto::KuramotoStatus::NullPointer.into();
    }
    let input_bytes = unsafe { core::slice::from_raw_parts(input_ptr, input_len) };
    let parsed = match kuramoto::parse_kuramoto_input(input_bytes) {
        Ok(parsed) => parsed,
        Err(status) => return status.into(),
    };
    let Some(needed_bytes) = kuramoto::output_len(&parsed).checked_mul(8) else {
        return kuramoto::KuramotoStatus::InvalidLength.into();
    };
    if output_len != needed_bytes {
        return kuramoto::KuramotoStatus::OutputMismatch.into();
    }
    let series = kuramoto::simulate(&parsed);
    let output = unsafe { core::slice::from_raw_parts_mut(output_ptr, output_len) };
    for (index, value) in series.iter().enumerate() {
        output[index * 8..index * 8 + 8].copy_from_slice(&value.to_le_bytes());
    }
    kuramoto::KuramotoStatus::Ok.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_payload() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(INPUT_VERSION.to_le_bytes());
        bytes.extend(2_u32.to_le_bytes());
        bytes.extend(0.1_f64.to_le_bytes());
        bytes.extend(1_u32.to_le_bytes());
        bytes.extend(1_u32.to_le_bytes());
        for value in [0.0_f64, 0.25, 0.25, 0.0] {
            bytes.extend(value.to_le_bytes());
        }
        for value in [1.0_f64, -0.5] {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn digest_is_deterministic() {
        let parsed = parse_compile_input(&sample_payload()).expect("valid input");
        let first = xy_compile_digest(&parsed).expect("digest");
        let second = xy_compile_digest(&parsed).expect("digest");
        assert_eq!(first, second);
    }

    #[test]
    fn digest_changes_when_coupling_changes() {
        let mut left = sample_payload();
        let mut right = sample_payload();
        let offset = HEADER_LEN + 8;
        right[offset..offset + 8].copy_from_slice(&0.3_f64.to_le_bytes());
        let left_digest = xy_compile_digest(&parse_compile_input(&left).expect("valid left"))
            .expect("left digest");
        let right_digest = xy_compile_digest(&parse_compile_input(&right).expect("valid right"))
            .expect("right digest");
        assert_ne!(left_digest, right_digest);
        left.truncate(HEADER_LEN);
        assert_eq!(
            parse_compile_input(&left).expect_err("bad length"),
            KernelStatus::InvalidLength
        );
    }

    #[test]
    fn alloc_round_trip_carries_host_bytes() {
        let payload = sample_payload();
        let ptr = scpn_alloc(payload.len());
        assert!(!ptr.is_null());
        let mut digest = [0_u8; DIGEST_LEN];
        unsafe {
            core::ptr::copy_nonoverlapping(payload.as_ptr(), ptr, payload.len());
            let status = scpn_xy_compile_digest(ptr, payload.len(), digest.as_mut_ptr());
            assert_eq!(status, i32::from(KernelStatus::Ok));
            scpn_free(ptr, payload.len());
        }
        let parsed = parse_compile_input(&payload).expect("valid input");
        let expected = xy_compile_digest(&parsed).expect("digest");
        assert_eq!(digest, expected);
    }

    #[test]
    fn alloc_fails_closed_on_zero_and_free_ignores_null() {
        assert!(scpn_alloc(0).is_null());
        unsafe {
            scpn_free(core::ptr::null_mut(), 8);
            scpn_free(scpn_alloc(8), 0);
        }
    }

    #[test]
    fn parser_rejects_bad_version_and_nan() {
        let mut bad_version = sample_payload();
        bad_version[0..4].copy_from_slice(&99_u32.to_le_bytes());
        assert_eq!(
            parse_compile_input(&bad_version).expect_err("bad version"),
            KernelStatus::InvalidVersion
        );

        let mut bad_float = sample_payload();
        bad_float[8..16].copy_from_slice(&f64::NAN.to_le_bytes());
        assert_eq!(
            parse_compile_input(&bad_float).expect_err("bad float"),
            KernelStatus::InvalidFloat
        );
    }

    fn kuramoto_mean_field_payload() -> Vec<u8> {
        let n = 5_usize;
        let steps = 120_u32;
        let mut bytes = Vec::new();
        bytes.extend(kuramoto::KURAMOTO_INPUT_VERSION.to_le_bytes());
        bytes.extend(0_u32.to_le_bytes()); // mean-field
        bytes.extend((n as u32).to_le_bytes());
        bytes.extend(steps.to_le_bytes());
        bytes.extend(0.01_f64.to_le_bytes()); // dt
        bytes.extend(1.6_f64.to_le_bytes()); // coupling
        for i in 0..n {
            bytes.extend((0.1 * i as f64).to_le_bytes()); // omega
        }
        for i in 0..n {
            bytes.extend((0.25 * i as f64).to_le_bytes()); // theta0
        }
        bytes
    }

    #[test]
    fn kuramoto_boundaries_are_exposed() {
        assert_eq!(
            scpn_kuramoto_max_oscillators(),
            kuramoto::MAX_OSCILLATORS as u32
        );
        assert_eq!(scpn_kuramoto_max_steps(), kuramoto::MAX_STEPS as u32);
    }

    #[test]
    fn kuramoto_ffi_round_trip_matches_the_reference() {
        let payload = kuramoto_mean_field_payload();
        let parsed = kuramoto::parse_kuramoto_input(&payload).expect("valid payload");
        let expected = kuramoto::simulate(&parsed);
        let output_bytes = expected.len() * 8;

        let input_ptr = scpn_alloc(payload.len());
        let output_ptr = scpn_alloc(output_bytes);
        assert!(!input_ptr.is_null() && !output_ptr.is_null());
        let mut produced = vec![0.0_f64; expected.len()];
        unsafe {
            core::ptr::copy_nonoverlapping(payload.as_ptr(), input_ptr, payload.len());
            let status = scpn_kuramoto_simulate(input_ptr, payload.len(), output_ptr, output_bytes);
            assert_eq!(status, i32::from(kuramoto::KuramotoStatus::Ok));
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
    fn kuramoto_ffi_fails_closed() {
        let payload = kuramoto_mean_field_payload();
        let mut sink = [0_u8; 8];
        unsafe {
            // null pointers
            assert_eq!(
                scpn_kuramoto_simulate(core::ptr::null(), 0, sink.as_mut_ptr(), sink.len()),
                i32::from(kuramoto::KuramotoStatus::NullPointer)
            );
            // wrong output length
            assert_eq!(
                scpn_kuramoto_simulate(payload.as_ptr(), payload.len(), sink.as_mut_ptr(), 8),
                i32::from(kuramoto::KuramotoStatus::OutputMismatch)
            );
            // malformed input propagates the parser status
            let mut bad = payload.clone();
            bad[0..4].copy_from_slice(&99_u32.to_le_bytes());
            assert_eq!(
                scpn_kuramoto_simulate(bad.as_ptr(), bad.len(), sink.as_mut_ptr(), sink.len()),
                i32::from(kuramoto::KuramotoStatus::InvalidVersion)
            );
        }
    }
}
