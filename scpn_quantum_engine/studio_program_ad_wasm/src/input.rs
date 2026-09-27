// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — borrowed WASM replay input validation

struct ReplayInputLayout<'a> {
    ir: &'a str,
    count: usize,
    inputs_offset: usize,
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut raw = [0_u8;4];
    raw.copy_from_slice(&bytes[offset..offset+4]);
    u32::from_le_bytes(raw)
}

fn read_f64(bytes: &[u8], offset: usize) -> f64 {
    let mut raw = [0_u8;8];
    raw.copy_from_slice(&bytes[offset..offset+8]);
    f64::from_le_bytes(raw)
}

fn replay_input_layout(bytes: &[u8]) -> Result<ReplayInputLayout<'_>,ProgramAdStatus> {
    if bytes.len() < 4 { return Err(ProgramAdStatus::InvalidLength); }
    let ir_len = read_u32(bytes,0) as usize;
    if ir_len == 0 || ir_len > MAX_PROGRAM_AD_REPLAY_IR_BYTES {
        return Err(ProgramAdStatus::InvalidLength);
    }
    let after_ir = 4usize.checked_add(ir_len).ok_or(ProgramAdStatus::InvalidLength)?;
    let inputs_offset = after_ir.checked_add(4).ok_or(ProgramAdStatus::InvalidLength)?;
    if bytes.len() < inputs_offset { return Err(ProgramAdStatus::InvalidLength); }
    let ir = core::str::from_utf8(&bytes[4..after_ir]).map_err(|_|ProgramAdStatus::InvalidUtf8)?;
    let count = read_u32(bytes,after_ir) as usize;
    if count > MAX_PROGRAM_AD_REPLAY_INPUTS { return Err(ProgramAdStatus::InvalidLength); }
    let expected = count.checked_mul(8).and_then(|length|inputs_offset.checked_add(length))
        .ok_or(ProgramAdStatus::InvalidLength)?;
    if bytes.len() != expected { return Err(ProgramAdStatus::InvalidLength); }
    for index in 0..count {
        if index % 256 == 0 {
            scpn_quantum_program_ad_replay::program_ad_lifecycle::replay_checkpoint()
                .map_err(|_|ProgramAdStatus::ReplayError)?;
        }
        if !read_f64(bytes,inputs_offset+index*8).is_finite() {
            return Err(ProgramAdStatus::NonFiniteInput);
        }
    }
    Ok(ReplayInputLayout {ir,count,inputs_offset})
}

/// Decode the canonical little-endian replay input after complete borrowed validation.
///
/// Layout: `u32 ir_len | ir_bytes (UTF-8 effect-IR JSON) | u32 n_inputs |
/// n_inputs * f64`. Invalid layouts and non-finite values refuse before owned
/// input copies; allocation refusal returns `ReplayError` without a fallback.
pub fn parse_replay_input(bytes: &[u8]) -> Result<(String,Vec<f64>),ProgramAdStatus> {
    let layout = replay_input_layout(bytes)?;
    let mut ir = String::new();
    ir.try_reserve_exact(layout.ir.len()).map_err(|_|ProgramAdStatus::ReplayError)?;
    ir.push_str(layout.ir);
    let mut inputs = Vec::new();
    inputs.try_reserve_exact(layout.count).map_err(|_|ProgramAdStatus::ReplayError)?;
    for index in 0..layout.count {
        if index % 256 == 0 {
            scpn_quantum_program_ad_replay::program_ad_lifecycle::replay_checkpoint()
                .map_err(|_|ProgramAdStatus::ReplayError)?;
        }
        inputs.push(read_f64(bytes,layout.inputs_offset+index*8));
    }
    Ok((ir,inputs))
}
