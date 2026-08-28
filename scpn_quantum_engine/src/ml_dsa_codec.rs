// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — FIPS 204 ML-DSA-65 encoding and sampling

use crate::ml_dsa::{rem_q, N, Q};

pub(crate) const D: usize = 13;
pub(crate) const TAU: usize = 49;
pub(crate) const GAMMA1: i64 = 1 << 19;
pub(crate) const GAMMA2: i64 = (Q - 1) / 32;
pub(crate) const K: usize = 6;
pub(crate) const L: usize = 5;
pub(crate) const ETA: i64 = 4;
pub(crate) const BETA: i64 = TAU as i64 * ETA;
pub(crate) const OMEGA: usize = 55;
pub(crate) const C_TILDE_BYTES: usize = 48;
pub(crate) const PUBLIC_KEY_BYTES: usize = 1952;
pub(crate) const SECRET_KEY_BYTES: usize = 4032;
pub(crate) const SIGNATURE_BYTES: usize = 3309;

pub(crate) type Poly = [i64; N];

const ROUND_CONSTANTS: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

const ROTATIONS: [u32; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

fn keccak_f1600(state: &mut [u64; 25]) {
    for round_constant in ROUND_CONSTANTS {
        let columns: [u64; 5] =
            std::array::from_fn(|x| (0..5).fold(0, |value, y| value ^ state[x + 5 * y]));
        let deltas: [u64; 5] =
            std::array::from_fn(|x| columns[(x + 4) % 5] ^ columns[(x + 1) % 5].rotate_left(1));
        for y in 0..5 {
            for x in 0..5 {
                state[x + 5 * y] ^= deltas[x];
            }
        }
        let mut rotated = [0u64; 25];
        for y in 0..5 {
            for x in 0..5 {
                rotated[y + 5 * ((2 * x + 3 * y) % 5)] =
                    state[x + 5 * y].rotate_left(ROTATIONS[x + 5 * y]);
            }
        }
        for y in 0..5 {
            for x in 0..5 {
                state[x + 5 * y] = rotated[x + 5 * y]
                    ^ ((!rotated[(x + 1) % 5 + 5 * y]) & rotated[(x + 2) % 5 + 5 * y]);
            }
        }
        state[0] ^= round_constant;
    }
}

struct ShakeReader {
    state: [u64; 25],
    rate: usize,
    offset: usize,
}

impl ShakeReader {
    fn new(rate: usize, chunks: &[&[u8]]) -> Self {
        let mut reader = Self {
            state: [0; 25],
            rate,
            offset: 0,
        };
        for chunk in chunks {
            reader.absorb(chunk);
        }
        reader.xor_byte(reader.offset, 0x1f);
        reader.xor_byte(rate - 1, 0x80);
        keccak_f1600(&mut reader.state);
        reader.offset = 0;
        reader
    }

    fn absorb(&mut self, input: &[u8]) {
        for byte in input {
            self.xor_byte(self.offset, *byte);
            self.offset += 1;
            if self.offset == self.rate {
                keccak_f1600(&mut self.state);
                self.offset = 0;
            }
        }
    }

    fn xor_byte(&mut self, offset: usize, value: u8) {
        self.state[offset / 8] ^= u64::from(value) << (8 * (offset % 8));
    }

    fn read(&mut self, output: &mut [u8]) {
        for byte in output {
            if self.offset == self.rate {
                keccak_f1600(&mut self.state);
                self.offset = 0;
            }
            *byte = ((self.state[self.offset / 8] >> (8 * (self.offset % 8))) & 0xff) as u8;
            self.offset += 1;
        }
    }
}

pub(crate) fn shake256<const OUTPUT: usize>(chunks: &[&[u8]]) -> [u8; OUTPUT] {
    let mut output = [0u8; OUTPUT];
    ShakeReader::new(136, chunks).read(&mut output);
    output
}

fn bit_pack(poly: &Poly, a: i64, b: i64) -> Vec<u8> {
    let width = (i64::BITS - (a + b).leading_zeros()) as usize;
    pack_values(poly.iter().map(|coefficient| b - coefficient), width)
}

pub(crate) fn simple_bit_pack(poly: &Poly, width: usize) -> Vec<u8> {
    pack_values(poly.iter().copied(), width)
}

fn pack_values(values: impl Iterator<Item = i64>, width: usize) -> Vec<u8> {
    let mut output = vec![0u8; N * width / 8];
    for (index, value) in values.enumerate() {
        let value = value as u64;
        let bit_offset = index * width;
        for bit in 0..width {
            output[(bit_offset + bit) / 8] |=
                (((value >> bit) & 1) as u8) << ((bit_offset + bit) % 8);
        }
    }
    output
}

pub(crate) fn bit_unpack(data: &[u8], a: i64, b: i64) -> Poly {
    let width = (i64::BITS - (a + b).leading_zeros()) as usize;
    let mut output = [0i64; N];
    for (index, coefficient) in output.iter_mut().enumerate() {
        let bit_offset = index * width;
        let mut value = 0i64;
        for bit in 0..width {
            value |=
                i64::from((data[(bit_offset + bit) / 8] >> ((bit_offset + bit) % 8)) & 1) << bit;
        }
        *coefficient = b - value;
    }
    output
}

pub(crate) fn encode_public_key(rho: &[u8; 32], t1: &[Poly]) -> [u8; PUBLIC_KEY_BYTES] {
    let mut output = [0u8; PUBLIC_KEY_BYTES];
    output[..32].copy_from_slice(rho);
    let mut offset = 32;
    for polynomial in t1 {
        let encoded = simple_bit_pack(polynomial, 10);
        output[offset..offset + encoded.len()].copy_from_slice(&encoded);
        offset += encoded.len();
    }
    output
}

pub(crate) fn encode_secret_key(
    rho: &[u8; 32],
    key_seed: &[u8; 32],
    tr: &[u8; 64],
    s1: &[Poly],
    s2: &[Poly],
    t0: &[Poly],
) -> Vec<u8> {
    let mut output = vec![0u8; SECRET_KEY_BYTES];
    output[..32].copy_from_slice(rho);
    output[32..64].copy_from_slice(key_seed);
    output[64..128].copy_from_slice(tr);
    let mut offset = 128;
    for polynomial in s1.iter().chain(s2) {
        let encoded = bit_pack(polynomial, ETA, ETA);
        output[offset..offset + encoded.len()].copy_from_slice(&encoded);
        offset += encoded.len();
    }
    let bound = (1 << (D - 1)) as i64;
    for polynomial in t0 {
        let encoded = bit_pack(polynomial, bound - 1, bound);
        output[offset..offset + encoded.len()].copy_from_slice(&encoded);
        offset += encoded.len();
    }
    output
}

pub(crate) fn encode_signature(
    challenge: &[u8; C_TILDE_BYTES],
    z: &[Poly],
    hints: &[Poly],
) -> [u8; SIGNATURE_BYTES] {
    let mut output = [0u8; SIGNATURE_BYTES];
    output[..C_TILDE_BYTES].copy_from_slice(challenge);
    let mut offset = C_TILDE_BYTES;
    for polynomial in z {
        let encoded = bit_pack(polynomial, GAMMA1 - 1, GAMMA1);
        output[offset..offset + encoded.len()].copy_from_slice(&encoded);
        offset += encoded.len();
    }
    let mut hint_index = 0;
    for (row, polynomial) in hints.iter().enumerate() {
        for (index, value) in polynomial.iter().enumerate() {
            if *value != 0 {
                output[offset + hint_index] = index as u8;
                hint_index += 1;
            }
        }
        output[offset + OMEGA + row] = hint_index as u8;
    }
    output
}

pub(crate) fn sample_in_ball(seed: &[u8]) -> Poly {
    let mut reader = ShakeReader::new(136, &[seed]);
    let mut sign_bytes = [0u8; 8];
    reader.read(&mut sign_bytes);
    let mut signs = u64::from_le_bytes(sign_bytes);
    let mut output = [0i64; N];
    for index in N - TAU..N {
        let selected = loop {
            let mut byte = [0u8; 1];
            reader.read(&mut byte);
            if usize::from(byte[0]) <= index {
                break usize::from(byte[0]);
            }
        };
        output[index] = output[selected];
        output[selected] = 1 - 2 * (signs & 1) as i64;
        signs >>= 1;
    }
    output
}

fn rejection_ntt_poly(rho: &[u8; 32], column: u8, row: u8) -> Poly {
    let coordinates = [column, row];
    let mut reader = ShakeReader::new(168, &[rho, &coordinates]);
    let mut output = [0i64; N];
    let mut accepted = 0;
    while accepted < N {
        let mut bytes = [0u8; 3];
        reader.read(&mut bytes);
        let value =
            i64::from(bytes[0]) | (i64::from(bytes[1]) << 8) | (i64::from(bytes[2] & 0x7f) << 16);
        if value < Q {
            output[accepted] = value;
            accepted += 1;
        }
    }
    output
}

fn rejection_bounded_poly(rho: &[u8], nonce: u16) -> Poly {
    let nonce_bytes = nonce.to_le_bytes();
    let mut reader = ShakeReader::new(136, &[rho, &nonce_bytes]);
    let mut output = [0i64; N];
    let mut accepted = 0;
    while accepted < N {
        let mut byte = [0u8; 1];
        reader.read(&mut byte);
        for half in [byte[0] & 0x0f, byte[0] >> 4] {
            if half < 9 {
                output[accepted] = ETA - i64::from(half);
                accepted += 1;
                if accepted == N {
                    break;
                }
            }
        }
    }
    output
}

pub(crate) fn expand_matrix(rho: &[u8; 32]) -> Vec<Vec<Poly>> {
    (0..K)
        .map(|row| {
            (0..L)
                .map(|column| rejection_ntt_poly(rho, column as u8, row as u8))
                .collect()
        })
        .collect()
}

pub(crate) fn expand_secret(rho: &[u8; 64]) -> (Vec<Poly>, Vec<Poly>) {
    let s1 = (0..L)
        .map(|nonce| rejection_bounded_poly(rho, nonce as u16))
        .collect();
    let s2 = (0..K)
        .map(|nonce| rejection_bounded_poly(rho, (nonce + L) as u16))
        .collect();
    (s1, s2)
}

pub(crate) fn expand_mask(rho: &[u8; 64], nonce: usize) -> Vec<Poly> {
    (0..L)
        .map(|row| {
            let encoded = shake256::<640>(&[rho, &((nonce + row) as u16).to_le_bytes()]);
            bit_unpack(&encoded, GAMMA1 - 1, GAMMA1).map(rem_q)
        })
        .collect()
}

#[cfg(test)]
mod tests;
