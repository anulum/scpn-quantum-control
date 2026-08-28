// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — FIPS 202 SHAKE tests for native ML-DSA

use super::*;

fn decode_hex(value: &str) -> Vec<u8> {
    let (pairs, remainder) = value.as_bytes().as_chunks::<2>();
    assert!(remainder.is_empty(), "hex vector must have an even length");
    pairs
        .iter()
        .map(|pair| {
            let text = std::str::from_utf8(pair).expect("ASCII test vector");
            u8::from_str_radix(text, 16).expect("hex test vector")
        })
        .collect()
}

#[test]
fn shake128_empty_matches_fips_202() {
    let expected = decode_hex(
        "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef263cb1eea9\
         88004b93103cfb0aeefd2a686e01fa4a58e8a3639ca8a1e3f9ae57e2",
    );
    let mut actual = [0u8; 64];
    ShakeReader::new(168, &[b""]).read(&mut actual);
    assert_eq!(actual.as_slice(), expected);
}

#[test]
fn shake256_chunking_matches_fips_202() {
    let expected = decode_hex(
        "483366601360a8771c6863080cc4114d8db44530f8f1e1ee4f94ea37e78b5739d5a15bef\
         186a5386c75744c0527e1faa9f8726e462a12a4feb06bd8801e751e4",
    );
    let actual = shake256::<64>(&[b"a", b"bc"]);
    assert_eq!(actual.as_slice(), expected);
}
