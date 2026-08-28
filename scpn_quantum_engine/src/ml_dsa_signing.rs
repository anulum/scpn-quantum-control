// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — zeroizing FIPS 204 ML-DSA-65 signing key

//! Native ML-DSA-65 key generation and signing with secret buffers zeroized on
//! drop. The PyO3 class never exports the expanded secret key to Python.

use zeroize::{Zeroize, Zeroizing};

use crate::{
    ml_dsa::{ml_dsa_intt_core, ml_dsa_ntt_core, rem_q, N, Q},
    ml_dsa_codec::{
        bit_unpack, encode_public_key, encode_secret_key, encode_signature, expand_mask,
        expand_matrix, expand_secret, sample_in_ball, shake256, simple_bit_pack, Poly, BETA,
        C_TILDE_BYTES, D, ETA, GAMMA1, GAMMA2, K, L, OMEGA, PUBLIC_KEY_BYTES, SIGNATURE_BYTES,
    },
};

fn poly_add(left: &Poly, right: &Poly) -> Poly {
    std::array::from_fn(|index| rem_q(left[index] + right[index]))
}

fn ntt_mul(left: &Poly, right: &Poly) -> Poly {
    std::array::from_fn(|index| rem_q(left[index] * right[index]))
}

fn centred(value: i64, modulus: i64) -> i64 {
    let mut reduced = value.rem_euclid(modulus);
    if reduced > modulus / 2 {
        reduced -= modulus;
    }
    reduced
}

fn power2round(value: i64) -> (i64, i64) {
    let reduced = rem_q(value);
    let high = (reduced + (1 << (D - 1)) - 1) >> D;
    (high, reduced - (high << D))
}

fn decompose(value: i64) -> (i64, i64) {
    let reduced = rem_q(value);
    let mut low = centred(reduced, 2 * GAMMA2);
    let high = if reduced - low == Q - 1 {
        low -= 1;
        0
    } else {
        (reduced - low) / (2 * GAMMA2)
    };
    (high, low)
}

fn high_bits(value: i64) -> i64 {
    decompose(value).0
}

fn low_bits(value: i64) -> i64 {
    decompose(value).1
}

fn make_hint(change: i64, value: i64) -> i64 {
    i64::from(high_bits(value) != high_bits(value + change))
}

fn centred_norm(polynomial: &Poly) -> i64 {
    polynomial
        .iter()
        .map(|coefficient| centred(*coefficient, Q).abs())
        .max()
        .unwrap_or(0)
}

struct DecodedSecretKey {
    rho: [u8; 32],
    key_seed: [u8; 32],
    tr: [u8; 64],
    s1: Vec<Poly>,
    s2: Vec<Poly>,
    t0: Vec<Poly>,
}

impl Zeroize for DecodedSecretKey {
    fn zeroize(&mut self) {
        self.rho.zeroize();
        self.key_seed.zeroize();
        self.tr.zeroize();
        self.s1.zeroize();
        self.s2.zeroize();
        self.t0.zeroize();
    }
}

fn decode_secret_key(key: &[u8]) -> DecodedSecretKey {
    let rho = key[..32].try_into().expect("fixed ML-DSA rho length");
    let mut key_seed = key[32..64]
        .try_into()
        .expect("fixed ML-DSA key-seed length");
    let mut tr = key[64..128].try_into().expect("fixed ML-DSA tr length");
    let mut offset = 128;
    let mut read_polynomial = |bytes: usize, a: i64, b: i64| {
        let polynomial = bit_unpack(&key[offset..offset + bytes], a, b).map(rem_q);
        offset += bytes;
        polynomial
    };
    let s1 = (0..L).map(|_| read_polynomial(128, ETA, ETA)).collect();
    let s2 = (0..K).map(|_| read_polynomial(128, ETA, ETA)).collect();
    let bound = (1 << (D - 1)) as i64;
    let t0 = (0..K)
        .map(|_| read_polynomial(416, bound - 1, bound))
        .collect();
    let material = DecodedSecretKey {
        rho,
        key_seed,
        tr,
        s1,
        s2,
        t0,
    };
    key_seed.zeroize();
    tr.zeroize();
    material
}

pub(crate) fn keygen(seed: &[u8; 32]) -> ([u8; PUBLIC_KEY_BYTES], Vec<u8>) {
    let expanded = Zeroizing::new(shake256::<128>(&[seed, &[K as u8, L as u8]]));
    let rho: [u8; 32] = expanded[..32].try_into().expect("fixed rho length");
    let rho_prime = Zeroizing::new(expanded[32..96].try_into().expect("fixed rho-prime length"));
    let key_seed = Zeroizing::new(expanded[96..128].try_into().expect("fixed key-seed length"));
    let matrix = expand_matrix(&rho);
    let (s1_value, s2_value) = expand_secret(&rho_prime);
    let s1 = Zeroizing::new(s1_value);
    let s2 = Zeroizing::new(s2_value);
    let s1_hat = Zeroizing::new(s1.iter().copied().map(ml_dsa_ntt_core).collect::<Vec<_>>());
    let mut t1 = Vec::with_capacity(K);
    let mut t0 = Zeroizing::new(Vec::with_capacity(K));
    for row in 0..K {
        let mut accumulator = Zeroizing::new([0i64; N]);
        for column in 0..L {
            *accumulator = poly_add(
                &accumulator,
                &ntt_mul(&matrix[row][column], &s1_hat[column]),
            );
        }
        let transformed = Zeroizing::new(poly_add(&ml_dsa_intt_core(*accumulator), &s2[row]));
        let mut high = [0i64; N];
        let mut low = [0i64; N];
        for index in 0..N {
            (high[index], low[index]) = power2round(transformed[index]);
        }
        t1.push(high);
        t0.push(low);
        low.zeroize();
    }
    let public_key = encode_public_key(&rho, &t1);
    let tr = Zeroizing::new(shake256::<64>(&[&public_key]));
    let secret_key = encode_secret_key(&rho, &key_seed, &tr, &s1, &s2, &t0);
    (public_key, secret_key)
}

pub(crate) fn sign_internal(
    secret_key: &[u8],
    message: &[u8],
    context: &[u8],
) -> Result<[u8; SIGNATURE_BYTES], &'static str> {
    let material = Zeroizing::new(decode_secret_key(secret_key));
    let matrix = expand_matrix(&material.rho);
    let s1_hat = Zeroizing::new(
        material
            .s1
            .iter()
            .copied()
            .map(ml_dsa_ntt_core)
            .collect::<Vec<_>>(),
    );
    let s2_hat = Zeroizing::new(
        material
            .s2
            .iter()
            .copied()
            .map(ml_dsa_ntt_core)
            .collect::<Vec<_>>(),
    );
    let t0_hat = Zeroizing::new(
        material
            .t0
            .iter()
            .copied()
            .map(ml_dsa_ntt_core)
            .collect::<Vec<_>>(),
    );
    let prefix = [0, context.len() as u8];
    let mu = Zeroizing::new(shake256::<64>(&[&material.tr, &prefix, context, message]));
    let randomness = Zeroizing::new([0u8; 32]);
    let mask_seed = Zeroizing::new(shake256::<64>(&[
        &material.key_seed,
        randomness.as_ref(),
        mu.as_ref(),
    ]));

    let mut nonce = 0usize;
    loop {
        let y = Zeroizing::new(expand_mask(&mask_seed, nonce));
        let y_hat = Zeroizing::new(y.iter().copied().map(ml_dsa_ntt_core).collect::<Vec<_>>());
        let mut w = Zeroizing::new(Vec::with_capacity(K));
        for matrix_row in matrix.iter().take(K) {
            let mut accumulator = Zeroizing::new([0i64; N]);
            for column in 0..L {
                *accumulator =
                    poly_add(&accumulator, &ntt_mul(&matrix_row[column], &y_hat[column]));
            }
            w.push(ml_dsa_intt_core(*accumulator));
        }
        let w1 = Zeroizing::new(
            w.iter()
                .map(|polynomial| polynomial.map(high_bits))
                .collect::<Vec<_>>(),
        );
        let mut encoded_w1 = Zeroizing::new(Vec::with_capacity(K * N / 2));
        for polynomial in w1.iter() {
            encoded_w1.extend(simple_bit_pack(polynomial, 4));
        }
        let challenge = shake256::<C_TILDE_BYTES>(&[mu.as_ref(), encoded_w1.as_ref()]);
        let challenge_hat = Zeroizing::new(ml_dsa_ntt_core(sample_in_ball(&challenge)));
        let cs1 = Zeroizing::new(
            s1_hat
                .iter()
                .map(|polynomial| ml_dsa_intt_core(ntt_mul(&challenge_hat, polynomial)))
                .collect::<Vec<_>>(),
        );
        let cs2 = Zeroizing::new(
            s2_hat
                .iter()
                .map(|polynomial| ml_dsa_intt_core(ntt_mul(&challenge_hat, polynomial)))
                .collect::<Vec<_>>(),
        );
        let z = Zeroizing::new(
            y.iter()
                .zip(cs1.iter())
                .map(|(left, right)| poly_add(left, right))
                .collect::<Vec<_>>(),
        );
        let residual = Zeroizing::new(
            (0..K)
                .map(|row| {
                    std::array::from_fn(|index| low_bits(rem_q(w[row][index] - cs2[row][index])))
                })
                .collect::<Vec<Poly>>(),
        );
        if z.iter().map(centred_norm).max().unwrap_or(0) >= GAMMA1 - BETA
            || residual
                .iter()
                .flat_map(|polynomial| polynomial.iter())
                .map(|value| value.abs())
                .max()
                .unwrap_or(0)
                >= GAMMA2 - BETA
        {
            nonce = advance_nonce(nonce)?;
            continue;
        }
        let ct0 = Zeroizing::new(
            t0_hat
                .iter()
                .map(|polynomial| ml_dsa_intt_core(ntt_mul(&challenge_hat, polynomial)))
                .collect::<Vec<_>>(),
        );
        if ct0.iter().map(centred_norm).max().unwrap_or(0) >= GAMMA2 {
            nonce = advance_nonce(nonce)?;
            continue;
        }
        let hints = Zeroizing::new(
            (0..K)
                .map(|row| {
                    std::array::from_fn(|index| {
                        make_hint(
                            rem_q(-ct0[row][index]),
                            rem_q(w[row][index] - cs2[row][index] + ct0[row][index]),
                        )
                    })
                })
                .collect::<Vec<Poly>>(),
        );
        let hint_weight: usize = hints
            .iter()
            .flat_map(|polynomial| polynomial.iter())
            .map(|value| *value as usize)
            .sum();
        if hint_weight > OMEGA {
            nonce = advance_nonce(nonce)?;
            continue;
        }
        let centred_z = Zeroizing::new(
            z.iter()
                .map(|polynomial| polynomial.map(|value| centred(value, Q)))
                .collect::<Vec<_>>(),
        );
        return Ok(encode_signature(&challenge, &centred_z, &hints));
    }
}

fn advance_nonce(nonce: usize) -> Result<usize, &'static str> {
    nonce
        .checked_add(L)
        .filter(|next| *next <= usize::from(u16::MAX) - (L - 1))
        .ok_or("ML-DSA signing rejection nonce exhausted")
}
