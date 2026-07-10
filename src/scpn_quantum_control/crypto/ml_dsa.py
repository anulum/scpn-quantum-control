# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — FIPS 204 ML-DSA-65 module-lattice signatures
"""ML-DSA-65 module-lattice digital signatures, implemented from FIPS 204.

Reference: NIST FIPS 204, *Module-Lattice-Based Digital Signature Standard*
(August 2024). This is a from-specification implementation of the ML-DSA-65
parameter set, validated against the official NIST ACVP known-answer vectors
(keyGen, sigGen deterministic, sigVer). It is FIPS 204-*conformant* (reproduces
the standard's test vectors); it is not a FIPS-140-validated cryptographic
module and carries no side-channel-resistance guarantee.

The polynomial ring is ``R_q = Z_q[X]/(X^256 + 1)`` with ``q = 8380417``. The
number-theoretic transform (``ntt`` / ``intt``) dispatches to a bit-true Rust
kernel when the acceleration engine is installed; any fallback to the
pure-Python reference is logged once per process, and setting
``SCPN_REQUIRE_NATIVE_CRYPTO=1`` turns a fallback into a ``RuntimeError``.

Secret material (seeds, secret keys) lives in ordinary Python ``bytes``;
CPython provides no reliable memory zeroisation, so secrets must be assumed
to persist in process memory until interpreter exit.
"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _hashlib import HASHXOF
    from collections.abc import Callable

_LOGGER = logging.getLogger(__name__)

REQUIRE_NATIVE_ENV = "SCPN_REQUIRE_NATIVE_CRYPTO"

_RESEARCH_BOUNDARY_WARNING = (
    "scpn_quantum_control.crypto.ml_dsa is a research implementation of FIPS 204:"
    " it reproduces the standard's ACVP test vectors but carries no constant-time,"
    " side-channel-resistance, or FIPS-140 validation guarantee; do not protect"
    " production secrets with it."
)

_ntt_fallback_logged = False
_research_warning_emitted = False


def _warn_research_boundary(suppress: bool) -> None:
    """Emit the research-boundary ``UserWarning`` once per process."""
    global _research_warning_emitted
    if suppress or _research_warning_emitted:
        return
    _research_warning_emitted = True
    warnings.warn(_RESEARCH_BOUNDARY_WARNING, UserWarning, stacklevel=3)


# --- ML-DSA-65 parameters (FIPS 204, Table 1) ------------------------------- #
Q = 8380417
N = 256
D = 13
TAU = 49
LAMBDA = 192
GAMMA1 = 1 << 19
GAMMA2 = (Q - 1) // 32
K = 6
L = 5
ETA = 4
BETA = TAU * ETA  # 196
OMEGA = 55
C_TILDE_BYTES = LAMBDA // 4  # 48
ZETA = 1753

PUBLIC_KEY_BYTES = 1952
SECRET_KEY_BYTES = 4032
SIGNATURE_BYTES = 3309
SEED_BYTES = 32


def _bit_reverse_8(value: int) -> int:
    result = 0
    for _ in range(8):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


# zetas[k] = ZETA^bitrev8(k) mod Q, used by the in-place NTT.
_ZETAS = [pow(ZETA, _bit_reverse_8(k), Q) for k in range(N)]
_F_256_INV = pow(N, -1, Q)  # 256^-1 mod q for the inverse NTT scaling


def _shake256(*chunks: bytes, length: int) -> bytes:
    h = hashlib.shake_256()
    for chunk in chunks:
        h.update(chunk)
    return h.digest(length)


def _shake128_xof(*chunks: bytes) -> HASHXOF:
    h = hashlib.shake_128()
    for chunk in chunks:
        h.update(chunk)
    return h


def _shake256_xof(*chunks: bytes) -> HASHXOF:
    h = hashlib.shake_256()
    for chunk in chunks:
        h.update(chunk)
    return h


# --- Number-theoretic transform --------------------------------------------- #
def _ntt_python(poly: list[int]) -> list[int]:
    w = list(poly)
    k = 0
    length = 128
    while length >= 1:
        start = 0
        while start < N:
            k += 1
            zeta = _ZETAS[k]
            for j in range(start, start + length):
                t = (zeta * w[j + length]) % Q
                w[j + length] = (w[j] - t) % Q
                w[j] = (w[j] + t) % Q
            start += 2 * length
        length //= 2
    return w


def _intt_python(poly: list[int]) -> list[int]:
    w = list(poly)
    k = N
    length = 1
    while length < N:
        start = 0
        while start < N:
            k -= 1
            zeta = (-_ZETAS[k]) % Q
            for j in range(start, start + length):
                t = w[j]
                w[j] = (t + w[j + length]) % Q
                w[j + length] = (t - w[j + length]) % Q
                w[j + length] = (zeta * w[j + length]) % Q
            start += 2 * length
        length *= 2
    return [(_F_256_INV * x) % Q for x in w]


def _native_kernel(name: str) -> Callable[[list[int]], list[int]] | None:
    """Return the Rust NTT kernel ``name``, or ``None`` when unavailable."""
    try:
        import scpn_quantum_engine as _engine
    except ImportError:
        return None
    kernel = getattr(_engine, name, None)
    return kernel if callable(kernel) else None


def _log_ntt_fallback(reason: str) -> None:
    """Log the active-path decision once per process."""
    global _ntt_fallback_logged
    if _ntt_fallback_logged:
        return
    _ntt_fallback_logged = True
    _LOGGER.warning(
        "ML-DSA NTT dispatch: using the pure-Python reference path (%s);"
        " set %s=1 to make any fallback an error.",
        reason,
        REQUIRE_NATIVE_ENV,
    )


def _dispatch_ntt(
    name: str, python_impl: Callable[[list[int]], list[int]], poly: list[int]
) -> list[int]:
    """Dispatch to the Rust kernel ``name`` with an observable fallback.

    The pure-Python reference is bit-true, so the fallback is safe — but it
    must never be silent: the first fallback in a process logs a warning
    naming the reason, and ``SCPN_REQUIRE_NATIVE_CRYPTO=1`` raises
    ``RuntimeError`` instead of falling back.
    """
    strict = os.environ.get(REQUIRE_NATIVE_ENV) == "1"
    kernel = _native_kernel(name)
    if kernel is None:
        if strict:
            raise RuntimeError(
                f"{REQUIRE_NATIVE_ENV}=1 but the Rust kernel '{name}' is unavailable"
            )
        _log_ntt_fallback(f"Rust kernel '{name}' unavailable")
        return python_impl(poly)
    try:
        return list(kernel(poly))
    except (ValueError, TypeError) as exc:
        if strict:
            raise RuntimeError(
                f"{REQUIRE_NATIVE_ENV}=1 and the Rust kernel '{name}' failed: {exc!r}"
            ) from exc
        _log_ntt_fallback(f"Rust kernel '{name}' failed: {exc!r}")
        return python_impl(poly)


def ntt(poly: list[int]) -> list[int]:
    """Forward NTT, dispatching to the Rust kernel when available.

    A fallback to the pure-Python reference logs a once-per-process warning;
    ``SCPN_REQUIRE_NATIVE_CRYPTO=1`` raises ``RuntimeError`` instead.
    """
    return _dispatch_ntt("ml_dsa_ntt", _ntt_python, poly)


def intt(poly: list[int]) -> list[int]:
    """Inverse NTT, dispatching to the Rust kernel when available.

    A fallback to the pure-Python reference logs a once-per-process warning;
    ``SCPN_REQUIRE_NATIVE_CRYPTO=1`` raises ``RuntimeError`` instead.
    """
    return _dispatch_ntt("ml_dsa_intt", _intt_python, poly)


def _ntt_mul(a_hat: list[int], b_hat: list[int]) -> list[int]:
    return [(a_hat[i] * b_hat[i]) % Q for i in range(N)]


def _poly_add(a: list[int], b: list[int]) -> list[int]:
    return [(a[i] + b[i]) % Q for i in range(N)]


def _poly_sub(a: list[int], b: list[int]) -> list[int]:
    return [(a[i] - b[i]) % Q for i in range(N)]


# --- Rounding (FIPS 204 §7.4) ----------------------------------------------- #
def _mod_pm(r: int, alpha: int) -> int:
    """Centred representative of r mod alpha in (-alpha/2, alpha/2]."""
    r = r % alpha
    if r > alpha // 2:
        r -= alpha
    return r


def _power2round(r: int) -> tuple[int, int]:
    r = r % Q
    r1 = (r + (1 << (D - 1)) - 1) >> D
    r0 = r - (r1 << D)
    return r1, r0


def _decompose(r: int) -> tuple[int, int]:
    r = r % Q
    r0 = _mod_pm(r, 2 * GAMMA2)
    if r - r0 == Q - 1:
        r1 = 0
        r0 -= 1
    else:
        r1 = (r - r0) // (2 * GAMMA2)
    return r1, r0


def _high_bits(r: int) -> int:
    return _decompose(r)[0]


def _low_bits(r: int) -> int:
    return _decompose(r)[1]


def _make_hint(z: int, r: int) -> int:
    return int(_high_bits(r) != _high_bits(r + z))


def _use_hint(h: int, r: int) -> int:
    m = (Q - 1) // (2 * GAMMA2)
    r1, r0 = _decompose(r)
    if h == 0:
        return r1
    if r0 > 0:
        return (r1 + 1) % m
    return (r1 - 1) % m


# --- Bit / byte conversions (FIPS 204 §7.1) --------------------------------- #
def _integer_to_bits(x: int, alpha: int) -> list[int]:
    return [(x >> i) & 1 for i in range(alpha)]


def _bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray(len(bits) // 8)
    for i, bit in enumerate(bits):
        out[i // 8] |= bit << (i % 8)
    return bytes(out)


def _bytes_to_bits(data: bytes) -> list[int]:
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits


# --- Bit packing (FIPS 204 §7.2) -------------------------------------------- #
def _simple_bit_pack(poly: list[int], bitlen: int) -> bytes:
    bits: list[int] = []
    for coeff in poly:
        bits.extend(_integer_to_bits(coeff, bitlen))
    return _bits_to_bytes(bits)


def _bit_pack(poly: list[int], a: int, b: int) -> bytes:
    bitlen = (a + b).bit_length()
    bits: list[int] = []
    for coeff in poly:
        bits.extend(_integer_to_bits(b - coeff, bitlen))
    return _bits_to_bytes(bits)


def _simple_bit_unpack(data: bytes, bitlen: int) -> list[int]:
    bits = _bytes_to_bits(data)
    return [sum(bits[i * bitlen + j] << j for j in range(bitlen)) for i in range(N)]


def _bit_unpack(data: bytes, a: int, b: int) -> list[int]:
    bitlen = (a + b).bit_length()
    bits = _bytes_to_bits(data)
    out = []
    for i in range(N):
        val = sum(bits[i * bitlen + j] << j for j in range(bitlen))
        out.append(b - val)
    return out


def _hint_bit_pack(h: list[list[int]]) -> bytes:
    out = bytearray(OMEGA + K)
    index = 0
    for i in range(K):
        for j in range(N):
            if h[i][j] != 0:
                out[index] = j
                index += 1
        out[OMEGA + i] = index
    return bytes(out)


def _hint_bit_unpack(data: bytes) -> list[list[int]] | None:
    h = [[0] * N for _ in range(K)]
    index = 0
    for i in range(K):
        end = data[OMEGA + i]
        if end < index or end > OMEGA:
            return None
        first = index
        while index < end:
            if index > first and data[index - 1] >= data[index]:
                return None
            h[i][data[index]] = 1
            index += 1
    for j in range(index, OMEGA):
        if data[j] != 0:
            return None
    return h


# --- Sampling (FIPS 204 §7.3) ----------------------------------------------- #
def _sample_in_ball(rho: bytes) -> list[int]:
    c = [0] * N
    xof = _shake256_xof(rho)
    stream = bytearray(xof.digest(8 + 256))
    h = int.from_bytes(stream[:8], "little")  # 64 sign bits, LSB first
    pos = 8
    for i in range(N - TAU, N):
        while True:
            if pos >= len(stream):
                stream.extend(xof.digest(pos + 256)[pos:])
            j = stream[pos]
            pos += 1
            if j <= i:
                break
        c[i] = c[j]
        c[j] = 1 - 2 * (h & 1)
        h >>= 1
    return c


def _coeff_from_three_bytes(b0: int, b1: int, b2: int) -> int | None:
    z = b0 | (b1 << 8) | ((b2 & 0x7F) << 16)
    return z if z < Q else None


def _coeff_from_half_byte(b: int) -> int | None:
    if ETA == 2 and b < 15:
        return 2 - (b % 5)
    if ETA == 4 and b < 9:
        return 4 - b
    return None


def _rej_ntt_poly(rho: bytes, s: int, r: int) -> list[int]:
    xof = _shake128_xof(rho + bytes([s, r]))
    a: list[int] = []
    buf = bytearray()
    pos = 0
    while len(a) < N:
        if pos + 3 > len(buf):
            buf.extend(xof.digest(len(buf) + 504)[len(buf) :])
        coeff = _coeff_from_three_bytes(buf[pos], buf[pos + 1], buf[pos + 2])
        pos += 3
        if coeff is not None:
            a.append(coeff)
    return a


def _rej_bounded_poly(rho: bytes, r: int) -> list[int]:
    xof = _shake256_xof(rho + r.to_bytes(2, "little"))
    a: list[int] = []
    buf = bytearray()
    pos = 0
    while len(a) < N:
        if pos >= len(buf):
            buf.extend(xof.digest(len(buf) + 272)[len(buf) :])
        byte = buf[pos]
        pos += 1
        z0 = _coeff_from_half_byte(byte & 0x0F)
        if z0 is not None:
            a.append(z0)  # signed in [-eta, eta] for BitPack
            if len(a) == N:
                break
        z1 = _coeff_from_half_byte(byte >> 4)
        if z1 is not None:
            a.append(z1)
    return a


def _expand_a(rho: bytes) -> list[list[list[int]]]:
    return [[_rej_ntt_poly(rho, s, r) for s in range(L)] for r in range(K)]


def _expand_s(rho: bytes) -> tuple[list[list[int]], list[list[int]]]:
    s1 = [_rej_bounded_poly(rho, r) for r in range(L)]
    s2 = [_rej_bounded_poly(rho, r + L) for r in range(K)]
    return s1, s2


def _expand_mask(rho: bytes, mu: int) -> list[list[int]]:
    c = 1 + (GAMMA1 - 1).bit_length()
    y = []
    for r in range(L):
        v = _shake256(rho, (mu + r).to_bytes(2, "little"), length=32 * c)
        y.append([coeff % Q for coeff in _bit_unpack(v, GAMMA1 - 1, GAMMA1)])
    return y


@dataclass(frozen=True)
class MLDSAKeyPair:
    """An ML-DSA-65 key pair (public and secret key bytes)."""

    public_key: bytes
    secret_key: bytes


# --- Key / signature encoding (FIPS 204 §7.2) ------------------------------- #
_T1_BITLEN = (Q - 1).bit_length() - D  # 10
_ETA_BYTES = N * (2 * ETA).bit_length() // 8  # 128
_T0_BYTES = N * D // 8  # 416
_Z_BYTES = N * (1 + (GAMMA1 - 1).bit_length()) // 8  # 640
_W1_BITLEN = ((Q - 1) // (2 * GAMMA2) - 1).bit_length()  # 4


def _pk_encode(rho: bytes, t1: list[list[int]]) -> bytes:
    out = bytearray(rho)
    for i in range(K):
        out += _simple_bit_pack(t1[i], _T1_BITLEN)
    return bytes(out)


def _pk_decode(pk: bytes) -> tuple[bytes, list[list[int]]]:
    rho = pk[:32]
    plen = N * _T1_BITLEN // 8
    off = 32
    t1 = []
    for _ in range(K):
        t1.append(_simple_bit_unpack(pk[off : off + plen], _T1_BITLEN))
        off += plen
    return rho, t1


def _sk_encode(
    rho: bytes,
    k_seed: bytes,
    tr: bytes,
    s1: list[list[int]],
    s2: list[list[int]],
    t0: list[list[int]],
) -> bytes:
    out = bytearray(rho + k_seed + tr)
    for p in s1:
        out += _bit_pack(p, ETA, ETA)
    for p in s2:
        out += _bit_pack(p, ETA, ETA)
    for p in t0:
        out += _bit_pack(p, (1 << (D - 1)) - 1, 1 << (D - 1))
    return bytes(out)


def _sk_decode(
    sk: bytes,
) -> tuple[bytes, bytes, bytes, list[list[int]], list[list[int]], list[list[int]]]:
    rho, k_seed, tr = sk[:32], sk[32:64], sk[64:128]
    off = 128
    s1, s2, t0 = [], [], []
    for _ in range(L):
        s1.append([x % Q for x in _bit_unpack(sk[off : off + _ETA_BYTES], ETA, ETA)])
        off += _ETA_BYTES
    for _ in range(K):
        s2.append([x % Q for x in _bit_unpack(sk[off : off + _ETA_BYTES], ETA, ETA)])
        off += _ETA_BYTES
    for _ in range(K):
        b = 1 << (D - 1)
        t0.append([x % Q for x in _bit_unpack(sk[off : off + _T0_BYTES], b - 1, b)])
        off += _T0_BYTES
    return rho, k_seed, tr, s1, s2, t0


def _sig_encode(c_tilde: bytes, z: list[list[int]], h: list[list[int]]) -> bytes:
    out = bytearray(c_tilde)
    for p in z:
        out += _bit_pack(p, GAMMA1 - 1, GAMMA1)
    out += _hint_bit_pack(h)
    return bytes(out)


def _sig_decode(sig: bytes) -> tuple[bytes, list[list[int]], list[list[int]] | None]:
    c_tilde = sig[:C_TILDE_BYTES]
    off = C_TILDE_BYTES
    z = []
    for _ in range(L):
        z.append([x % Q for x in _bit_unpack(sig[off : off + _Z_BYTES], GAMMA1 - 1, GAMMA1)])
        off += _Z_BYTES
    h = _hint_bit_unpack(sig[off : off + OMEGA + K])
    return c_tilde, z, h


def _w1_encode(w1: list[list[int]]) -> bytes:
    out = bytearray()
    for p in w1:
        out += _simple_bit_pack(p, _W1_BITLEN)
    return bytes(out)


def _centered_norm(poly: list[int]) -> int:
    return max(abs(_mod_pm(c, Q)) for c in poly)


# --- KeyGen / Sign / Verify (FIPS 204 §6) ----------------------------------- #
def _keygen_internal(xi: bytes) -> tuple[bytes, bytes]:
    seed = _shake256(xi, bytes([K, L]), length=128)
    rho, rho_prime, k_seed = seed[:32], seed[32:96], seed[96:128]
    a_hat = _expand_a(rho)
    s1, s2 = _expand_s(rho_prime)
    s1_hat = [ntt(p) for p in s1]
    t1, t0 = [], []
    for i in range(K):
        acc = [0] * N
        for j in range(L):
            acc = _poly_add(acc, _ntt_mul(a_hat[i][j], s1_hat[j]))
        t_i = _poly_add(intt(acc), s2[i])
        r1 = [0] * N
        r0 = [0] * N
        for n in range(N):
            r1[n], r0[n] = _power2round(t_i[n])
        t1.append(r1)
        t0.append(r0)
    pk = _pk_encode(rho, t1)
    tr = _shake256(pk, length=64)
    sk = _sk_encode(rho, k_seed, tr, s1, s2, t0)
    return pk, sk


def _sign_internal(sk: bytes, m_prime: bytes, rnd: bytes) -> bytes:
    rho, k_seed, tr, s1, s2, t0 = _sk_decode(sk)
    a_hat = _expand_a(rho)
    s1_hat = [ntt(p) for p in s1]
    s2_hat = [ntt(p) for p in s2]
    t0_hat = [ntt(p) for p in t0]
    mu = _shake256(tr, m_prime, length=64)
    rho_pp = _shake256(k_seed, rnd, mu, length=64)

    kappa = 0
    while True:
        y = _expand_mask(rho_pp, kappa)
        y_hat = [ntt(p) for p in y]
        w = []
        for i in range(K):
            acc = [0] * N
            for j in range(L):
                acc = _poly_add(acc, _ntt_mul(a_hat[i][j], y_hat[j]))
            w.append(intt(acc))
        w1 = [[_high_bits(c) for c in w[i]] for i in range(K)]
        c_tilde = _shake256(mu, _w1_encode(w1), length=C_TILDE_BYTES)
        c_hat = ntt(_sample_in_ball(c_tilde))
        cs1 = [intt(_ntt_mul(c_hat, s1_hat[j])) for j in range(L)]
        cs2 = [intt(_ntt_mul(c_hat, s2_hat[i])) for i in range(K)]
        z = [_poly_add(y[j], cs1[j]) for j in range(L)]
        r0 = [[_low_bits((w[i][n] - cs2[i][n]) % Q) for n in range(N)] for i in range(K)]
        if max(_centered_norm(p) for p in z) >= GAMMA1 - BETA:
            kappa += L
            continue
        if max(max(abs(c) for c in p) for p in r0) >= GAMMA2 - BETA:
            kappa += L
            continue
        ct0 = [intt(_ntt_mul(c_hat, t0_hat[i])) for i in range(K)]
        h = []
        for i in range(K):
            row = []
            for n in range(N):
                z_arg = (-ct0[i][n]) % Q
                r_arg = (w[i][n] - cs2[i][n] + ct0[i][n]) % Q
                row.append(_make_hint(z_arg, r_arg))
            h.append(row)
        if max(_centered_norm(p) for p in ct0) >= GAMMA2:
            kappa += L
            continue
        if sum(sum(row) for row in h) > OMEGA:
            kappa += L
            continue
        z_centered = [[_mod_pm(c, Q) for c in p] for p in z]
        return _sig_encode(c_tilde, z_centered, h)


def _verify_internal(pk: bytes, m_prime: bytes, sig: bytes) -> bool:
    rho, t1 = _pk_decode(pk)
    c_tilde, z, h = _sig_decode(sig)
    if h is None:
        return False
    if max(_centered_norm(p) for p in z) >= GAMMA1 - BETA:
        return False
    a_hat = _expand_a(rho)
    tr = _shake256(pk, length=64)
    mu = _shake256(tr, m_prime, length=64)
    c_hat = ntt(_sample_in_ball(c_tilde))
    z_hat = [ntt(p) for p in z]
    t1_hat = [ntt([(coeff << D) % Q for coeff in t1[i]]) for i in range(K)]
    w_approx = []
    for i in range(K):
        acc = [0] * N
        for j in range(L):
            acc = _poly_add(acc, _ntt_mul(a_hat[i][j], z_hat[j]))
        acc = _poly_sub(acc, _ntt_mul(c_hat, t1_hat[i]))
        w_approx.append(intt(acc))
    w1 = [[_use_hint(h[i][n], w_approx[i][n]) for n in range(N)] for i in range(K)]
    c_tilde_prime = _shake256(mu, _w1_encode(w1), length=C_TILDE_BYTES)
    return c_tilde == c_tilde_prime


def _encode_message(message: bytes, context: bytes) -> bytes:
    if len(context) > 255:
        raise ValueError("context must be at most 255 bytes")
    return bytes([0, len(context)]) + context + message


def key_gen(seed: bytes, *, suppress_research_warning: bool = False) -> MLDSAKeyPair:
    """Deterministic ML-DSA-65 key generation from a 32-byte seed.

    The first key-material operation in a process emits a ``UserWarning``
    stating the research boundary (no constant-time or FIPS-140 guarantee);
    pass ``suppress_research_warning=True`` to acknowledge it explicitly.
    """
    _warn_research_boundary(suppress_research_warning)
    if len(seed) != SEED_BYTES:
        raise ValueError(f"seed must be {SEED_BYTES} bytes")
    pk, sk = _keygen_internal(seed)
    return MLDSAKeyPair(public_key=pk, secret_key=sk)


def sign(
    secret_key: bytes,
    message: bytes,
    *,
    context: bytes = b"",
    randomness: bytes | None = None,
    suppress_research_warning: bool = False,
) -> bytes:
    """ML-DSA-65 signature (pure, external interface). Deterministic by default.

    The first key-material operation in a process emits a ``UserWarning``
    stating the research boundary (no constant-time or FIPS-140 guarantee);
    pass ``suppress_research_warning=True`` to acknowledge it explicitly.
    """
    _warn_research_boundary(suppress_research_warning)
    if len(secret_key) != SECRET_KEY_BYTES:
        raise ValueError(f"secret_key must be {SECRET_KEY_BYTES} bytes")
    rnd = bytes(32) if randomness is None else randomness
    if len(rnd) != 32:
        raise ValueError("randomness must be 32 bytes")
    return _sign_internal(secret_key, _encode_message(message, context), rnd)


def verify(public_key: bytes, message: bytes, signature: bytes, *, context: bytes = b"") -> bool:
    """Verify an ML-DSA-65 signature (pure, external interface)."""
    if len(public_key) != PUBLIC_KEY_BYTES or len(signature) != SIGNATURE_BYTES:
        return False
    return _verify_internal(public_key, _encode_message(message, context), signature)


__all__ = [
    "MLDSAKeyPair",
    "PUBLIC_KEY_BYTES",
    "SECRET_KEY_BYTES",
    "SIGNATURE_BYTES",
    "key_gen",
    "ntt",
    "intt",
    "sign",
    "verify",
]
