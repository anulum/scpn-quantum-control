# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio recompute kernels
"""Recompute-verifiable Studio units for deterministic compile claims."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scpn_studio_platform.exactness import (
    ExactnessClass,
    ReproVerdict,
    compare_bit_exact,
)

from ..kuramoto_core import validate_kuramoto_inputs

XY_COMPILE_RECOMPUTE_SCHEMA = "studio.xy-compile-recompute.v1"
XY_COMPILE_INPUT_VERSION = 1
XY_COMPILE_WASM_CRATE = "scpn_quantum_engine/studio_wasm_kernel"
XY_COMPILE_WASM_EXPORT = "scpn_xy_compile_digest"
_SCHEMA_TAG = b"scpn.quantum.xy_compile.v1\0"


def _pack_u32(value: int) -> bytes:
    """Return ``value`` encoded as little-endian ``u32``."""
    if value < 0 or value > 2**32 - 1:
        raise ValueError("u32 value out of range")
    return struct.pack("<I", value)


def _pack_f64(value: float, *, name: str) -> bytes:
    """Return ``value`` encoded as little-endian finite ``f64``."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return struct.pack("<d", parsed)


def canonical_xy_compile_input_bytes(
    K_nm: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    time: float,
    trotter_steps: int,
    trotter_order: int,
) -> bytes:
    """Return the canonical binary input consumed by the WASM verifier kernel."""
    if trotter_steps < 1:
        raise ValueError("trotter_steps must be >= 1")
    if trotter_order < 1:
        raise ValueError("trotter_order must be >= 1")
    K_arr, omega_arr = validate_kuramoto_inputs(K_nm, omega)
    n_qubits = int(omega_arr.shape[0])
    payload = bytearray()
    payload.extend(_pack_u32(XY_COMPILE_INPUT_VERSION))
    payload.extend(_pack_u32(n_qubits))
    payload.extend(_pack_f64(float(time), name="time"))
    payload.extend(_pack_u32(int(trotter_steps)))
    payload.extend(_pack_u32(int(trotter_order)))
    for value in K_arr.reshape(-1):
        payload.extend(_pack_f64(float(value), name="K_nm"))
    for value in omega_arr:
        payload.extend(_pack_f64(float(value), name="omega"))
    return bytes(payload)


@dataclass(frozen=True, eq=False)
class DecodedXYCompileInput:
    """Structured view of a decoded canonical XY-compile input payload.

    This is the inverse image of :func:`canonical_xy_compile_input_bytes`: it
    exposes the coupling matrix, natural frequencies, and compile parameters a
    committed recompute unit was built from, so a caller can bind the frozen
    input back to its physical source by tolerance rather than by a bit-exact
    rebuild (``np.exp`` in the source matrix is not reproducible to the last
    ULP across platforms).
    """

    K_nm: NDArray[np.float64]
    omega: NDArray[np.float64]
    time: float
    trotter_steps: int
    trotter_order: int


def decode_xy_compile_input_bytes(input_bytes: bytes) -> DecodedXYCompileInput:
    """Return the structured fields packed into a canonical XY-compile input.

    Applies the same version and length validation as the reference digest so a
    malformed payload fails closed rather than decoding to silent garbage.
    """
    if len(input_bytes) < 24:
        raise ValueError("compile input is too short")
    version, n_qubits = struct.unpack_from("<II", input_bytes, 0)
    if version != XY_COMPILE_INPUT_VERSION:
        raise ValueError("compile input version mismatch")
    time = struct.unpack_from("<d", input_bytes, 8)[0]
    trotter_steps, trotter_order = struct.unpack_from("<II", input_bytes, 16)
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if trotter_steps < 1 or trotter_order < 1:
        raise ValueError("trotter steps/order must be >= 1")
    n = int(n_qubits)
    expected_len = 24 + ((n * n) + n) * 8
    if len(input_bytes) != expected_len:
        raise ValueError("compile input length mismatch")
    offset = 24
    K_values = struct.unpack_from(f"<{n * n}d", input_bytes, offset)
    offset += n * n * 8
    omega_values = struct.unpack_from(f"<{n}d", input_bytes, offset)
    return DecodedXYCompileInput(
        K_nm=np.asarray(K_values, dtype=np.float64).reshape(n, n),
        omega=np.asarray(omega_values, dtype=np.float64),
        time=float(time),
        trotter_steps=int(trotter_steps),
        trotter_order=int(trotter_order),
    )


def xy_compile_digest_python(input_bytes: bytes) -> str:
    """Return the Python reference digest for the WASM compile verifier input."""
    if len(input_bytes) < 24:
        raise ValueError("compile input is too short")
    version, n_qubits = struct.unpack_from("<II", input_bytes, 0)
    if version != XY_COMPILE_INPUT_VERSION:
        raise ValueError("compile input version mismatch")
    time = struct.unpack_from("<d", input_bytes, 8)[0]
    trotter_steps, trotter_order = struct.unpack_from("<II", input_bytes, 16)
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    if trotter_steps < 1 or trotter_order < 1:
        raise ValueError("trotter steps/order must be >= 1")
    if not math.isfinite(float(time)):
        raise ValueError("time must be finite")
    n = int(n_qubits)
    expected_len = 24 + ((n * n) + n) * 8
    if len(input_bytes) != expected_len:
        raise ValueError("compile input length mismatch")
    offset = 24
    K_values = list(struct.unpack_from(f"<{n * n}d", input_bytes, offset))
    offset += n * n * 8
    omega_values = list(struct.unpack_from(f"<{n}d", input_bytes, offset))
    if any(not math.isfinite(value) for value in [*K_values, *omega_values]):
        raise ValueError("compile input contains non-finite floats")

    dt = float(time) / float(trotter_steps)
    digest = hashlib.sha256()
    digest.update(_SCHEMA_TAG)
    digest.update(_pack_u32(n))
    digest.update(_pack_f64(float(time), name="time"))
    digest.update(_pack_u32(int(trotter_steps)))
    digest.update(_pack_u32(int(trotter_order)))
    digest.update(_pack_f64(dt, name="dt"))
    for qubit, omega_value in enumerate(omega_values):
        digest.update(_pack_u32(qubit))
        digest.update(_pack_f64(omega_value, name="omega"))
        digest.update(_pack_f64(omega_value * dt, name="frequency_angle"))
    for source in range(n):
        for target in range(source + 1, n):
            forward = K_values[source * n + target]
            reverse = K_values[target * n + source]
            if forward == 0.0 and reverse == 0.0:
                continue
            coupling = 0.5 * (forward + reverse)
            if coupling == 0.0:
                continue
            angle = coupling * dt
            digest.update(_pack_u32(source))
            digest.update(_pack_u32(target))
            digest.update(_pack_f64(forward, name="forward_coupling"))
            digest.update(_pack_f64(reverse, name="reverse_coupling"))
            digest.update(_pack_f64(coupling, name="symmetric_coupling"))
            digest.update(_pack_f64(angle, name="xx_angle"))
            digest.update(_pack_f64(angle, name="yy_angle"))
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class XYCompileRecomputeUnit:
    """Signed-unit payload for bit-exact XY compile recomputation."""

    input_bytes: bytes
    claimed_digest: str
    exactness_class: Literal["bit-exact"] = "bit-exact"
    verifiability_mode: Literal["recompute"] = "recompute"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready recompute unit."""
        return {
            "schema": XY_COMPILE_RECOMPUTE_SCHEMA,
            "verifiability_mode": self.verifiability_mode,
            "exactness_class": self.exactness_class,
            "wasm_kernel": {
                "crate": XY_COMPILE_WASM_CRATE,
                "target": "wasm32-unknown-unknown",
                "export": XY_COMPILE_WASM_EXPORT,
                "input_encoding": "little-endian-xy-compile-v1",
            },
            "input_sha256": f"sha256:{hashlib.sha256(self.input_bytes).hexdigest()}",
            "input_hex": self.input_bytes.hex(),
            "claimed_digest": self.claimed_digest,
        }


def build_xy_compile_recompute_unit(
    K_nm: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    time: float,
    trotter_steps: int,
    trotter_order: int,
) -> XYCompileRecomputeUnit:
    """Build a bit-exact Studio recompute unit for an XY compile claim."""
    input_bytes = canonical_xy_compile_input_bytes(
        K_nm,
        omega,
        time=time,
        trotter_steps=trotter_steps,
        trotter_order=trotter_order,
    )
    return XYCompileRecomputeUnit(
        input_bytes=input_bytes,
        claimed_digest=xy_compile_digest_python(input_bytes),
        exactness_class=ExactnessClass.BIT_EXACT.value,
        verifiability_mode="recompute",
    )


def verify_xy_compile_recompute_unit(
    unit: XYCompileRecomputeUnit | dict[str, Any],
) -> ReproVerdict:
    """Verify a recompute unit by comparing its claimed and recomputed digests."""
    payload = unit.to_dict() if isinstance(unit, XYCompileRecomputeUnit) else dict(unit)
    if payload.get("schema") != XY_COMPILE_RECOMPUTE_SCHEMA:
        raise ValueError("recompute unit schema mismatch")
    if payload.get("verifiability_mode") != "recompute":
        raise ValueError("verifiability_mode must be recompute")
    if payload.get("exactness_class") != ExactnessClass.BIT_EXACT.value:
        raise ValueError("exactness_class must be bit-exact")
    input_hex = payload.get("input_hex")
    if not isinstance(input_hex, str) or not input_hex:
        raise ValueError("input_hex must be a non-empty hex string")
    input_bytes = bytes.fromhex(input_hex)
    recomputed = xy_compile_digest_python(input_bytes)
    claimed = payload.get("claimed_digest")
    if not isinstance(claimed, str):
        raise ValueError("claimed_digest must be a string")
    return compare_bit_exact(recomputed, claimed)


__all__ = [
    "XY_COMPILE_INPUT_VERSION",
    "XY_COMPILE_RECOMPUTE_SCHEMA",
    "XY_COMPILE_WASM_CRATE",
    "XY_COMPILE_WASM_EXPORT",
    "DecodedXYCompileInput",
    "XYCompileRecomputeUnit",
    "build_xy_compile_recompute_unit",
    "canonical_xy_compile_input_bytes",
    "decode_xy_compile_input_bytes",
    "verify_xy_compile_recompute_unit",
    "xy_compile_digest_python",
]
