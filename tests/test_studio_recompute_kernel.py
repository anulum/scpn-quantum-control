# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio recompute kernel tests
"""Tests for Studio recompute-verifiable compile units."""

from __future__ import annotations

import copy
import struct
from typing import Any

import numpy as np
import pytest

_studio_exactness = pytest.importorskip(
    "scpn_studio_platform.exactness",
    reason="Studio recompute exactness package requires Python >=3.12.",
)
_studio = pytest.importorskip("scpn_quantum_control.studio")
_kernel = pytest.importorskip("scpn_quantum_control.studio.recompute_kernel")
ReproVerdict = _studio_exactness.ReproVerdict
XY_COMPILE_RECOMPUTE_SCHEMA: str = _studio.XY_COMPILE_RECOMPUTE_SCHEMA
XY_COMPILE_WASM_CRATE: str = _studio.XY_COMPILE_WASM_CRATE
XY_COMPILE_WASM_EXPORT: str = _studio.XY_COMPILE_WASM_EXPORT
build_xy_compile_recompute_unit: Any = _studio.build_xy_compile_recompute_unit
canonical_xy_compile_input_bytes: Any = _studio.canonical_xy_compile_input_bytes
decode_xy_compile_input_bytes: Any = _studio.decode_xy_compile_input_bytes
verify_xy_compile_recompute_unit: Any = _studio.verify_xy_compile_recompute_unit
xy_compile_digest_python: Any = _studio.xy_compile_digest_python


def _problem() -> tuple[np.ndarray, np.ndarray]:
    K_nm = np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64)
    omega = np.array([1.0, -0.5], dtype=np.float64)
    return K_nm, omega


def test_builds_bit_exact_recompute_unit() -> None:
    K_nm, omega = _problem()

    unit = build_xy_compile_recompute_unit(
        K_nm,
        omega,
        time=0.1,
        trotter_steps=1,
        trotter_order=1,
    )
    payload = unit.to_dict()

    assert payload["schema"] == XY_COMPILE_RECOMPUTE_SCHEMA
    assert payload["verifiability_mode"] == "recompute"
    assert payload["exactness_class"] == "bit-exact"
    assert payload["wasm_kernel"]["crate"] == XY_COMPILE_WASM_CRATE
    assert payload["wasm_kernel"]["export"] == XY_COMPILE_WASM_EXPORT
    assert payload["claimed_digest"].startswith("sha256:")
    assert payload["input_sha256"].startswith("sha256:")
    assert verify_xy_compile_recompute_unit(unit) is ReproVerdict.MATCH


def test_recompute_unit_detects_digest_drift() -> None:
    K_nm, omega = _problem()
    unit = build_xy_compile_recompute_unit(
        K_nm,
        omega,
        time=0.1,
        trotter_steps=1,
        trotter_order=1,
    )
    payload = unit.to_dict()
    tampered = copy.deepcopy(payload)
    tampered["claimed_digest"] = "sha256:" + ("0" * 64)

    assert verify_xy_compile_recompute_unit(tampered) is ReproVerdict.DRIFT


def test_digest_changes_when_compile_input_changes() -> None:
    K_nm, omega = _problem()
    base = canonical_xy_compile_input_bytes(
        K_nm,
        omega,
        time=0.1,
        trotter_steps=1,
        trotter_order=1,
    )
    changed = canonical_xy_compile_input_bytes(
        K_nm,
        omega,
        time=0.2,
        trotter_steps=1,
        trotter_order=1,
    )

    assert xy_compile_digest_python(base) != xy_compile_digest_python(changed)


def test_recompute_unit_rejects_invalid_exactness_or_inputs() -> None:
    K_nm, omega = _problem()
    unit = build_xy_compile_recompute_unit(
        K_nm,
        omega,
        time=0.1,
        trotter_steps=1,
        trotter_order=1,
    )
    payload = unit.to_dict()
    payload["exactness_class"] = "tolerance"

    with pytest.raises(ValueError, match="exactness_class"):
        verify_xy_compile_recompute_unit(payload)

    with pytest.raises(ValueError, match="trotter_steps"):
        canonical_xy_compile_input_bytes(
            K_nm,
            omega,
            time=0.1,
            trotter_steps=0,
            trotter_order=1,
        )

    with pytest.raises(ValueError, match="trotter_order"):
        canonical_xy_compile_input_bytes(
            K_nm,
            omega,
            time=0.1,
            trotter_steps=1,
            trotter_order=0,
        )


def _raw_digest_bytes(
    K_flat: list[float],
    omega: list[float],
    *,
    version: int = 1,
    time: float = 0.1,
    trotter_steps: int = 1,
    trotter_order: int = 1,
) -> bytes:
    """Pack a raw canonical digest input, bypassing input normalisation.

    Lets a test feed the reference digest structurally invalid or physically
    exotic matrices (asymmetric, all-zero) that the normalised packer forbids.
    """
    n = len(omega)
    header = struct.pack("<IIdII", version, n, time, trotter_steps, trotter_order)
    body = struct.pack(f"<{n * n}d", *K_flat) + struct.pack(f"<{n}d", *omega)
    return header + body


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"\x00" * 10, "too short"),
        (_raw_digest_bytes([0.0], [1.0], version=2), "version mismatch"),
        (struct.pack("<IIdII", 1, 0, 0.1, 1, 1), "n_qubits must be"),
        (_raw_digest_bytes([0.0], [1.0], trotter_steps=0), "trotter steps/order"),
        (_raw_digest_bytes([0.0], [1.0], time=float("inf")), "time must be finite"),
        (struct.pack("<IIdII", 1, 1, 0.1, 1, 1), "length mismatch"),
        (_raw_digest_bytes([0.0], [float("nan")]), "non-finite floats"),
    ],
)
def test_digest_fails_closed_on_malformed_input(payload: bytes, match: str) -> None:
    """The reference digest rejects malformed input rather than hashing garbage."""
    with pytest.raises(ValueError, match=match):
        xy_compile_digest_python(payload)


def test_digest_skips_zero_and_antisymmetric_couplings() -> None:
    """A pair with no coupling, or an antisymmetric one, is skipped, not hashed.

    Both branches are unreachable through the normalised packer (which zeros the
    diagonal and enforces symmetry), so they are exercised via raw bytes.
    """
    all_zero = xy_compile_digest_python(_raw_digest_bytes([0.0, 0.0, 0.0, 0.0], [1.0, -0.5]))
    antisymmetric = xy_compile_digest_python(
        _raw_digest_bytes([0.0, 0.25, -0.25, 0.0], [1.0, -0.5])
    )
    # Both collapse to zero net coupling, so they hash identically to each other
    # and to a genuinely uncoupled pair.
    assert all_zero == antisymmetric
    assert all_zero.startswith("sha256:")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"verifiability_mode": "attestation"}, "must be recompute"),
        ({"input_hex": ""}, "non-empty hex"),
        ({"claimed_digest": 123}, "must be a string"),
    ],
)
def test_verify_fails_closed_on_stripped_unit_fields(mutation: dict[str, Any], match: str) -> None:
    """Stripping the mode, input, or claimed digest raises rather than passing."""
    K_nm, omega = _problem()
    payload = build_xy_compile_recompute_unit(
        K_nm, omega, time=0.1, trotter_steps=1, trotter_order=1
    ).to_dict()
    payload.update(mutation)
    with pytest.raises(ValueError, match=match):
        verify_xy_compile_recompute_unit(payload)


def test_pack_helpers_fail_closed_on_out_of_domain_values() -> None:
    """The private packers reject out-of-range integers and non-finite floats."""
    with pytest.raises(ValueError, match="u32 value out of range"):
        _kernel._pack_u32(2**32)
    with pytest.raises(ValueError, match="u32 value out of range"):
        _kernel._pack_u32(-1)
    with pytest.raises(ValueError, match="must be finite"):
        _kernel._pack_f64(float("inf"), name="probe")


def test_decode_round_trips_the_canonical_input() -> None:
    """The decoder is the inverse of the canonical packer for K, omega, params."""
    K_nm, omega = _problem()
    packed = canonical_xy_compile_input_bytes(
        K_nm,
        omega,
        time=0.1,
        trotter_steps=2,
        trotter_order=3,
    )
    decoded = decode_xy_compile_input_bytes(packed)

    np.testing.assert_array_equal(decoded.K_nm, K_nm)
    np.testing.assert_array_equal(decoded.omega, omega)
    assert decoded.time == 0.1
    assert decoded.trotter_steps == 2
    assert decoded.trotter_order == 3


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"\x00" * 10, "too short"),
        # version 2 header over a 1x1 problem
        (
            struct.pack("<IIdII", 2, 1, 0.1, 1, 1) + struct.pack("<dd", 0.0, 1.0),
            "version mismatch",
        ),
        # zero qubits
        (struct.pack("<IIdII", 1, 0, 0.1, 1, 1), "n_qubits must be"),
        # bad trotter steps
        (
            struct.pack("<IIdII", 1, 1, 0.1, 0, 1) + struct.pack("<dd", 0.0, 1.0),
            "trotter steps/order",
        ),
        # header promises one qubit but the body is truncated
        (struct.pack("<IIdII", 1, 1, 0.1, 1, 1), "length mismatch"),
    ],
)
def test_decode_fails_closed_on_malformed_input(payload: bytes, match: str) -> None:
    """A malformed canonical payload is rejected rather than decoded to garbage."""
    with pytest.raises(ValueError, match=match):
        decode_xy_compile_input_bytes(payload)
