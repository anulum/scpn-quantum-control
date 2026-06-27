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

import numpy as np
import pytest

from scpn_quantum_control.studio import (
    XY_COMPILE_RECOMPUTE_SCHEMA,
    XY_COMPILE_WASM_CRATE,
    XY_COMPILE_WASM_EXPORT,
    build_xy_compile_recompute_unit,
    canonical_xy_compile_input_bytes,
    verify_xy_compile_recompute_unit,
    xy_compile_digest_python,
)

_studio_exactness = pytest.importorskip(
    "scpn_studio_platform.exactness",
    reason="Studio recompute exactness package requires Python >=3.12.",
)
ReproVerdict = _studio_exactness.ReproVerdict


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
