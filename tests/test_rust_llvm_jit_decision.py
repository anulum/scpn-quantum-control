# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-38 Rust LLVM/JIT decision tests
"""Tests for the bounded BL-38 decision harness."""

from __future__ import annotations

from typing import cast

import pytest

from tools.rust_llvm_jit_decision import (
    SCHEMA,
    capture_decision_evidence,
    decision_kernels,
    inventory_matrix,
    validate_decision_evidence,
)


def test_kernel_set_is_frozen_bounded_and_unique() -> None:
    """S38.2 must remain a small, named, non-cherry-picked comparison set."""
    kernels = decision_kernels()
    assert 1 <= len(kernels) <= 10
    assert len({row.case_id for row in kernels}) == len(kernels)
    assert {row.family for row in kernels} == {
        "scalar",
        "determinant",
        "inverse",
        "solve",
        "trace",
    }


def test_inventory_has_no_unproven_rust_jit_product_gap() -> None:
    """The current role matrix must not invent a blocked product path."""
    inventory = inventory_matrix()
    assert inventory
    assert all(row["rust_jit_product_gap"] is False for row in inventory)


def test_validation_fails_closed_for_unearned_go() -> None:
    """A GO cannot pass without both product role and isolated evidence."""
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "decision": "GO",
        "criteria": {
            "parity_passed": True,
            "product_role_proven": False,
            "isolated_performance_evidence": False,
        },
        "inventory": [{"surface": "x"}],
        "kernels": [{"case_id": "x"}],
    }
    with pytest.raises(ValueError, match="proven product role"):
        validate_decision_evidence(payload)


def test_capture_produces_valid_no_go_evidence() -> None:
    """The live installed engines must agree on every frozen comparison row."""
    payload = capture_decision_evidence(
        stamp="test",
        rounds=3,
        repetitions=1,
        warmups=0,
        isolated=False,
    )
    validate_decision_evidence(payload)
    assert payload["decision"] == "NO-GO"
    criteria = cast(dict[str, object], payload["criteria"])
    assert criteria["parity_passed"] is True
    assert criteria["performance_claim_made"] is False
    assert criteria["product_role_proven"] is False
    assert criteria["bl49_family_expansion"] is False
    kernels = cast(list[dict[str, object]], payload["kernels"])
    assert len(kernels) == len(decision_kernels())
    assert all(row["rust_replay_supported"] is True for row in kernels)
    assert all(row["native_jit_supported"] is True for row in kernels)
    assert isinstance(payload["sha256"], str)
    assert len(payload["sha256"]) == 64
