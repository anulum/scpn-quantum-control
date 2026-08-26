# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Rust-JIT-decision Rust LLVM/JIT decision tests
"""Tests for the bounded Rust-JIT decision decision harness."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import tools.rust_llvm_jit_decision as decision_module
from scpn_quantum_control.differentiable import (
    whole_program_value_and_grad as original_whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_rust_bridge import (
    RustProgramADValueAndGradientResult,
)
from scpn_quantum_control.whole_program_ad_result import WholeProgramADResult
from tools.rust_llvm_jit_decision import (
    SCHEMA,
    capture_decision_evidence,
    decision_kernels,
    inventory_matrix,
    validate_decision_evidence,
)


def _native_rust_replay_available() -> bool:
    """Return whether the optional extension exposes the required replay API."""
    try:
        import scpn_quantum_engine as engine
    except ModuleNotFoundError:
        return False
    return callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))


def _install_reference_replay_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a deterministic test-only aggregate replay contract.

    The production decision harness remains fail-closed. This seam lets the
    deliberately engine-free reproduction image validate evidence assembly,
    native-JIT parity, and schema invariants against the canonical Python
    trace result without claiming that Rust executed there.
    """
    traced_results: dict[int, WholeProgramADResult] = {}

    def trace_and_remember(*args: Any, **kwargs: Any) -> WholeProgramADResult:
        traced = original_whole_program_value_and_grad(*args, **kwargs)
        program_ir = traced.program_ir
        if program_ir is not None:
            traced_results[id(program_ir)] = traced
        return traced

    def replay_reference(
        program_ir: object,
        _inputs: object,
    ) -> RustProgramADValueAndGradientResult:
        traced = traced_results.get(id(program_ir))
        if traced is None:
            raise AssertionError("test replay contract did not observe the Program AD trace")
        if traced.program_ir is None:
            raise AssertionError("test replay contract observed an empty Program AD trace")
        gradient = np.asarray(traced.gradient, dtype=np.float64)
        effect_count = len(traced.program_ir.effects)
        return RustProgramADValueAndGradientResult(
            supported=True,
            value=float(traced.value),
            gradient=gradient,
            parameter_targets=tuple(f"input[{index}]" for index in range(gradient.size)),
            effect_count=effect_count,
            supported_effect_count=effect_count,
            blocked_reasons=(),
            claim_boundary="test-only canonical Python aggregate replay contract",
        )

    monkeypatch.setattr(decision_module, "whole_program_value_and_grad", trace_and_remember)
    monkeypatch.setattr(
        decision_module,
        "value_and_grad_program_ad_effect_ir_with_rust",
        replay_reference,
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


def test_capture_produces_valid_no_go_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installed engines, or the engine-free image contract, must agree."""
    if not _native_rust_replay_available():
        _install_reference_replay_contract(monkeypatch)
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
    assert criteria["parity_certificate_family_expansion"] is False
    kernels = cast(list[dict[str, object]], payload["kernels"])
    assert len(kernels) == len(decision_kernels())
    assert all(row["rust_replay_supported"] is True for row in kernels)
    assert all(row["native_jit_supported"] is True for row in kernels)
    assert isinstance(payload["sha256"], str)
    assert len(payload["sha256"]) == 64


def test_engine_free_image_validates_aggregate_capture_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reproduction image must exercise capture without native Rust."""
    _install_reference_replay_contract(monkeypatch)
    payload = capture_decision_evidence(
        stamp="test-engine-free-contract",
        rounds=3,
        repetitions=1,
        warmups=0,
        isolated=False,
    )
    validate_decision_evidence(payload)
    assert payload["decision"] == "NO-GO"


def test_capture_remains_fail_closed_for_unsupported_rust_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production harness must never promote an unsupported Rust replay."""
    monkeypatch.setattr(
        decision_module,
        "value_and_grad_program_ad_effect_ir_with_rust",
        lambda *_args, **_kwargs: RustProgramADValueAndGradientResult(
            supported=False,
            value=None,
            gradient=np.array([], dtype=np.float64),
            parameter_targets=(),
            effect_count=0,
            supported_effect_count=0,
            blocked_reasons=("native replay unavailable",),
            claim_boundary="test-only unsupported replay contract",
        ),
    )
    with pytest.raises(ValueError, match="Rust replay blocked"):
        capture_decision_evidence(
            stamp="test-fail-closed",
            rounds=3,
            repetitions=1,
            warmups=0,
            isolated=False,
        )
