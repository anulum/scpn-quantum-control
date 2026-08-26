# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Rust replay/native-JIT decision-evidence tests
"""Tests for the bounded Rust replay/native-JIT decision-evidence harness."""

from __future__ import annotations

import json
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import scripts.run_rust_llvm_jit_decision_evidence as runner_module
import tools.rust_llvm_jit_decision_evidence as decision_module
from scpn_quantum_control.differentiable import (
    whole_program_value_and_grad as original_whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_rust_bridge import (
    RustProgramADValueAndGradientResult,
)
from scpn_quantum_control.whole_program_ad_result import WholeProgramADResult
from tools.rust_llvm_jit_decision_evidence import (
    ARTIFACT_ID_PREFIX,
    CLAIM_BOUNDARY,
    FOLLOW_UP_BOUNDARY,
    RATIONALE,
    SCHEMA,
    DecisionKernel,
    capture_decision_evidence,
    decision_kernels,
    inventory_matrix,
    validate_decision_evidence,
    write_decision_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    REPO_ROOT / "data/differentiable_phase_qnode/rust_llvm_jit_decision_evidence_20260726.json"
)
RUNNER_PATH = REPO_ROOT / "scripts/run_rust_llvm_jit_decision_evidence.py"


def _canonical_payload() -> dict[str, Any]:
    """Return a mutable copy of the committed evidence payload."""
    return cast(dict[str, Any], json.loads(ARTIFACT_PATH.read_text(encoding="utf-8")))


def _resign(payload: dict[str, Any]) -> None:
    """Refresh the embedded digest after a deliberate test mutation."""
    payload["sha256"] = decision_module._decision_evidence_digest(payload)


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
    """The comparison must remain a small, named, non-cherry-picked set."""
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
    observed = [float(kernel.objective(kernel.values)) for kernel in kernels]
    assert observed == pytest.approx(
        [-2.355782312762309, 3.7954080659812126, 20.0, 0.35, 0.5, 18.0]
    )


def test_inventory_has_no_unproven_rust_jit_product_gap() -> None:
    """The current role matrix must not invent a blocked product path."""
    inventory = inventory_matrix()
    assert inventory
    assert all(row["rust_jit_product_gap"] is False for row in inventory)


def test_committed_payload_is_canonical_and_valid() -> None:
    """The tracked evidence must satisfy the complete versioned contract."""
    payload = _canonical_payload()
    validate_decision_evidence(payload)
    assert payload["schema"] == SCHEMA
    assert payload["artifact_id"] == f"{ARTIFACT_ID_PREFIX}20260726"
    assert payload["claim_boundary"] == CLAIM_BOUNDARY
    assert payload["rationale"] == RATIONALE
    assert payload["follow_up_boundary"] == FOLLOW_UP_BOUNDARY


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "rust_llvm_jit_decision.v1", "unexpected .* schema"),
        ("stamp", "", "stamp must be non-empty"),
        ("stamp", 7, "stamp must be non-empty"),
        ("artifact_id", "wrong", "artifact ID"),
        ("claim_boundary", "wrong", "claim boundary"),
        ("rationale", "wrong", "rationale"),
        ("follow_up_boundary", "wrong", "follow-up boundary"),
        ("decision", "MAYBE", "decision must be"),
    ],
)
def test_validation_rejects_contract_metadata_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    """Every machine-facing contract name must fail closed on drift."""
    payload = _canonical_payload()
    payload[field] = value
    _resign(payload)
    with pytest.raises(ValueError, match=message):
        validate_decision_evidence(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"criteria": None}, "criteria and kernels are required"),
        ({"kernels": ()}, "criteria and kernels are required"),
        ({"inventory": []}, "non-empty inventory is required"),
        ({"inventory": "invalid"}, "non-empty inventory is required"),
        ({"kernels": []}, "kernel set must contain"),
        ({"kernels": [{"case_id": str(index)} for index in range(11)]}, "kernel set must contain"),
    ],
)
def test_validation_rejects_missing_or_unbounded_collections(
    mutation: dict[str, object],
    message: str,
) -> None:
    """Required evidence collections must keep their frozen shapes."""
    payload = _canonical_payload()
    payload.update(mutation)
    _resign(payload)
    with pytest.raises(ValueError, match=message):
        validate_decision_evidence(payload)


def test_validation_rejects_failed_parity() -> None:
    """No decision may validate after functional parity fails."""
    payload = _canonical_payload()
    cast(dict[str, object], payload["criteria"])["parity_passed"] = False
    _resign(payload)
    with pytest.raises(ValueError, match="requires functional parity"):
        validate_decision_evidence(payload)


def test_validation_requires_both_go_criteria() -> None:
    """A GO requires a product role and isolated performance evidence."""
    payload = _canonical_payload()
    payload["decision"] = "GO"
    _resign(payload)
    with pytest.raises(ValueError, match="proven product role"):
        validate_decision_evidence(payload)

    criteria = cast(dict[str, object], payload["criteria"])
    criteria["product_role_proven"] = True
    _resign(payload)
    with pytest.raises(ValueError, match="isolated performance evidence"):
        validate_decision_evidence(payload)

    criteria["isolated_performance_evidence"] = True
    _resign(payload)
    validate_decision_evidence(payload)


def test_validation_rejects_no_go_with_product_role() -> None:
    """A NO-GO cannot simultaneously claim that a product role is proven."""
    payload = _canonical_payload()
    cast(dict[str, object], payload["criteria"])["product_role_proven"] = True
    _resign(payload)
    with pytest.raises(ValueError, match="NO-GO cannot claim"):
        validate_decision_evidence(payload)


@pytest.mark.parametrize(
    "kernels",
    [
        ["not-a-mapping"],
        [{"case_id": ""}],
        [{"case_id": "same"}, {"case_id": "same"}],
    ],
)
def test_validation_rejects_invalid_kernel_identities(kernels: list[object]) -> None:
    """Kernel rows require unique, non-empty, string identities."""
    payload = _canonical_payload()
    payload["kernels"] = kernels
    _resign(payload)
    with pytest.raises(ValueError, match="unique non-empty mappings"):
        validate_decision_evidence(payload)


@pytest.mark.parametrize("digest", [None, "0" * 63, "G" * 64])
def test_validation_rejects_malformed_digest(digest: object) -> None:
    """The embedded digest must be exactly lowercase hexadecimal SHA-256."""
    payload = _canonical_payload()
    payload["sha256"] = digest
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        validate_decision_evidence(payload)


def test_validation_rejects_digest_mismatch() -> None:
    """Canonical-looking but incorrect digests must fail closed."""
    payload = _canonical_payload()
    payload["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_decision_evidence(payload)


def test_capture_produces_valid_no_go_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine-free aggregate contract must produce valid bounded evidence."""
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


@pytest.mark.parametrize(
    ("stamp", "rounds", "repetitions", "warmups", "message"),
    [
        (" ", 5, 200, 10, "stamp must be non-empty"),
        ("x", 2, 200, 10, "rounds must be"),
        ("x", 5, 0, 10, "rounds must be"),
        ("x", 5, 200, -1, "rounds must be"),
    ],
)
def test_capture_rejects_invalid_protocol(
    stamp: str,
    rounds: int,
    repetitions: int,
    warmups: int,
    message: str,
) -> None:
    """Capture parameters must remain non-empty and bounded."""
    with pytest.raises(ValueError, match=message):
        capture_decision_evidence(
            stamp=stamp,
            rounds=rounds,
            repetitions=repetitions,
            warmups=warmups,
        )


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


def test_capture_rejects_missing_program_ir(monkeypatch: pytest.MonkeyPatch) -> None:
    """A kernel without captured Program AD IR cannot enter the evidence set."""
    monkeypatch.setattr(
        decision_module,
        "whole_program_value_and_grad",
        lambda *_args, **_kwargs: SimpleNamespace(program_ir=None),
    )
    with pytest.raises(ValueError, match="did not emit Program AD IR"):
        decision_module._capture_kernel(
            decision_kernels()[0],
            rounds=3,
            repetitions=1,
            warmups=0,
        )


@pytest.mark.parametrize(("supported", "value"), [(False, 0.0), (True, None)])
def test_kernel_capture_rejects_incomplete_rust_result(
    monkeypatch: pytest.MonkeyPatch,
    supported: bool,
    value: float | None,
) -> None:
    """Both Rust support and a scalar value are mandatory."""
    traced = SimpleNamespace(
        program_ir=SimpleNamespace(effects=(object(),)),
        value=0.0,
        gradient=np.array([0.0]),
    )
    native = SimpleNamespace(value_and_grad=lambda _values: (0.0, np.array([0.0])))
    monkeypatch.setattr(decision_module, "whole_program_value_and_grad", lambda *_a: traced)
    monkeypatch.setattr(
        decision_module,
        "compile_whole_program_ad_trace_to_native_llvm_jit",
        lambda *_a: native,
    )
    monkeypatch.setattr(
        decision_module,
        "value_and_grad_program_ad_effect_ir_with_rust",
        lambda *_a: SimpleNamespace(
            supported=supported,
            value=value,
            gradient=np.array([0.0]),
            blocked_reasons=("blocked",),
        ),
    )
    with pytest.raises(ValueError, match="Rust replay blocked"):
        decision_module._capture_kernel(
            DecisionKernel("case", "scalar", lambda values: values[0], np.array([0.0])),
            rounds=3,
            repetitions=1,
            warmups=0,
        )


def test_kernel_capture_rejects_parity_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A numerical mismatch above tolerance must fail before timing."""
    traced = SimpleNamespace(
        program_ir=SimpleNamespace(effects=(object(),)),
        value=0.0,
        gradient=np.array([0.0]),
    )
    native = SimpleNamespace(value_and_grad=lambda _values: (1.0, np.array([0.0])))
    rust = SimpleNamespace(
        supported=True,
        value=0.0,
        gradient=np.array([0.0]),
        blocked_reasons=(),
    )
    monkeypatch.setattr(decision_module, "whole_program_value_and_grad", lambda *_a: traced)
    monkeypatch.setattr(
        decision_module,
        "compile_whole_program_ad_trace_to_native_llvm_jit",
        lambda *_a: native,
    )
    monkeypatch.setattr(
        decision_module,
        "value_and_grad_program_ad_effect_ir_with_rust",
        lambda *_a: rust,
    )
    with pytest.raises(ValueError, match="exceeded parity tolerance"):
        decision_module._capture_kernel(
            DecisionKernel("case", "scalar", lambda values: values[0], np.array([0.0])),
            rounds=3,
            repetitions=1,
            warmups=0,
        )


def test_capture_rejects_inconsistent_proven_role(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixed NO-GO must fail if inventory unexpectedly proves a product role."""
    _install_reference_replay_contract(monkeypatch)
    monkeypatch.setattr(
        decision_module,
        "inventory_matrix",
        lambda: ({"rust_jit_product_gap": True},),
    )
    with pytest.raises(ValueError, match="NO-GO cannot claim"):
        capture_decision_evidence(
            stamp="inconsistent-role",
            rounds=3,
            repetitions=1,
            warmups=0,
        )


def test_median_runtime_uses_warmups_repetitions_and_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The descriptive timer must use the declared bounded protocol."""
    ticks = iter((0, 10, 10, 30, 30, 60))
    calls = 0

    def callback() -> object:
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(time, "perf_counter_ns", lambda: next(ticks))
    assert (
        decision_module._median_runtime_ns(
            callback,
            rounds=3,
            repetitions=1,
            warmups=1,
        )
        == 20
    )
    assert calls == 4


def test_write_decision_evidence_is_atomic(tmp_path: Path) -> None:
    """The writer must replace its temporary file with canonical JSON."""
    payload = _canonical_payload()
    destination = tmp_path / "nested/evidence.json"
    assert write_decision_evidence(payload, destination) == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert not destination.with_suffix(".json.tmp").exists()


def test_runner_loads_repo_module_when_root_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct script execution must resolve the repository-local tool cleanly."""
    repository = str(runner_module.REPO_ROOT)
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != repository])
    assert runner_module._load_decision_evidence_module() is decision_module
    assert sys.path[0] == repository


def test_runner_exercises_real_entry_point(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The command-line entry point must capture and write through the renamed tool."""
    _install_reference_replay_contract(monkeypatch)
    destination = tmp_path / "runner-evidence.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--stamp",
            "runner-test",
            "--output",
            str(destination),
            "--rounds",
            "3",
            "--repetitions",
            "1",
            "--warmups",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(RUNNER_PATH), run_name="__main__")
    assert stopped.value.code == 0
    payload = cast(dict[str, object], json.loads(destination.read_text(encoding="utf-8")))
    validate_decision_evidence(payload)
    assert payload["artifact_id"] == f"{ARTIFACT_ID_PREFIX}runner-test"
