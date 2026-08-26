# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for adjoint reversible replay product
"""Real-surface tests for ``scpn_quantum_control.adjoint_replay_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.adjoint_replay_product as adjoint_replay_product
from scpn_quantum_control.adjoint_replay_product import (
    ADJOINT_REPLAY_CLAIM_BOUNDARY,
    ADJOINT_REPLAY_PRODUCT_SCHEMA,
    AdjointReplaySurfaceRow,
    CheckpointPolicy,
    MaterialisedAdjointReplayProbe,
    PathEligibilityDecision,
    ReversibilityReport,
    assert_adjoint_replay_product_integrity,
    assess_reversibility,
    build_adjoint_replay_product_registry,
    build_checkpoint_policy,
    decide_adjoint_replay_path,
    get_adjoint_replay_surface,
    iter_adjoint_replay_surfaces,
    list_adjoint_replay_surface_ids,
    map_adjoint_replay_public_surfaces,
    materialise_demo_adjoint_replay_probe,
)
from tools import adjoint_replay_product_quality_gates as adjoint_quality_gates


def test_adjoint_replay_quality_gate_spec_is_exact_and_focused() -> None:
    """The replay-product owner gate mirrors strict static and branch checks."""
    static_gates = dict(adjoint_quality_gates.build_static_quality_gates("python"))
    cohort = adjoint_quality_gates.ADJOINT_REPLAY_PRODUCT_QUALITY_RATCHET
    assert static_gates["mypy-strict-adjoint-replay-product-quality"][-len(cohort) :] == cohort
    assert static_gates["ruff D adjoint-replay-product quality ratchet"][-len(cohort) :] == cohort
    coverage_gates = adjoint_quality_gates.build_coverage_gates("python")
    assert "--branch" in coverage_gates[0][1]
    assert "--fail-under=100" in coverage_gates[1][1]
    assert "--include=*/adjoint_replay_product.py" in coverage_gates[1][1]


def test_list_surfaces_and_filters() -> None:
    """Surface discovery and filtering preserve canonical catalogue order."""
    ids = list_adjoint_replay_surface_ids()
    assert "reverse_adjoint_grad" in ids
    assert "executable_adjoint_replay" in ids
    assert "irreversible_mid_circuit_refuse" in ids
    assert "catalyst_hardware_adjoint_refuse" in ids
    assert ids == list_adjoint_replay_surface_ids()
    refuse = iter_adjoint_replay_surfaces(support_posture="refuse_only")
    assert refuse
    assert all(row.support_posture == "refuse_only" for row in refuse)
    replay = iter_adjoint_replay_surfaces(kind="executable_replay")
    assert len(replay) == 1


def test_get_known_and_unknown_fail_closed() -> None:
    """Known lookup returns a claim-bound row and unknown identifiers fail closed."""
    row = get_adjoint_replay_surface("reverse_adjoint_grad")
    assert row.claim_boundary == ADJOINT_REPLAY_CLAIM_BOUNDARY
    assert row.allows_catalyst_parity is False
    assert row.allows_hardware_adjoint is False
    with pytest.raises(ValueError, match="non-empty"):
        get_adjoint_replay_surface("  ")
    with pytest.raises(ValueError, match="unknown surface_id"):
        get_adjoint_replay_surface("not_a_surface")


def test_checkpoint_policy_and_reversibility() -> None:
    """Checkpoint schedules and reversibility reports enforce boundary invariants."""
    policy = build_checkpoint_policy(schedule="every_k", interval_k=2, max_checkpoints=4)
    assert isinstance(policy, CheckpointPolicy)
    assert policy.interval_k == 2
    with pytest.raises(ValueError, match="interval_k"):
        build_checkpoint_policy(interval_k=0)
    with pytest.raises(ValueError, match="schedule"):
        CheckpointPolicy(schedule=cast(Any, "nope"))

    ok = assess_reversibility(has_supported_unitary_ir=True)
    assert ok.reversible is True
    assert ok.supported_ops

    mid = assess_reversibility(has_mid_circuit_measurement=True)
    assert mid.reversible is False
    assert any("mid-circuit" in b.lower() for b in mid.blockers)

    irrev = assess_reversibility(has_irreversible_ops=True)
    assert irrev.reversible is False


def test_path_eligibility_refuse_and_allow() -> None:
    """Eligibility distinguishes allowed replay from Catalyst and hardware refusals."""
    allowed = decide_adjoint_replay_path(has_supported_unitary_ir=True)
    assert allowed.allowed is True
    assert allowed.outcome == "allowed"

    refused = decide_adjoint_replay_path(has_mid_circuit_measurement=True)
    assert refused.allowed is False
    assert refused.blockers

    cat = decide_adjoint_replay_path(request_catalyst_parity=True)
    assert cat.allowed is False
    assert any("catalyst" in b.lower() for b in cat.blockers)

    hw = decide_adjoint_replay_path(request_hardware_adjoint=True)
    assert hw.allowed is False
    assert any("hardware" in b.lower() or "qpu" in b.lower() for b in hw.blockers)


def test_materialise_demo_adjoint_replay_probe() -> None:
    """The quadratic replay agrees with the ambient adjoint gradient."""
    probe = materialise_demo_adjoint_replay_probe()
    assert probe.demo_label == "quadratic_sum_of_squares"
    assert probe.supported is True
    assert len(probe.adjoint_gradient) == 2
    assert abs(probe.adjoint_gradient[0] - 1.0) < 1e-9
    assert abs(probe.adjoint_gradient[1] - (-0.5)) < 1e-9
    assert abs(probe.replay_gradient[0] - 1.0) < 1e-9
    assert abs(probe.replay_gradient[1] - (-0.5)) < 1e-9
    assert probe.agreement_max_abs < 1e-9
    assert probe.replay_node_count > 0
    assert abs(probe.value - 0.3125) < 1e-12
    payload = probe.to_dict()
    assert payload["supported"] is True


def test_public_surfaces_and_registry() -> None:
    """Public mapping and integrity validation share one canonical catalogue."""
    surfaces = map_adjoint_replay_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.program_ad_adjoint" in paths
    assert "scpn_quantum_control.adjoint_replay_product" in paths

    registry = build_adjoint_replay_product_registry()
    assert registry["schema"] == ADJOINT_REPLAY_PRODUCT_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["default_surface_id"] == "reverse_adjoint_grad"
    validated = assert_adjoint_replay_product_integrity(registry)
    assert validated["surface_count"] == len(list_adjoint_replay_surface_ids())
    assert assert_adjoint_replay_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_invent_green() -> None:
    """Integrity rejects catalogue drift and unsupported claim promotion."""
    registry = build_adjoint_replay_product_registry()
    surfaces = cast(list[dict[str, object]], registry["surfaces"])

    stale_schema = dict(registry)
    stale_schema["schema"] = "adjoint_replay_product.v1"
    with pytest.raises(ValueError, match="schema mismatch"):
        assert_adjoint_replay_product_integrity(stale_schema)

    broken = dict(registry)
    broken["surfaces"] = surfaces + [
        {
            "surface_id": "ghost",
            "kind": "reverse_adjoint_grad",
            "title": "t",
            "summary": "s",
            "module_path": "m",
            "symbol_name": "x",
            "support_posture": "local_materialised",
            "allows_catalyst_parity": False,
            "allows_hardware_adjoint": False,
            "as_of": "2026-07-24",
            "claim_boundary": ADJOINT_REPLAY_CLAIM_BOUNDARY,
        }
    ]
    broken["surface_count"] = len(cast(list[object], broken["surfaces"]))
    with pytest.raises(ValueError, match="drift"):
        assert_adjoint_replay_product_integrity(broken)

    empty: dict[str, object] = {
        "schema": ADJOINT_REPLAY_PRODUCT_SCHEMA,
        "surfaces": [],
        "blank_entry_count": 0,
        "surface_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty surfaces"):
        assert_adjoint_replay_product_integrity(empty)

    cat = dict(registry)
    cat_rows = [dict(row) for row in surfaces]
    cat_rows[0]["allows_catalyst_parity"] = True
    cat["surfaces"] = cat_rows
    with pytest.raises(ValueError, match="Catalyst|allows_catalyst"):
        assert_adjoint_replay_product_integrity(cat)

    hw = dict(registry)
    hw_rows = [dict(row) for row in surfaces]
    hw_rows[0]["allows_hardware_adjoint"] = True
    hw["surfaces"] = hw_rows
    with pytest.raises(ValueError, match="hardware|allows_hardware"):
        assert_adjoint_replay_product_integrity(hw)


def test_integrity_rejects_blank_invalid() -> None:
    """Integrity rejects malformed rows and cross-field count drift."""
    registry = build_adjoint_replay_product_registry()
    surfaces = cast(list[dict[str, object]], registry["surfaces"])

    non_map = dict(registry)
    non_map["surfaces"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_adjoint_replay_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in surfaces]
    rows[0]["surface_id"] = "  "
    blank_id["surfaces"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_adjoint_replay_product_integrity(blank_id)

    bad_kind = dict(registry)
    krows = [dict(row) for row in surfaces]
    krows[1]["kind"] = "nope"
    bad_kind["surfaces"] = krows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_adjoint_replay_product_integrity(bad_kind)

    no_symbol = dict(registry)
    srows = [dict(row) for row in surfaces]
    srows[0]["symbol_name"] = ""
    no_symbol["surfaces"] = srows
    with pytest.raises(ValueError, match="symbol_name"):
        assert_adjoint_replay_product_integrity(no_symbol)

    no_default = dict(registry)
    renamed = [dict(row) for row in surfaces]
    for row in renamed:
        if row.get("surface_id") == "reverse_adjoint_grad":
            row["surface_id"] = "renamed"
    no_default["surfaces"] = renamed
    with pytest.raises(ValueError, match="missing reverse_adjoint_grad|drift"):
        assert_adjoint_replay_product_integrity(no_default)

    no_refuse = dict(registry)
    without = [
        dict(row) for row in surfaces if row.get("surface_id") != "irreversible_mid_circuit_refuse"
    ]
    no_refuse["surfaces"] = without
    no_refuse["surface_count"] = len(without)
    with pytest.raises(ValueError, match="missing irreversible_mid_circuit|drift"):
        assert_adjoint_replay_product_integrity(no_refuse)

    dup = dict(registry)
    drows = [dict(row) for row in surfaces]
    drows.append(dict(drows[0]))
    dup["surfaces"] = drows
    dup["surface_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate surface_id"):
        assert_adjoint_replay_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_adjoint_replay_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["surface_count"] = 0
    with pytest.raises(ValueError, match="surface_count"):
        assert_adjoint_replay_product_integrity(count_mismatch)


def test_module_exports() -> None:
    """The module exports replay discovery and execution entry points."""
    assert "materialise_demo_adjoint_replay_probe" in adjoint_replay_product.__all__
    assert "decide_adjoint_replay_path" in adjoint_replay_product.__all__
    assert "list_adjoint_replay_surface_ids" in adjoint_replay_product.__all__


def test_surface_row_and_probe_validation() -> None:
    """Public records and builders reject malformed product boundaries."""
    base: dict[str, Any] = {
        "surface_id": "x",
        "kind": "reverse_adjoint_grad",
        "title": "t",
        "summary": "s",
        "module_path": "m",
        "symbol_name": "fn",
        "support_posture": "local_materialised",
    }
    assert AdjointReplaySurfaceRow(**base).surface_id == "x"
    with pytest.raises(ValueError, match="surface_id"):
        AdjointReplaySurfaceRow(**{**base, "surface_id": ""})
    with pytest.raises(ValueError, match="kind"):
        AdjointReplaySurfaceRow(**{**base, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="allows_catalyst"):
        AdjointReplaySurfaceRow(**{**base, "allows_catalyst_parity": True})
    with pytest.raises(ValueError, match="allows_hardware"):
        AdjointReplaySurfaceRow(**{**base, "allows_hardware_adjoint": True})
    with pytest.raises(ValueError, match="title"):
        AdjointReplaySurfaceRow(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        AdjointReplaySurfaceRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="module_path"):
        AdjointReplaySurfaceRow(**{**base, "module_path": ""})
    with pytest.raises(ValueError, match="symbol_name"):
        AdjointReplaySurfaceRow(**{**base, "symbol_name": ""})
    with pytest.raises(ValueError, match="support_posture"):
        AdjointReplaySurfaceRow(**{**base, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="as_of"):
        AdjointReplaySurfaceRow(**{**base, "as_of": ""})

    with pytest.raises(ValueError, match="max_checkpoints"):
        build_checkpoint_policy(max_checkpoints=0)
    with pytest.raises(ValueError, match="reason"):
        ReversibilityReport(
            reversible=False,
            supported_ops=(),
            blockers=("b",),
            reason="",
        )
    with pytest.raises(ValueError, match="require blockers"):
        ReversibilityReport(
            reversible=False,
            supported_ops=(),
            blockers=(),
            reason="r",
        )
    with pytest.raises(ValueError, match="blockers entries"):
        ReversibilityReport(
            reversible=False,
            supported_ops=(),
            blockers=("",),
            reason="r",
        )
    with pytest.raises(ValueError, match="supported_ops entries"):
        ReversibilityReport(
            reversible=True,
            supported_ops=("",),
            blockers=(),
            reason="r",
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        ReversibilityReport(
            reversible=True,
            supported_ops=("a",),
            blockers=("x",),
            reason="r",
        )

    with pytest.raises(ValueError, match="outcome"):
        PathEligibilityDecision(
            outcome=cast(Any, "nope"),
            allowed=False,
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="reason"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="outcome=allowed"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=True,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="outcome=refused"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=False,
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="r",
            blockers=("x",),
        )
    with pytest.raises(ValueError, match="require blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="blockers entries"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("",),
        )

    with pytest.raises(ValueError, match="values must be non-empty"):
        MaterialisedAdjointReplayProbe(
            values=(),
            value=0.0,
            adjoint_gradient=(),
            replay_gradient=(),
            agreement_max_abs=0.0,
            replay_node_count=0,
            supported=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="adjoint_gradient must be non-empty"):
        MaterialisedAdjointReplayProbe(
            values=(1.0,),
            value=0.0,
            adjoint_gradient=(),
            replay_gradient=(0.0,),
            agreement_max_abs=0.0,
            replay_node_count=1,
            supported=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="adjoint_gradient length"):
        MaterialisedAdjointReplayProbe(
            values=(1.0, 2.0),
            value=0.0,
            adjoint_gradient=(0.0,),
            replay_gradient=(0.0, 0.0),
            agreement_max_abs=0.0,
            replay_node_count=1,
            supported=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="replay_gradient length"):
        MaterialisedAdjointReplayProbe(
            values=(1.0, 2.0),
            value=0.0,
            adjoint_gradient=(0.0, 0.0),
            replay_gradient=(0.0,),
            agreement_max_abs=0.0,
            replay_node_count=1,
            supported=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="value must be finite"):
        MaterialisedAdjointReplayProbe(
            values=(1.0,),
            value=float("nan"),
            adjoint_gradient=(0.0,),
            replay_gradient=(0.0,),
            agreement_max_abs=0.0,
            replay_node_count=1,
            supported=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="agreement_max_abs"):
        MaterialisedAdjointReplayProbe(
            values=(1.0,),
            value=0.0,
            adjoint_gradient=(0.0,),
            replay_gradient=(0.0,),
            agreement_max_abs=-0.1,
            replay_node_count=1,
            supported=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="replay_node_count"):
        MaterialisedAdjointReplayProbe(
            values=(1.0,),
            value=0.0,
            adjoint_gradient=(0.0,),
            replay_gradient=(0.0,),
            agreement_max_abs=0.0,
            replay_node_count=-1,
            supported=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedAdjointReplayProbe(
            values=(1.0,),
            value=0.0,
            adjoint_gradient=(0.0,),
            replay_gradient=(0.0,),
            agreement_max_abs=0.0,
            replay_node_count=1,
            supported=True,
            demo_label="",
        )
    with pytest.raises(ValueError, match="supported ambient"):
        MaterialisedAdjointReplayProbe(
            values=(1.0,),
            value=0.0,
            adjoint_gradient=(0.0,),
            replay_gradient=(0.0,),
            agreement_max_abs=0.0,
            replay_node_count=1,
            supported=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="values"):
        materialise_demo_adjoint_replay_probe(values=[1.0])
    with pytest.raises(ValueError, match="finite"):
        materialise_demo_adjoint_replay_probe(values=[float("nan"), 1.0])
    no_ir = assess_reversibility(has_supported_unitary_ir=False)
    assert no_ir.reversible is False
    ok_payload = build_checkpoint_policy(schedule="binomial").to_dict()
    assert ok_payload["schedule"] == "binomial"
    assert decide_adjoint_replay_path().to_dict()["allowed"] is True


def test_reversibility_report_to_dict() -> None:
    """ReversibilityReport.to_dict exposes the public contract fields."""
    report = assess_reversibility(has_supported_unitary_ir=True)
    payload = report.to_dict()
    assert payload["reversible"] is True
    assert payload["blockers"] == []
    assert "supported_ops" in payload
    assert payload["reason"]
    assert payload["claim_boundary"] == ADJOINT_REPLAY_CLAIM_BOUNDARY


def test_catalogue_map_rejects_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_catalogue_map`` refuses an empty canonical catalogue."""
    monkeypatch.setattr(adjoint_replay_product, "_CANONICAL_SURFACES", ())
    with pytest.raises(RuntimeError, match="catalogue must be non-empty"):
        adjoint_replay_product._catalogue_map()


def test_catalogue_map_rejects_blank_surface_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_catalogue_map`` refuses a blank surface_id after construction."""
    from dataclasses import replace

    # Fresh row so live catalogue objects are never mutated.
    blank = replace(get_adjoint_replay_surface("reverse_adjoint_grad"))
    object.__setattr__(blank, "surface_id", "  ")
    monkeypatch.setattr(adjoint_replay_product, "_CANONICAL_SURFACES", (blank,))
    with pytest.raises(RuntimeError, match="blank surface_id"):
        adjoint_replay_product._catalogue_map()


def test_catalogue_map_rejects_duplicate_surface_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_catalogue_map`` refuses duplicate surface identifiers."""
    from dataclasses import replace

    good = replace(get_adjoint_replay_surface("reverse_adjoint_grad"))
    monkeypatch.setattr(adjoint_replay_product, "_CANONICAL_SURFACES", (good, good))
    with pytest.raises(RuntimeError, match="duplicate surface_id"):
        adjoint_replay_product._catalogue_map()


def test_materialise_refuses_when_path_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Materialised demo fails closed when path eligibility refuses."""

    def _refuse(**_kwargs: Any) -> PathEligibilityDecision:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="forced refuse",
            blockers=("forced",),
        )

    monkeypatch.setattr(adjoint_replay_product, "decide_adjoint_replay_path", _refuse)
    with pytest.raises(ValueError, match="demo path refused"):
        materialise_demo_adjoint_replay_probe()


def test_materialise_refuses_missing_adjoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Materialised demo fails closed when whole-program adjoint metadata is absent."""
    from dataclasses import replace

    from scpn_quantum_control.differentiable import whole_program_value_and_grad

    real = whole_program_value_and_grad

    def _no_adjoint(objective: Any, values: Any, **kwargs: Any) -> Any:
        result = real(objective, values, **kwargs)
        return replace(result, adjoint_result=None)

    monkeypatch.setattr(
        "scpn_quantum_control.differentiable.whole_program_value_and_grad",
        _no_adjoint,
    )
    with pytest.raises(ValueError, match="missing adjoint metadata"):
        materialise_demo_adjoint_replay_probe()


def test_materialise_refuses_unsupported_adjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Materialised demo fails closed when ambient adjoint generation is unsupported."""
    from dataclasses import replace

    from scpn_quantum_control.differentiable import whole_program_value_and_grad

    real = whole_program_value_and_grad

    def _unsupported(objective: Any, values: Any, **kwargs: Any) -> Any:
        result = real(objective, values, **kwargs)
        assert result.adjoint_result is not None
        return replace(
            result,
            adjoint_result=replace(
                result.adjoint_result,
                supported=False,
                unsupported_ops=("fake_op",),
            ),
        )

    monkeypatch.setattr(
        "scpn_quantum_control.differentiable.whole_program_value_and_grad",
        _unsupported,
    )
    with pytest.raises(ValueError, match="unsupported for demo objective"):
        materialise_demo_adjoint_replay_probe()
