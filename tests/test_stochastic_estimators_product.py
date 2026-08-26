# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for stochastic estimators product
"""Real-surface tests for ``scpn_quantum_control.stochastic_estimators_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.stochastic_estimators_product as stochastic_estimators_product
from scpn_quantum_control.differentiable_stochastic_policy import GradientFailurePolicy
from scpn_quantum_control.stochastic_estimators_product import (
    STOCHASTIC_ESTIMATORS_CLAIM_BOUNDARY,
    STOCHASTIC_ESTIMATORS_PRODUCT_SCHEMA,
    EstimatorDryRunDecision,
    MaterialisedSPSAProbe,
    StochasticEstimatorRow,
    assert_stochastic_estimators_product_integrity,
    build_product_failure_policy,
    build_stochastic_estimators_product_registry,
    dry_run_stochastic_estimator,
    get_stochastic_estimator,
    iter_stochastic_estimators,
    list_stochastic_estimator_ids,
    map_stochastic_estimators_public_surfaces,
    materialise_demo_spsa_probe,
)


def _registry_estimators(registry: dict[str, object]) -> list[dict[str, object]]:
    """Narrow a validated registry estimator collection for drift fixtures."""
    raw = registry["estimators"]
    assert isinstance(raw, list)
    return cast(list[dict[str, object]], raw)


def test_list_estimators_and_filters() -> None:
    """Expose stable estimator ids and deterministic catalogue filters."""
    ids = list_stochastic_estimator_ids()
    assert "spsa_gradient" in ids
    assert "score_function_gradient" in ids
    assert "parameter_shift_shot_allocation" in ids
    assert "gradient_failure_policy" in ids
    assert ids == list_stochastic_estimator_ids()
    spsa = iter_stochastic_estimators(kind="spsa")
    assert len(spsa) == 1
    finite = iter_stochastic_estimators(support_posture="finite_shot_materialised")
    assert finite
    assert all(row.support_posture == "finite_shot_materialised" for row in finite)


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve known estimators while rejecting blank and unknown ids."""
    row = get_stochastic_estimator("spsa_gradient")
    assert row.allows_hardware_shots is False
    assert row.claim_boundary == STOCHASTIC_ESTIMATORS_CLAIM_BOUNDARY
    assert (
        "hardware_safety_audit" in row.hardware_safety_pointer
        or "hardware" in row.hardware_safety_pointer
    )
    with pytest.raises(ValueError, match="non-empty"):
        get_stochastic_estimator("  ")
    with pytest.raises(ValueError, match="unknown estimator_id"):
        get_stochastic_estimator("not_an_estimator")


def test_dry_run_allowed_and_hardware_refuse() -> None:
    """Allow local shot planning while refusing hardware execution."""
    decision = dry_run_stochastic_estimator("spsa_gradient", planned_shots=50)
    assert decision.allowed is True
    assert decision.outcome == "allowed_dry_run"
    assert decision.planned_shots == 50
    assert "no QPU submission" in decision.reason

    refused = dry_run_stochastic_estimator(
        "spsa_gradient",
        request_hardware_shots=True,
    )
    assert refused.allowed is False
    assert refused.blockers
    assert any("hardware" in item.lower() or "qpu" in item.lower() for item in refused.blockers)


def test_dry_run_invalid_shots() -> None:
    """Reject non-positive and non-integer dry-run shot budgets."""
    with pytest.raises(ValueError, match="planned_shots"):
        dry_run_stochastic_estimator("score_function_gradient", planned_shots=0)
    with pytest.raises(ValueError, match="planned_shots"):
        dry_run_stochastic_estimator(
            "score_function_gradient",
            planned_shots=cast(Any, 1.5),
        )


def test_product_failure_policy() -> None:
    """Build the ambient fail-closed confidence policy through the product."""
    policy = build_product_failure_policy(max_standard_error=0.1)
    assert isinstance(policy, GradientFailurePolicy)
    assert policy.max_standard_error == 0.1
    assert policy.require_trainable is True


def test_materialise_demo_spsa_probe() -> None:
    """Materialise a deterministic local SPSA product probe."""
    probe = materialise_demo_spsa_probe(seed=1, repetitions=3)
    assert probe.gradient
    assert len(probe.gradient) == 2
    assert probe.seed == 1
    assert probe.repetitions == 3
    assert probe.shots is None
    assert probe.max_abs_gradient >= 0.0
    # true grad of sum(x^2) at [0.5, -0.25] is [1.0, -0.5]; SPSA is noisy but
    # must return a real finite gradient vector
    assert all(abs(v) < 1e6 for v in probe.gradient)
    payload = probe.to_dict()
    assert payload["max_abs_gradient"] == probe.max_abs_gradient


def test_public_surfaces_and_registry() -> None:
    """Map ambient owners and validate the complete estimator registry."""
    surfaces = map_stochastic_estimators_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.differentiable_stochastic_estimators" in paths
    assert "scpn_quantum_control.differentiable_stochastic_policy" in paths

    registry = build_stochastic_estimators_product_registry()
    assert registry["schema"] == STOCHASTIC_ESTIMATORS_PRODUCT_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["default_estimator_id"] == "spsa_gradient"
    validated = assert_stochastic_estimators_product_integrity(registry)
    assert validated["estimator_count"] == len(list_stochastic_estimator_ids())
    assert assert_stochastic_estimators_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_hardware() -> None:
    """Reject estimator-set drift and hardware-shot claim relaxation."""
    registry = build_stochastic_estimators_product_registry()
    estimators = _registry_estimators(registry)

    stale_schema = dict(registry)
    stale_schema["schema"] = "stochastic_estimators_product.v1"
    with pytest.raises(ValueError, match="schema mismatch"):
        assert_stochastic_estimators_product_integrity(stale_schema)

    broken = dict(registry)
    broken["estimators"] = estimators + [
        {
            "estimator_id": "ghost",
            "kind": "spsa",
            "title": "t",
            "summary": "s",
            "module_path": "m",
            "symbol_name": "x",
            "support_posture": "local_materialised",
            "allows_hardware_shots": False,
            "hardware_safety_pointer": "p",
            "as_of": "2026-07-24",
            "claim_boundary": STOCHASTIC_ESTIMATORS_CLAIM_BOUNDARY,
        }
    ]
    broken["estimator_count"] = len(cast(list[object], broken["estimators"]))
    with pytest.raises(ValueError, match="drift"):
        assert_stochastic_estimators_product_integrity(broken)

    empty: dict[str, object] = {
        "schema": STOCHASTIC_ESTIMATORS_PRODUCT_SCHEMA,
        "estimators": [],
        "blank_entry_count": 0,
        "estimator_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty estimators"):
        assert_stochastic_estimators_product_integrity(empty)

    hw = dict(registry)
    hw_rows = [dict(row) for row in estimators]
    hw_rows[0]["allows_hardware_shots"] = True
    hw["estimators"] = hw_rows
    with pytest.raises(ValueError, match="invent-green hardware|allows_hardware"):
        assert_stochastic_estimators_product_integrity(hw)


def test_integrity_rejects_blank_invalid() -> None:
    """Reject malformed rows, missing defaults, duplicates, and count drift."""
    registry = build_stochastic_estimators_product_registry()
    estimators = _registry_estimators(registry)

    non_map = dict(registry)
    non_map["estimators"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_stochastic_estimators_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in estimators]
    rows[0]["estimator_id"] = "  "
    blank_id["estimators"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_stochastic_estimators_product_integrity(blank_id)

    bad_kind = dict(registry)
    krows = [dict(row) for row in estimators]
    krows[1]["kind"] = "nope"
    bad_kind["estimators"] = krows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_stochastic_estimators_product_integrity(bad_kind)

    no_symbol = dict(registry)
    srows = [dict(row) for row in estimators]
    srows[0]["symbol_name"] = ""
    no_symbol["estimators"] = srows
    with pytest.raises(ValueError, match="symbol_name"):
        assert_stochastic_estimators_product_integrity(no_symbol)

    no_default = dict(registry)
    renamed = [dict(row) for row in estimators]
    for row in renamed:
        if row.get("estimator_id") == "spsa_gradient":
            row["estimator_id"] = "renamed"
    no_default["estimators"] = renamed
    with pytest.raises(ValueError, match="missing spsa_gradient|drift"):
        assert_stochastic_estimators_product_integrity(no_default)

    dup = dict(registry)
    drows = [dict(row) for row in estimators]
    drows.append(dict(drows[0]))
    dup["estimators"] = drows
    dup["estimator_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate estimator_id"):
        assert_stochastic_estimators_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_stochastic_estimators_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["estimator_count"] = 0
    with pytest.raises(ValueError, match="estimator_count"):
        assert_stochastic_estimators_product_integrity(count_mismatch)


def test_module_exports() -> None:
    """Keep every documented stochastic product entry point public."""
    assert "dry_run_stochastic_estimator" in stochastic_estimators_product.__all__
    assert "materialise_demo_spsa_probe" in stochastic_estimators_product.__all__
    assert "list_stochastic_estimator_ids" in stochastic_estimators_product.__all__


def test_estimator_row_validation() -> None:
    """Enforce immutable estimator catalogue-row invariants."""
    base: dict[str, Any] = {
        "estimator_id": "x",
        "kind": "spsa",
        "title": "t",
        "summary": "s",
        "module_path": "m",
        "symbol_name": "fn",
        "support_posture": "local_materialised",
    }
    assert StochasticEstimatorRow(**base).estimator_id == "x"
    with pytest.raises(ValueError, match="estimator_id"):
        StochasticEstimatorRow(**{**base, "estimator_id": ""})
    with pytest.raises(ValueError, match="kind"):
        StochasticEstimatorRow(**{**base, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="title"):
        StochasticEstimatorRow(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        StochasticEstimatorRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="module_path"):
        StochasticEstimatorRow(**{**base, "module_path": ""})
    with pytest.raises(ValueError, match="symbol_name"):
        StochasticEstimatorRow(**{**base, "symbol_name": ""})
    with pytest.raises(ValueError, match="support_posture"):
        StochasticEstimatorRow(**{**base, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="allows_hardware_shots=False"):
        StochasticEstimatorRow(**{**base, "allows_hardware_shots": True})
    with pytest.raises(ValueError, match="as_of"):
        StochasticEstimatorRow(**{**base, "as_of": ""})
    with pytest.raises(ValueError, match="hardware_safety_pointer"):
        StochasticEstimatorRow(**{**base, "hardware_safety_pointer": ""})


def test_decision_invariants() -> None:
    """Enforce dry-run outcome, blocker, reason, and budget invariants."""
    with pytest.raises(ValueError, match="estimator_id"):
        EstimatorDryRunDecision(
            estimator_id="",
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("b",),
            planned_shots=0,
        )
    with pytest.raises(ValueError, match="outcome"):
        EstimatorDryRunDecision(
            estimator_id="e",
            outcome=cast(Any, "nope"),
            allowed=False,
            reason="r",
            blockers=("b",),
            planned_shots=0,
        )
    with pytest.raises(ValueError, match="require blockers"):
        EstimatorDryRunDecision(
            estimator_id="e",
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
            planned_shots=0,
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        EstimatorDryRunDecision(
            estimator_id="e",
            outcome="allowed_dry_run",
            allowed=True,
            reason="r",
            blockers=("b",),
            planned_shots=10,
        )
    with pytest.raises(ValueError, match="must use outcome=allowed_dry_run"):
        EstimatorDryRunDecision(
            estimator_id="e",
            outcome="refused",
            allowed=True,
            reason="r",
            blockers=(),
            planned_shots=10,
        )
    with pytest.raises(ValueError, match="must use outcome=refused"):
        EstimatorDryRunDecision(
            estimator_id="e",
            outcome="allowed_dry_run",
            allowed=False,
            reason="r",
            blockers=("b",),
            planned_shots=0,
        )
    with pytest.raises(ValueError, match="reason"):
        EstimatorDryRunDecision(
            estimator_id="e",
            outcome="refused",
            allowed=False,
            reason="",
            blockers=("b",),
            planned_shots=0,
        )
    with pytest.raises(ValueError, match="blockers entries"):
        EstimatorDryRunDecision(
            estimator_id="e",
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("ok", "  "),
            planned_shots=0,
        )
    with pytest.raises(ValueError, match="planned_shots"):
        EstimatorDryRunDecision(
            estimator_id="e",
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("b",),
            planned_shots=-1,
        )
    with pytest.raises(ValueError, match="positive planned_shots"):
        EstimatorDryRunDecision(
            estimator_id="e",
            outcome="allowed_dry_run",
            allowed=True,
            reason="r",
            blockers=(),
            planned_shots=0,
        )
    ok = dry_run_stochastic_estimator("parameter_shift_shot_allocation")
    assert ok.to_dict()["allowed"] is True


def test_materialised_probe_validation() -> None:
    """Enforce materialised SPSA result invariants and serialization."""
    ok = MaterialisedSPSAProbe(
        gradient=(1.0, -0.5),
        seed=0,
        repetitions=1,
        shots=None,
        max_abs_gradient=1.0,
    )
    assert ok.to_dict()["seed"] == 0
    with pytest.raises(ValueError, match="gradient"):
        MaterialisedSPSAProbe(
            gradient=(),
            seed=0,
            repetitions=1,
            shots=None,
            max_abs_gradient=0.0,
        )
    with pytest.raises(ValueError, match="seed"):
        MaterialisedSPSAProbe(
            gradient=(1.0,),
            seed=-1,
            repetitions=1,
            shots=None,
            max_abs_gradient=1.0,
        )
    with pytest.raises(ValueError, match="repetitions"):
        MaterialisedSPSAProbe(
            gradient=(1.0,),
            seed=0,
            repetitions=0,
            shots=None,
            max_abs_gradient=1.0,
        )
    with pytest.raises(ValueError, match="shots"):
        MaterialisedSPSAProbe(
            gradient=(1.0,),
            seed=0,
            repetitions=1,
            shots=0,
            max_abs_gradient=1.0,
        )
    with pytest.raises(ValueError, match="max_abs_gradient"):
        MaterialisedSPSAProbe(
            gradient=(1.0,),
            seed=0,
            repetitions=1,
            shots=None,
            max_abs_gradient=-0.1,
        )


def test_catalogue_map_runtime_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject blank, duplicate, and empty runtime catalogues."""
    from scpn_quantum_control import stochastic_estimators_product as mod

    good = get_stochastic_estimator("spsa_gradient")
    blank = StochasticEstimatorRow(
        estimator_id="tmp",
        kind="spsa",
        title="t",
        summary="s",
        module_path="m",
        symbol_name="fn",
        support_posture="local_materialised",
    )
    object.__setattr__(blank, "estimator_id", "  ")
    monkeypatch.setattr(mod, "_CANONICAL_ESTIMATORS", (blank,))
    with pytest.raises(RuntimeError, match="blank estimator_id"):
        mod._catalogue_map()

    a = get_stochastic_estimator("spsa_gradient")
    monkeypatch.setattr(mod, "_CANONICAL_ESTIMATORS", (a, a))
    with pytest.raises(RuntimeError, match="duplicate estimator_id"):
        mod._catalogue_map()

    monkeypatch.setattr(mod, "_CANONICAL_ESTIMATORS", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        mod._catalogue_map()

    monkeypatch.setattr(mod, "_CANONICAL_ESTIMATORS", (good,))
    assert mod._catalogue_map()[good.estimator_id].estimator_id == good.estimator_id


def test_row_to_dict() -> None:
    """Serialize catalogue rows without relaxing hardware boundaries."""
    row = get_stochastic_estimator("gradient_failure_policy")
    payload = row.to_dict()
    assert payload["kind"] == "confidence_policy"
    assert payload["allows_hardware_shots"] is False


def test_materialise_demo_spsa_probe_rejects_empty_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty ambient SPSA gradient is refused (fail closed)."""
    from types import SimpleNamespace

    import scpn_quantum_control.differentiable_stochastic_estimators as ambient

    def empty_gradient(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(gradient=())

    monkeypatch.setattr(ambient, "spsa_gradient_estimate", empty_gradient)
    with pytest.raises(ValueError, match="SPSA probe returned empty gradient"):
        materialise_demo_spsa_probe()
