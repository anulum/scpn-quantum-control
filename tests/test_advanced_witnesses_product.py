# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for advanced witnesses product (BL-44)
"""Behaviour tests for :mod:`scpn_quantum_control.advanced_witnesses_product`."""

from __future__ import annotations

import math

import pytest

from scpn_quantum_control.advanced_witnesses_product import (
    ADVANCED_WITNESSES_CLAIM_BOUNDARY,
    ADVANCED_WITNESSES_PRODUCT_SCHEMA,
    MAX_DEMO_SHADOW_SHOTS,
    MAX_WITNESS_QUBITS,
    MIN_SHADOW_SHOTS,
    WITNESS_GLOSSARY,
    MaterialisedKrylovProbe,
    MaterialisedOtocProbe,
    MaterialisedShadowProbe,
    PathEligibilityDecision,
    WitnessBoundaryRow,
    WitnessCapabilityRow,
    WitnessEstimate,
    assert_advanced_witnesses_product_integrity,
    build_advanced_witnesses_product_registry,
    decide_witness_path,
    get_witness_boundary,
    get_witness_capability,
    get_witness_glossary_entry,
    iter_witness_boundaries,
    iter_witness_capabilities,
    list_witness_ambient_inventory,
    list_witness_boundary_ids,
    list_witness_capability_ids,
    list_witness_glossary_keys,
    map_advanced_witnesses_public_surfaces,
    materialise_bl18_order_parameter_compose,
    materialise_demo_krylov_probe,
    materialise_demo_otoc_probe,
    materialise_demo_shadow_probe,
    materialise_krylov_probe,
    materialise_otoc_probe,
    materialise_shadow_probe,
)


def test_registry_integrity_and_schema() -> None:
    """Built registry must pass integrity and expose schema v1."""
    reg = assert_advanced_witnesses_product_integrity(build_advanced_witnesses_product_registry())
    assert reg["schema"] == ADVANCED_WITNESSES_PRODUCT_SCHEMA
    assert reg["hardware_submit_allowed_policy"] is False
    assert reg["otoc_advantage_claim_policy"] is False
    assert reg["topology_certification_policy"] is False
    assert reg["live_qpu_witness_policy"] is False
    assert reg["unrestricted_shadow_tomography_policy"] is False
    assert reg["under_sampled_silent_green_policy"] is False
    assert reg["capability_count"] == len(list_witness_capability_ids())
    assert reg["boundary_count"] == len(list_witness_boundary_ids())
    assert reg["blank_entry_count"] == 0
    assert reg["max_witness_qubits"] == MAX_WITNESS_QUBITS


def test_assert_integrity_none_builds_fresh() -> None:
    """Integrity helper builds a registry when called with None."""
    reg = assert_advanced_witnesses_product_integrity(None)
    assert reg["schema"] == ADVANCED_WITNESSES_PRODUCT_SCHEMA


def test_capability_catalogue_covers_core_kinds() -> None:
    """Capability catalogue includes Krylov, OTOC, shadows, and BL-18 compose."""
    ids = list_witness_capability_ids()
    assert "krylov_complexity" in ids
    assert "otoc_probe" in ids
    assert "classical_shadows" in ids
    assert "bl18_sync_compose" in ids
    for row in iter_witness_capabilities():
        assert row.hardware_submit_allowed is False
        assert row.capability_id
        d = row.to_dict()
        assert d["capability_id"] == row.capability_id
    row = get_witness_capability("otoc_probe")
    assert row.kind == "otoc_probe"
    assert "compute_otoc" in row.ambient_symbol


def test_get_capability_unknown_raises() -> None:
    """Unknown capability id raises KeyError."""
    with pytest.raises(KeyError, match="unknown witness capability"):
        get_witness_capability("not_a_real_capability")


def test_boundary_catalogue_fail_closed() -> None:
    """All hard-gap boundaries are fail-closed with known kinds."""
    ids = list_witness_boundary_ids()
    assert "otoc_advantage_claim" in ids
    assert "topology_certification" in ids
    assert "live_qpu_witness" in ids
    for row in iter_witness_boundaries():
        assert row.fail_closed is True
        assert row.to_dict()["fail_closed"] is True
    b = get_witness_boundary("live_qpu_witness")
    assert b.kind == "live_qpu_witness"


def test_get_boundary_unknown_raises() -> None:
    """Unknown boundary id raises KeyError."""
    with pytest.raises(KeyError, match="unknown witness boundary"):
        get_witness_boundary("nope")


def test_glossary_and_inventory() -> None:
    """Glossary keys and ambient inventory are non-empty and coherent."""
    keys = list_witness_glossary_keys()
    assert "Krylov" in keys
    assert "OTOC" in keys
    assert "classical_shadow" in keys
    entry = get_witness_glossary_entry("OTOC")
    assert "correlator" in entry.lower() or "OTOC" in entry or "order" in entry.lower()
    with pytest.raises(KeyError):
        get_witness_glossary_entry("not_a_glossary_key")
    inv = list_witness_ambient_inventory()
    assert len(inv) >= 4
    modules = {row["module"] for row in inv}
    assert any("krylov" in m for m in modules)
    assert any("otoc" in m for m in modules)
    assert any("shadow" in m for m in modules)
    surfaces = map_advanced_witnesses_public_surfaces()
    assert "krylov_probe" in surfaces
    assert "product_registry" in surfaces


def test_decide_path_allows_local_and_refuses_invent_green() -> None:
    """Path decisions allow local research and refuse invent-green flags."""
    ok = decide_witness_path("krylov")
    assert ok.allowed is True
    assert ok.outcome == "allowed"
    assert ok.to_dict()["allowed"] is True

    r1 = decide_witness_path("otoc", invent_green_otoc_advantage=True)
    assert r1.allowed is False
    assert r1.invent_green_refused is True

    r2 = decide_witness_path("shadow", invent_green_topology_cert=True)
    assert r2.allowed is False
    assert r2.invent_green_refused is True

    r3 = decide_witness_path("krylov", invent_green_live_qpu=True)
    assert r3.allowed is False
    assert r3.invent_green_refused is True

    r4 = decide_witness_path("shadow", unrestricted_shadow=True)
    assert r4.allowed is False
    assert r4.invent_green_refused is True

    r5 = decide_witness_path("otoc", n_qubits=MAX_WITNESS_QUBITS + 1)
    assert r5.allowed is False
    assert r5.invent_green_refused is False

    r6 = decide_witness_path("not_a_path")
    assert r6.allowed is False


def test_decide_path_empty_raises() -> None:
    """Empty path_id raises ValueError."""
    with pytest.raises(ValueError, match="path_id"):
        decide_witness_path("")


def test_materialise_krylov_probe_real_ambient() -> None:
    """Krylov probe exercises ambient krylov_complexity and is digest-stable."""
    p1 = materialise_demo_krylov_probe()
    p2 = materialise_demo_krylov_probe()
    assert isinstance(p1, MaterialisedKrylovProbe)
    assert p1.digest == p2.digest
    assert p1.n_lanczos >= 0
    assert p1.n_times >= 2
    assert math.isfinite(p1.peak_complexity)
    assert p1.estimate.estimator_id == "krylov_peak"
    assert p1.estimate.support_status == "supported"
    assert p1.invent_green_live_qpu is False
    assert p1.estimate.invent_green_live_qpu is False
    assert len(p1.digest) == 64
    d = p1.to_dict()
    assert d["peak_complexity"] == p1.peak_complexity


def test_materialise_krylov_single_qubit_trivial_basis() -> None:
    """n_qubits=1 is inside product cap; ambient may return n_lanczos=0 (commuting).

    Must not crash MaterialisedKrylovProbe — zero Lanczos steps with finite peak
    is a supported trivial diagnostic, not a construction error.
    """
    p = materialise_krylov_probe(n_qubits=1, coupling=0.0, t_max=1.0, n_times=6, max_lanczos=8)
    assert p.estimate.n_qubits == 1
    assert p.n_lanczos >= 0
    assert math.isfinite(p.peak_complexity)
    assert p.estimate.support_status == "supported"
    assert p.invent_green_live_qpu is False


def test_materialise_krylov_cap_and_validation() -> None:
    """Krylov probe fails closed on qubit cap and invalid params."""
    with pytest.raises(ValueError, match="exceeds product cap"):
        materialise_krylov_probe(n_qubits=MAX_WITNESS_QUBITS + 1)
    with pytest.raises(ValueError, match="n_times"):
        materialise_krylov_probe(n_times=1)
    with pytest.raises(ValueError, match="t_max"):
        materialise_krylov_probe(t_max=0.0)


def test_materialise_otoc_probe_real_ambient() -> None:
    """OTOC probe exercises ambient compute_otoc without invent-green advantage."""
    p1 = materialise_demo_otoc_probe()
    p2 = materialise_demo_otoc_probe()
    assert isinstance(p1, MaterialisedOtocProbe)
    assert p1.digest == p2.digest
    assert math.isfinite(p1.final_otoc)
    assert p1.n_times >= 2
    assert p1.invent_green_otoc_advantage is False
    assert p1.invent_green_live_qpu is False
    assert p1.estimate.support_status == "supported"
    assert 0.0 <= abs(p1.final_otoc) <= 1.5  # numerical band for |F|
    d = p1.to_dict()
    assert "lyapunov_estimate" in d


def test_materialise_otoc_cap() -> None:
    """OTOC probe fails closed beyond qubit cap."""
    with pytest.raises(ValueError, match="exceeds product cap"):
        materialise_otoc_probe(n_qubits=MAX_WITNESS_QUBITS + 2)
    with pytest.raises(ValueError, match="n_times"):
        materialise_otoc_probe(n_times=1)


def test_materialise_shadow_probe_real_ambient() -> None:
    """Shadow probe exercises ambient classical_shadow_estimation."""
    p1 = materialise_demo_shadow_probe()
    p2 = materialise_demo_shadow_probe()
    assert isinstance(p1, MaterialisedShadowProbe)
    assert p1.digest == p2.digest
    assert p1.n_shots == 80
    assert p1.shadow_norm_bound >= 0.0
    assert p1.estimate.support_status == "supported"
    assert p1.invent_green_live_qpu is False
    assert "zz" in p1.observables or len(p1.observables) >= 1
    d = p1.to_dict()
    assert d["n_shots"] == 80


def test_shadow_under_sampled_status() -> None:
    """Low-shot shadow probes must report under_sampled (not silent green)."""
    p = materialise_shadow_probe(n_qubits=2, n_shots=max(1, MIN_SHADOW_SHOTS - 1), seed=3)
    assert p.estimate.support_status == "under_sampled"
    assert p.estimate.uncertainty >= 0.0


def test_shadow_caps_and_validation() -> None:
    """Shadow probe enforces shot and qubit caps."""
    with pytest.raises(ValueError, match="exceeds product cap"):
        materialise_shadow_probe(n_qubits=MAX_WITNESS_QUBITS + 1)
    with pytest.raises(ValueError, match="MAX_DEMO_SHADOW_SHOTS"):
        materialise_shadow_probe(n_shots=MAX_DEMO_SHADOW_SHOTS + 1)
    with pytest.raises(ValueError, match="n_shots"):
        materialise_shadow_probe(n_shots=0)
    with pytest.raises(ValueError, match="label length"):
        materialise_shadow_probe(n_qubits=2, observables={"bad": "Z"})


def test_bl18_order_parameter_compose() -> None:
    """BL-18 compose returns a WitnessEstimate with high R for sync phases."""
    est = materialise_bl18_order_parameter_compose()
    assert isinstance(est, WitnessEstimate)
    assert est.mean > 0.9
    assert est.support_status == "supported"
    assert est.invent_green_live_qpu is False
    assert "bl18" in est.estimator_id
    d = est.to_dict()
    assert d["mean"] == est.mean
    with pytest.raises(ValueError, match="harmonic"):
        materialise_bl18_order_parameter_compose(harmonic=0)
    with pytest.raises(ValueError, match="phases"):
        materialise_bl18_order_parameter_compose([])


def test_witness_estimate_validation() -> None:
    """WitnessEstimate rejects invent-green and non-finite values."""
    good = WitnessEstimate(
        estimator_id="x",
        mean=0.5,
        uncertainty=0.1,
        support_status="supported",
        backend="local",
        n_qubits=2,
        n_shots_or_times=10,
    )
    assert good.to_dict()["estimator_id"] == "x"
    with pytest.raises(ValueError, match="invent_green"):
        WitnessEstimate(
            estimator_id="x",
            mean=0.0,
            uncertainty=0.0,
            support_status="supported",
            backend="local",
            n_qubits=1,
            n_shots_or_times=1,
            invent_green_live_qpu=True,
        )
    with pytest.raises(ValueError, match="mean"):
        WitnessEstimate(
            estimator_id="x",
            mean=float("nan"),
            uncertainty=0.0,
            support_status="supported",
            backend="local",
            n_qubits=1,
            n_shots_or_times=1,
        )
    with pytest.raises(ValueError, match="uncertainty"):
        WitnessEstimate(
            estimator_id="x",
            mean=0.0,
            uncertainty=-1.0,
            support_status="supported",
            backend="local",
            n_qubits=1,
            n_shots_or_times=1,
        )


def test_capability_row_validation() -> None:
    """Capability rows reject hardware submit and blank ids."""
    with pytest.raises(ValueError, match="hardware_submit"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="s",
            hardware_submit_allowed=True,
        )
    with pytest.raises(ValueError, match="capability_id"):
        WitnessCapabilityRow(
            capability_id="  ",
            kind="krylov_complexity",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="s",
        )
    with pytest.raises(ValueError, match="unknown capability kind"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="not_a_kind",  # type: ignore[arg-type]
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="s",
        )


def test_boundary_row_validation() -> None:
    """Boundary rows require fail_closed True."""
    with pytest.raises(ValueError, match="fail_closed"):
        WitnessBoundaryRow(
            boundary_id="x",
            kind="otoc_advantage_claim",
            title="t",
            summary="s",
            fail_closed=False,
        )


def test_path_decision_validation() -> None:
    """PathEligibilityDecision validates outcome and empty fields."""
    d = PathEligibilityDecision(path_id="krylov", outcome="allowed", reason="ok")
    assert d.allowed is True
    with pytest.raises(ValueError, match="outcome"):
        PathEligibilityDecision(
            path_id="x",
            outcome="maybe",  # type: ignore[arg-type]
            reason="r",
        )
    with pytest.raises(ValueError, match="path_id"):
        PathEligibilityDecision(path_id="", outcome="allowed", reason="r")


def test_integrity_rejects_tampered_registry() -> None:
    """Integrity fails on policy drift, missing caps, and blank capabilities."""
    base = build_advanced_witnesses_product_registry()

    bad_schema = dict(base)
    bad_schema["schema"] = "wrong.v0"
    with pytest.raises(ValueError, match="schema"):
        assert_advanced_witnesses_product_integrity(bad_schema)

    bad_policy = dict(base)
    bad_policy["otoc_advantage_claim_policy"] = True
    with pytest.raises(ValueError, match="otoc_advantage"):
        assert_advanced_witnesses_product_integrity(bad_policy)

    bad_hw = dict(base)
    caps = [dict(c) for c in base["capabilities"]]  # type: ignore[index]
    caps[0]["hardware_submit_allowed"] = True
    bad_hw["capabilities"] = caps
    with pytest.raises(ValueError, match="hardware_submit"):
        assert_advanced_witnesses_product_integrity(bad_hw)

    bad_blank = dict(base)
    bad_blank["blank_entry_count"] = 3
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_advanced_witnesses_product_integrity(bad_blank)

    bad_cap = dict(base)
    bad_cap["max_witness_qubits"] = 99
    with pytest.raises(ValueError, match="max_witness_qubits"):
        assert_advanced_witnesses_product_integrity(bad_cap)

    missing = dict(base)
    missing["capabilities"] = [
        c
        for c in base["capabilities"]  # type: ignore[union-attr]
        if c["capability_id"] != "krylov_complexity"
    ]
    missing["capability_count"] = len(missing["capabilities"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing|drift"):
        assert_advanced_witnesses_product_integrity(missing)


def test_claim_boundary_string_present() -> None:
    """Shared claim boundary is non-promotional and mentions refuse paths."""
    assert "refuse" in ADVANCED_WITNESSES_CLAIM_BOUNDARY.lower()
    assert (
        "OTOC" in ADVANCED_WITNESSES_CLAIM_BOUNDARY
        or "otoc" in ADVANCED_WITNESSES_CLAIM_BOUNDARY.lower()
    )
    assert "Krylov" in WITNESS_GLOSSARY


def test_materialised_probe_invent_green_guards() -> None:
    """Materialised dataclasses reject invent-green construction."""
    est = WitnessEstimate(
        estimator_id="e",
        mean=0.0,
        uncertainty=0.0,
        support_status="supported",
        backend="b",
        n_qubits=1,
        n_shots_or_times=1,
    )
    digest = "a" * 64
    with pytest.raises(ValueError, match="invent_green"):
        MaterialisedKrylovProbe(
            estimate=est,
            peak_complexity=1.0,
            n_lanczos=2,
            n_times=4,
            digest=digest,
            invent_green_live_qpu=True,
        )
    with pytest.raises(ValueError, match="invent_green"):
        MaterialisedOtocProbe(
            estimate=est,
            final_otoc=0.5,
            lyapunov_estimate=None,
            scrambling_time=None,
            n_times=4,
            digest=digest,
            invent_green_otoc_advantage=True,
        )
    with pytest.raises(ValueError, match="digest"):
        MaterialisedShadowProbe(
            estimate=est,
            observables={"z": 0.0},
            shadow_norm_bound=0.1,
            n_shots=10,
            digest="short",
        )


def test_shadow_one_qubit_default_and_empty_observables() -> None:
    """One-qubit default observable path and empty custom map fail closed."""
    p = materialise_shadow_probe(n_qubits=1, n_shots=40, seed=11)
    assert p.estimate.n_qubits == 1
    assert p.observables
    with pytest.raises(ValueError, match="non-empty"):
        materialise_shadow_probe(n_qubits=2, observables={})


def test_shadow_subprocess_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shadow ambient helper fails closed on subprocess and JSON errors."""
    import scpn_quantum_control.advanced_witnesses_product as mod

    def boom(*_a: object, **_k: object) -> object:
        raise mod.subprocess.CalledProcessError(1, "x", stderr="shadow boom")

    monkeypatch.setattr(mod.subprocess, "run", boom)
    with pytest.raises(ValueError, match="shadow subprocess failed"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)

    class FakeCompleted:
        stdout = "not-json\n"

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_a, **_k: FakeCompleted(),
    )
    with pytest.raises(ValueError, match="non-JSON"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)

    class TimeoutExc(mod.subprocess.TimeoutExpired):
        pass

    def timeout(*_a: object, **_k: object) -> object:
        raise mod.subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(mod.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="timed out"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)


def test_more_dataclass_and_integrity_edges() -> None:
    """Cover remaining validation branches for rows, estimates, integrity."""
    with pytest.raises(ValueError, match="title"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title=" ",
            summary="s",
            ambient_module="m",
            ambient_symbol="s",
        )
    with pytest.raises(ValueError, match="summary"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title="t",
            summary="",
            ambient_module="m",
            ambient_symbol="s",
        )
    with pytest.raises(ValueError, match="ambient_module"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title="t",
            summary="s",
            ambient_module="",
            ambient_symbol="s",
        )
    with pytest.raises(ValueError, match="ambient_symbol"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol=" ",
        )
    with pytest.raises(ValueError, match="support_posture"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="s",
            support_posture="nope",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="as_of"):
        WitnessCapabilityRow(
            capability_id="x",
            kind="krylov_complexity",
            title="t",
            summary="s",
            ambient_module="m",
            ambient_symbol="s",
            as_of="",
        )
    with pytest.raises(ValueError, match="boundary_id"):
        WitnessBoundaryRow(
            boundary_id="",
            kind="otoc_advantage_claim",
            title="t",
            summary="s",
        )
    with pytest.raises(ValueError, match="unknown boundary kind"):
        WitnessBoundaryRow(
            boundary_id="x",
            kind="nope",  # type: ignore[arg-type]
            title="t",
            summary="s",
        )
    with pytest.raises(ValueError, match="support_status"):
        WitnessEstimate(
            estimator_id="x",
            mean=0.0,
            uncertainty=0.0,
            support_status="weird",  # type: ignore[arg-type]
            backend="b",
            n_qubits=1,
            n_shots_or_times=1,
        )
    with pytest.raises(ValueError, match="n_qubits"):
        WitnessEstimate(
            estimator_id="x",
            mean=0.0,
            uncertainty=0.0,
            support_status="supported",
            backend="b",
            n_qubits=0,
            n_shots_or_times=1,
        )
    with pytest.raises(ValueError, match="n_shots_or_times"):
        WitnessEstimate(
            estimator_id="x",
            mean=0.0,
            uncertainty=0.0,
            support_status="supported",
            backend="b",
            n_qubits=1,
            n_shots_or_times=-1,
        )
    with pytest.raises(ValueError, match="estimator_id"):
        WitnessEstimate(
            estimator_id="",
            mean=0.0,
            uncertainty=0.0,
            support_status="supported",
            backend="b",
            n_qubits=1,
            n_shots_or_times=1,
        )
    with pytest.raises(ValueError, match="backend"):
        WitnessEstimate(
            estimator_id="x",
            mean=0.0,
            uncertainty=0.0,
            support_status="supported",
            backend="",
            n_qubits=1,
            n_shots_or_times=1,
        )

    base = build_advanced_witnesses_product_registry()
    bad_claim = dict(base)
    bad_claim["claim_boundary"] = "not the product boundary"
    with pytest.raises(ValueError, match="claim_boundary"):
        assert_advanced_witnesses_product_integrity(bad_claim)

    bad_gloss = dict(base)
    bad_gloss["glossary"] = {"only": "one"}
    with pytest.raises(ValueError, match="glossary"):
        assert_advanced_witnesses_product_integrity(bad_gloss)

    bad_inv = dict(base)
    bad_inv["ambient_inventory"] = []
    with pytest.raises(ValueError, match="ambient_inventory"):
        assert_advanced_witnesses_product_integrity(bad_inv)

    bad_count = dict(base)
    bad_count["capability_count"] = 0
    with pytest.raises(ValueError, match="capability_count"):
        assert_advanced_witnesses_product_integrity(bad_count)

    bad_bcount = dict(base)
    bad_bcount["boundary_count"] = 0
    with pytest.raises(ValueError, match="boundary_count"):
        assert_advanced_witnesses_product_integrity(bad_bcount)

    bad_shots = dict(base)
    bad_shots["min_shadow_shots"] = 999
    with pytest.raises(ValueError, match="min_shadow_shots"):
        assert_advanced_witnesses_product_integrity(bad_shots)

    bad_max_shots = dict(base)
    bad_max_shots["max_demo_shadow_shots"] = 1
    with pytest.raises(ValueError, match="max_demo_shadow_shots"):
        assert_advanced_witnesses_product_integrity(bad_max_shots)

    bad_caps_empty = dict(base)
    bad_caps_empty["capabilities"] = []
    with pytest.raises(ValueError, match="capabilities"):
        assert_advanced_witnesses_product_integrity(bad_caps_empty)

    bad_bounds_empty = dict(base)
    bad_bounds_empty["boundaries"] = []
    with pytest.raises(ValueError, match="boundaries"):
        assert_advanced_witnesses_product_integrity(bad_bounds_empty)

    # Duplicate capability id
    dup = dict(base)
    caps = [dict(c) for c in base["capabilities"]]  # type: ignore[index]
    caps.append(dict(caps[0]))
    dup["capabilities"] = caps
    dup["capability_count"] = len(caps)
    with pytest.raises(ValueError, match="duplicate capability"):
        assert_advanced_witnesses_product_integrity(dup)

    # Boundary fail_closed not True
    bf = dict(base)
    bounds = [dict(b) for b in base["boundaries"]]  # type: ignore[index]
    bounds[0]["fail_closed"] = False
    bf["boundaries"] = bounds
    with pytest.raises(ValueError, match="fail_closed"):
        assert_advanced_witnesses_product_integrity(bf)

    # Krylov max_lanczos and OTOC t_max edges
    with pytest.raises(ValueError, match="max_lanczos"):
        materialise_krylov_probe(max_lanczos=1)
    with pytest.raises(ValueError, match="t_max"):
        materialise_otoc_probe(t_max=-1.0)

    # Materialised Krylov n_lanczos / n_times validation
    est = WitnessEstimate(
        estimator_id="e",
        mean=1.0,
        uncertainty=0.0,
        support_status="supported",
        backend="b",
        n_qubits=2,
        n_shots_or_times=4,
    )
    digest = "b" * 64
    with pytest.raises(ValueError, match="n_lanczos"):
        MaterialisedKrylovProbe(
            estimate=est,
            peak_complexity=1.0,
            n_lanczos=-1,
            n_times=4,
            digest=digest,
        )
    # n_lanczos==0 is allowed (trivial ambient basis)
    ok_zero = MaterialisedKrylovProbe(
        estimate=est,
        peak_complexity=0.0,
        n_lanczos=0,
        n_times=4,
        digest=digest,
    )
    assert ok_zero.n_lanczos == 0
    with pytest.raises(ValueError, match="n_times"):
        MaterialisedOtocProbe(
            estimate=est,
            final_otoc=0.1,
            lyapunov_estimate=None,
            scrambling_time=None,
            n_times=0,
            digest=digest,
        )
    with pytest.raises(ValueError, match="shadow_norm_bound"):
        MaterialisedShadowProbe(
            estimate=est,
            observables={"z": 0.0},
            shadow_norm_bound=-0.1,
            n_shots=10,
            digest=digest,
        )
    with pytest.raises(ValueError, match="observables"):
        MaterialisedShadowProbe(
            estimate=est,
            observables={},
            shadow_norm_bound=0.1,
            n_shots=10,
            digest=digest,
        )
    with pytest.raises(ValueError, match="n_shots"):
        MaterialisedShadowProbe(
            estimate=est,
            observables={"z": 0.0},
            shadow_norm_bound=0.1,
            n_shots=0,
            digest=digest,
        )


def test_remaining_validation_and_integrity_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hit residual dataclass / integrity / payload edge branches for coverage."""
    with pytest.raises(ValueError, match="title"):
        WitnessBoundaryRow(
            boundary_id="x",
            kind="otoc_advantage_claim",
            title=" ",
            summary="s",
        )
    with pytest.raises(ValueError, match="summary"):
        WitnessBoundaryRow(
            boundary_id="x",
            kind="otoc_advantage_claim",
            title="t",
            summary="",
        )
    with pytest.raises(ValueError, match="reason"):
        PathEligibilityDecision(path_id="x", outcome="allowed", reason="  ")

    est = WitnessEstimate(
        estimator_id="e",
        mean=0.0,
        uncertainty=0.0,
        support_status="supported",
        backend="b",
        n_qubits=1,
        n_shots_or_times=1,
    )
    digest = "c" * 64
    with pytest.raises(ValueError, match="peak_complexity"):
        MaterialisedKrylovProbe(
            estimate=est,
            peak_complexity=float("nan"),
            n_lanczos=2,
            n_times=4,
            digest=digest,
        )
    with pytest.raises(ValueError, match="digest"):
        MaterialisedKrylovProbe(
            estimate=est,
            peak_complexity=1.0,
            n_lanczos=2,
            n_times=4,
            digest="nope",
        )
    with pytest.raises(ValueError, match="n_times"):
        MaterialisedKrylovProbe(
            estimate=est,
            peak_complexity=1.0,
            n_lanczos=2,
            n_times=0,
            digest=digest,
        )
    with pytest.raises(ValueError, match="final_otoc"):
        MaterialisedOtocProbe(
            estimate=est,
            final_otoc=float("inf"),
            lyapunov_estimate=None,
            scrambling_time=None,
            n_times=2,
            digest=digest,
        )
    with pytest.raises(ValueError, match="lyapunov"):
        MaterialisedOtocProbe(
            estimate=est,
            final_otoc=0.1,
            lyapunov_estimate=float("nan"),
            scrambling_time=None,
            n_times=2,
            digest=digest,
        )
    with pytest.raises(ValueError, match="scrambling"):
        MaterialisedOtocProbe(
            estimate=est,
            final_otoc=0.1,
            lyapunov_estimate=None,
            scrambling_time=float("nan"),
            n_times=2,
            digest=digest,
        )
    with pytest.raises(ValueError, match="digest"):
        MaterialisedOtocProbe(
            estimate=est,
            final_otoc=0.1,
            lyapunov_estimate=None,
            scrambling_time=None,
            n_times=2,
            digest="x",
        )
    with pytest.raises(ValueError, match="invent_green"):
        MaterialisedOtocProbe(
            estimate=est,
            final_otoc=0.1,
            lyapunov_estimate=None,
            scrambling_time=None,
            n_times=2,
            digest=digest,
            invent_green_live_qpu=True,
        )
    with pytest.raises(ValueError, match="observable"):
        MaterialisedShadowProbe(
            estimate=est,
            observables={"": 0.0},
            shadow_norm_bound=0.1,
            n_shots=10,
            digest=digest,
        )
    with pytest.raises(ValueError, match="finite"):
        MaterialisedShadowProbe(
            estimate=est,
            observables={"z": float("nan")},
            shadow_norm_bound=0.1,
            n_shots=10,
            digest=digest,
        )
    with pytest.raises(ValueError, match="invent_green"):
        MaterialisedShadowProbe(
            estimate=est,
            observables={"z": 0.0},
            shadow_norm_bound=0.1,
            n_shots=10,
            digest=digest,
            invent_green_live_qpu=True,
        )

    import scpn_quantum_control.advanced_witnesses_product as mod

    with pytest.raises(ValueError, match="n_qubits must be >= 1"):
        mod._require_qubit_cap(0, label="t")

    # Integrity: non-mapping capability / boundary rows
    base = build_advanced_witnesses_product_registry()
    bad_row = dict(base)
    bad_row["capabilities"] = ["not-a-mapping"]
    bad_row["capability_count"] = 1
    with pytest.raises(ValueError, match="mapping"):
        assert_advanced_witnesses_product_integrity(bad_row)

    bad_brow = dict(base)
    bad_brow["boundaries"] = [42]
    bad_brow["boundary_count"] = 1
    with pytest.raises(ValueError, match="mapping"):
        assert_advanced_witnesses_product_integrity(bad_brow)

    # Blank capability_id counted then fails blank_entry or required missing
    blank_cap = dict(base)
    caps = [dict(c) for c in base["capabilities"]]  # type: ignore[index]
    caps[0]["capability_id"] = "  "
    blank_cap["capabilities"] = caps
    with pytest.raises(ValueError, match="blank|missing|drift"):
        assert_advanced_witnesses_product_integrity(blank_cap)

    # Empty ambient_symbol
    empty_sym = dict(base)
    caps2 = [dict(c) for c in base["capabilities"]]  # type: ignore[index]
    caps2[0]["ambient_symbol"] = ""
    empty_sym["capabilities"] = caps2
    with pytest.raises(ValueError, match="ambient_symbol"):
        assert_advanced_witnesses_product_integrity(empty_sym)

    # Boundary blank id / duplicate / drift
    blank_b = dict(base)
    bounds = [dict(b) for b in base["boundaries"]]  # type: ignore[index]
    bounds[0]["boundary_id"] = ""
    blank_b["boundaries"] = bounds
    with pytest.raises(ValueError, match="boundary_id"):
        assert_advanced_witnesses_product_integrity(blank_b)

    dup_b = dict(base)
    bounds2 = [dict(b) for b in base["boundaries"]]  # type: ignore[index]
    bounds2.append(dict(bounds2[0]))
    dup_b["boundaries"] = bounds2
    dup_b["boundary_count"] = len(bounds2)
    with pytest.raises(ValueError, match="duplicate boundary"):
        assert_advanced_witnesses_product_integrity(dup_b)

    # Shadow payload validation via monkeypatch of helper
    def bad_payload(**_k: object) -> dict[str, object]:
        return {"estimated_observables": {}}

    monkeypatch.setattr(mod, "_run_ambient_shadow_json", bad_payload)
    with pytest.raises(ValueError, match="estimated_observables"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)

    def bad_bound(**_k: object) -> dict[str, object]:
        return {
            "estimated_observables": {"z": 0.0},
            "shadow_norm_bound": "nope",
            "n_qubits": 2,
            "n_shots": 20,
        }

    monkeypatch.setattr(mod, "_run_ambient_shadow_json", bad_bound)
    with pytest.raises(ValueError, match="shadow_norm_bound"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)

    def non_dict_payload(*_a: object, **_k: object) -> object:
        class Fake:
            stdout = "[]\n"

        return Fake()

    monkeypatch.setattr(mod.subprocess, "run", non_dict_payload)
    with pytest.raises(ValueError, match="object|payload"):
        materialise_shadow_probe(n_qubits=2, n_shots=20, seed=1)

    # Custom phases compose
    est2 = materialise_bl18_order_parameter_compose([0.0, 0.0, 0.0], harmonic=1)
    assert est2.mean == pytest.approx(1.0, abs=1e-9)
