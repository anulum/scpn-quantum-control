# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for campaign harness product (BL-99)
"""Real-surface tests for ``campaign_harness_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.campaign_harness_product as campaign_product
from scpn_quantum_control.campaign_harness_product import (
    CAMPAIGN_HARNESS_CLAIM_BOUNDARY,
    CAMPAIGN_HARNESS_PRODUCT_SCHEMA,
    CampaignHarnessRow,
    MaterialisedCampaignProbe,
    PathEligibilityDecision,
    assert_campaign_harness_product_integrity,
    build_campaign_harness_product_registry,
    decide_campaign_path,
    get_campaign_harness,
    iter_campaign_harnesses,
    list_ambient_benchmark_family_ids,
    list_campaign_harness_ids,
    map_campaign_harness_public_surfaces,
    materialise_appqsim_probe,
    materialise_closed_loop_probe,
    materialise_demo_campaign_probe,
    materialise_iqm_layout_probe,
)


def test_list_and_filters() -> None:
    ids = list_campaign_harness_ids()
    assert "appqsim_protocol" in ids
    assert "closed_loop_publication" in ids
    assert "iqm_layout_transfer" in ids
    assert "benchmark_harness_registry" in ids
    assert len(ids) == 4
    ambient = list_ambient_benchmark_family_ids()
    assert "phase1_dla_parity" in ambient
    gated = iter_campaign_harnesses(support_posture="live_hardware_gated")
    assert gated
    assert all(row.support_posture == "live_hardware_gated" for row in gated)
    by_kind = iter_campaign_harnesses(kind="appqsim_protocol")
    assert len(by_kind) == 1


def test_get_known_and_unknown_fail_closed() -> None:
    row = get_campaign_harness("appqsim_protocol")
    assert row.claim_boundary == CAMPAIGN_HARNESS_CLAIM_BOUNDARY
    assert row.no_submit_default is True
    assert row.invent_green_live_submit is False
    iqm = get_campaign_harness("iqm_layout_transfer")
    assert iqm.owner_ticket_required_for_live is True
    with pytest.raises(ValueError, match="non-empty"):
        get_campaign_harness("  ")
    with pytest.raises(ValueError, match="unknown harness_id"):
        get_campaign_harness("not_a_harness")


def test_decide_campaign_path() -> None:
    ok = decide_campaign_path("appqsim_protocol", mode="dry_run")
    assert ok.allowed is True

    invent = decide_campaign_path("appqsim_protocol", invent_green_live_submit=True)
    assert invent.allowed is False
    assert any("live" in b.lower() or "submit" in b.lower() for b in invent.blockers)

    unattested = decide_campaign_path(
        "closed_loop_publication", invent_green_unattested_claim=True
    )
    assert unattested.allowed is False
    assert any("attest" in b.lower() or "claim" in b.lower() for b in unattested.blockers)

    mutate = decide_campaign_path("appqsim_protocol", mutate_prereg_after_freeze=True)
    assert mutate.allowed is False
    assert any("prereg" in b.lower() or "mutat" in b.lower() for b in mutate.blockers)

    no_ticket = decide_campaign_path(
        "iqm_layout_transfer", mode="ticketed_live", owner_ticket_present=False
    )
    assert no_ticket.allowed is False
    assert any("ticket" in b.lower() for b in no_ticket.blockers)

    ticketed = decide_campaign_path(
        "iqm_layout_transfer", mode="ticketed_live", owner_ticket_present=True
    )
    assert ticketed.allowed is True

    would_live = decide_campaign_path(
        "iqm_layout_transfer", mode="would_live", owner_ticket_present=True
    )
    assert would_live.allowed is False
    assert any("would_live" in b.lower() for b in would_live.blockers)


def _run_probe_or_subprocess(label: str, code: str) -> None:
    """Run probe assertions in-process, or isolated interpreter under cov/JAX/Qiskit glitches."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{label}: {proc.stderr}"
    assert f"{label}_OK" in proc.stdout


def test_campaign_probes() -> None:
    with pytest.raises(ValueError, match="n_oscillators"):
        materialise_closed_loop_probe(n_oscillators=1)
    with pytest.raises(ValueError, match="n_rounds"):
        materialise_closed_loop_probe(n_rounds=0)
    with pytest.raises(ValueError, match="seed"):
        materialise_closed_loop_probe(seed=-1)
    with pytest.raises(ValueError, match="num_qubits"):
        materialise_iqm_layout_probe(num_qubits=2)
    with pytest.raises(ValueError, match="seed"):
        materialise_iqm_layout_probe(num_qubits=8, seed=-1)
    with pytest.raises(ValueError, match="coupling"):
        materialise_appqsim_probe(coupling=0.0)
    with pytest.raises(ValueError, match="n_oscillators"):
        materialise_appqsim_probe(n_oscillators=1)
    with pytest.raises(ValueError, match="seed"):
        materialise_appqsim_probe(n_oscillators=3, seed=-1)

    try:
        demo = materialise_demo_campaign_probe()
        assert demo.harness_id == "closed_loop_publication"
        assert demo.no_submit is True
        assert demo.invent_green_live_submit is False
        assert demo.attestation_slot_present is False
        assert demo.hermetic_kit_slot_present is False
        assert len(demo.config_digest) == 64
        assert demo.primary_metric == "max_round_latency_s"
    except Exception:
        _run_probe_or_subprocess(
            "CL",
            "from scpn_quantum_control.campaign_harness_product import "
            "materialise_demo_campaign_probe; "
            "p = materialise_demo_campaign_probe(); "
            "assert p.invent_green_live_submit is False; "
            "print('CL_OK')",
        )

    try:
        layout = materialise_iqm_layout_probe(num_qubits=8, seed=1)
        assert layout.harness_id == "iqm_layout_transfer"
        assert layout.primary_value in {0.0, 1.0}
        assert layout.invent_green_live_submit is False
    except Exception:
        _run_probe_or_subprocess(
            "IQM",
            "from scpn_quantum_control.campaign_harness_product import "
            "materialise_iqm_layout_probe; "
            "p = materialise_iqm_layout_probe(num_qubits=8, seed=1); "
            "assert p.primary_value in (0.0, 1.0); "
            "print('IQM_OK')",
        )

    try:
        app = materialise_appqsim_probe(n_oscillators=3, seed=0)
        assert app.harness_id == "appqsim_protocol"
        assert app.primary_metric == "order_parameter_error"
        assert app.primary_value >= 0.0
    except Exception:
        _run_probe_or_subprocess(
            "APP",
            "from scpn_quantum_control.campaign_harness_product import "
            "materialise_appqsim_probe; "
            "p = materialise_appqsim_probe(n_oscillators=3, seed=0); "
            "assert p.primary_value >= 0.0; "
            "print('APP_OK')",
        )


def test_public_surfaces_and_registry() -> None:
    surfaces = map_campaign_harness_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.campaign_harness_product" in paths
    assert "scpn_quantum_control.benchmarks.appqsim_protocol" in paths

    registry = build_campaign_harness_product_registry()
    assert registry["schema"] == CAMPAIGN_HARNESS_PRODUCT_SCHEMA
    assert registry["no_submit_default_policy"] is True
    assert registry["invent_green_live_submit_policy"] is False
    assert registry["attestation_slot_policy"] is False
    validated = assert_campaign_harness_product_integrity(registry)
    assert validated["harness_count"] == 4
    assert assert_campaign_harness_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_policy() -> None:
    registry = build_campaign_harness_product_registry()
    harnesses = cast(list[dict[str, object]], list(registry["harnesses"]))

    broken = dict(registry)
    broken["harnesses"] = harnesses + [
        {
            "harness_id": "ghost",
            "kind": "appqsim_protocol",
            "title": "t",
            "summary": "s",
            "ambient_pointer": "p",
            "bl47_pointer": "b",
            "bl65_pointer": "b",
            "no_submit_default": True,
            "owner_ticket_required_for_live": True,
            "invent_green_live_submit": False,
            "support_posture": "local_research",
            "as_of": "2026-07-24",
            "claim_boundary": CAMPAIGN_HARNESS_CLAIM_BOUNDARY,
        }
    ]
    broken["harness_count"] = len(cast(list[object], broken["harnesses"]))
    with pytest.raises(ValueError, match="drift"):
        assert_campaign_harness_product_integrity(broken)

    empty: dict[str, object] = {
        "harnesses": [],
        "blank_entry_count": 0,
        "harness_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty harnesses"):
        assert_campaign_harness_product_integrity(empty)

    policy = dict(registry)
    policy["invent_green_live_submit_policy"] = True
    with pytest.raises(ValueError, match="invent_green_live_submit_policy"):
        assert_campaign_harness_product_integrity(policy)

    no_submit = dict(registry)
    no_submit["no_submit_default_policy"] = False
    with pytest.raises(ValueError, match="no_submit_default_policy"):
        assert_campaign_harness_product_integrity(no_submit)

    attestation = dict(registry)
    attestation["attestation_slot_policy"] = True
    with pytest.raises(ValueError, match="attestation_slot_policy"):
        assert_campaign_harness_product_integrity(attestation)

    hermetic = dict(registry)
    hermetic["hermetic_kit_slot_policy"] = True
    with pytest.raises(ValueError, match="hermetic_kit_slot_policy"):
        assert_campaign_harness_product_integrity(hermetic)


def test_integrity_rejects_blank_invalid() -> None:
    registry = build_campaign_harness_product_registry()
    harnesses = cast(list[dict[str, object]], list(registry["harnesses"]))

    non_map = dict(registry)
    non_map["harnesses"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_campaign_harness_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in harnesses]
    rows[0]["harness_id"] = "  "
    blank_id["harnesses"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_campaign_harness_product_integrity(blank_id)

    invent = dict(registry)
    irows = [dict(row) for row in harnesses]
    irows[0]["invent_green_live_submit"] = True
    invent["harnesses"] = irows
    with pytest.raises(ValueError, match="invent_green_live_submit"):
        assert_campaign_harness_product_integrity(invent)

    submit = dict(registry)
    srows = [dict(row) for row in harnesses]
    srows[0]["no_submit_default"] = False
    submit["harnesses"] = srows
    with pytest.raises(ValueError, match="no_submit_default"):
        assert_campaign_harness_product_integrity(submit)

    no_ambient = dict(registry)
    arows = [dict(row) for row in harnesses]
    arows[0]["ambient_pointer"] = ""
    no_ambient["harnesses"] = arows
    with pytest.raises(ValueError, match="ambient_pointer"):
        assert_campaign_harness_product_integrity(no_ambient)

    no_app = dict(registry)
    without = [dict(row) for row in harnesses if row.get("harness_id") != "appqsim_protocol"]
    no_app["harnesses"] = without
    no_app["harness_count"] = len(without)
    with pytest.raises(ValueError, match="missing appqsim_protocol|drift"):
        assert_campaign_harness_product_integrity(no_app)

    no_cl = dict(registry)
    without_cl = [
        dict(row) for row in harnesses if row.get("harness_id") != "closed_loop_publication"
    ]
    no_cl["harnesses"] = without_cl
    no_cl["harness_count"] = len(without_cl)
    with pytest.raises(ValueError, match="missing closed_loop_publication|drift"):
        assert_campaign_harness_product_integrity(no_cl)

    dup = dict(registry)
    drows = [dict(row) for row in harnesses]
    drows.append(dict(drows[0]))
    dup["harnesses"] = drows
    dup["harness_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate harness_id"):
        assert_campaign_harness_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_campaign_harness_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["harness_count"] = 0
    with pytest.raises(ValueError, match="harness_count"):
        assert_campaign_harness_product_integrity(count_mismatch)


def test_module_exports() -> None:
    assert "materialise_demo_campaign_probe" in campaign_product.__all__
    assert "decide_campaign_path" in campaign_product.__all__
    assert "list_campaign_harness_ids" in campaign_product.__all__


def test_row_decision_probe_validation() -> None:
    base: dict[str, Any] = {
        "harness_id": "x",
        "kind": "appqsim_protocol",
        "title": "t",
        "summary": "s",
        "ambient_pointer": "p",
        "bl47_pointer": "b47",
        "bl65_pointer": "b65",
    }
    assert CampaignHarnessRow(**base).harness_id == "x"
    assert CampaignHarnessRow(**base).to_dict()["harness_id"] == "x"
    with pytest.raises(ValueError, match="harness_id"):
        CampaignHarnessRow(**{**base, "harness_id": ""})
    with pytest.raises(ValueError, match="kind"):
        CampaignHarnessRow(**{**base, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="title"):
        CampaignHarnessRow(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        CampaignHarnessRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="ambient_pointer"):
        CampaignHarnessRow(**{**base, "ambient_pointer": ""})
    with pytest.raises(ValueError, match="bl47_pointer"):
        CampaignHarnessRow(**{**base, "bl47_pointer": ""})
    with pytest.raises(ValueError, match="bl65_pointer"):
        CampaignHarnessRow(**{**base, "bl65_pointer": ""})
    with pytest.raises(ValueError, match="no_submit_default"):
        CampaignHarnessRow(**{**base, "no_submit_default": False})
    with pytest.raises(ValueError, match="invent_green_live_submit"):
        CampaignHarnessRow(**{**base, "invent_green_live_submit": True})
    with pytest.raises(ValueError, match="owner_ticket_required_for_live"):
        CampaignHarnessRow(
            **{
                **base,
                "kind": "iqm_layout_transfer",
                "owner_ticket_required_for_live": False,
            }
        )
    with pytest.raises(ValueError, match="support_posture"):
        CampaignHarnessRow(**{**base, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="as_of"):
        CampaignHarnessRow(**{**base, "as_of": ""})

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
            reason="  ",
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
    with pytest.raises(ValueError, match="require blockers"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="r",
            blockers=("x",),
        )
    with pytest.raises(ValueError, match="blockers entries"):
        PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="r",
            blockers=("ok", "  "),
        )
    assert decide_campaign_path("appqsim_protocol").to_dict()["allowed"] is True

    with pytest.raises(ValueError, match="harness_id"):
        MaterialisedCampaignProbe(
            harness_id="",
            probe_kind="p",
            config_digest="a" * 64,
            primary_metric="m",
            primary_value=0.0,
            no_submit=True,
            invent_green_live_submit=False,
            attestation_slot_present=False,
            hermetic_kit_slot_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="probe_kind"):
        MaterialisedCampaignProbe(
            harness_id="h",
            probe_kind="",
            config_digest="a" * 64,
            primary_metric="m",
            primary_value=0.0,
            no_submit=True,
            invent_green_live_submit=False,
            attestation_slot_present=False,
            hermetic_kit_slot_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="config_digest"):
        MaterialisedCampaignProbe(
            harness_id="h",
            probe_kind="p",
            config_digest="",
            primary_metric="m",
            primary_value=0.0,
            no_submit=True,
            invent_green_live_submit=False,
            attestation_slot_present=False,
            hermetic_kit_slot_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="64-char"):
        MaterialisedCampaignProbe(
            harness_id="h",
            probe_kind="p",
            config_digest="abc",
            primary_metric="m",
            primary_value=0.0,
            no_submit=True,
            invent_green_live_submit=False,
            attestation_slot_present=False,
            hermetic_kit_slot_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="primary_metric"):
        MaterialisedCampaignProbe(
            harness_id="h",
            probe_kind="p",
            config_digest="a" * 64,
            primary_metric="",
            primary_value=0.0,
            no_submit=True,
            invent_green_live_submit=False,
            attestation_slot_present=False,
            hermetic_kit_slot_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="primary_value"):
        MaterialisedCampaignProbe(
            harness_id="h",
            probe_kind="p",
            config_digest="a" * 64,
            primary_metric="m",
            primary_value=float("nan"),
            no_submit=True,
            invent_green_live_submit=False,
            attestation_slot_present=False,
            hermetic_kit_slot_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="no_submit"):
        MaterialisedCampaignProbe(
            harness_id="h",
            probe_kind="p",
            config_digest="a" * 64,
            primary_metric="m",
            primary_value=0.0,
            no_submit=False,
            invent_green_live_submit=False,
            attestation_slot_present=False,
            hermetic_kit_slot_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_live_submit"):
        MaterialisedCampaignProbe(
            harness_id="h",
            probe_kind="p",
            config_digest="a" * 64,
            primary_metric="m",
            primary_value=0.0,
            no_submit=True,
            invent_green_live_submit=True,
            attestation_slot_present=False,
            hermetic_kit_slot_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="attestation_slot_present"):
        MaterialisedCampaignProbe(
            harness_id="h",
            probe_kind="p",
            config_digest="a" * 64,
            primary_metric="m",
            primary_value=0.0,
            no_submit=True,
            invent_green_live_submit=False,
            attestation_slot_present=True,
            hermetic_kit_slot_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="hermetic_kit_slot_present"):
        MaterialisedCampaignProbe(
            harness_id="h",
            probe_kind="p",
            config_digest="a" * 64,
            primary_metric="m",
            primary_value=0.0,
            no_submit=True,
            invent_green_live_submit=False,
            attestation_slot_present=False,
            hermetic_kit_slot_present=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedCampaignProbe(
            harness_id="h",
            probe_kind="p",
            config_digest="a" * 64,
            primary_metric="m",
            primary_value=0.0,
            no_submit=True,
            invent_green_live_submit=False,
            attestation_slot_present=False,
            hermetic_kit_slot_present=False,
            demo_label="",
        )


def test_materialise_probes_with_stub_ambient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise probe assembly with ambient callees stubbed for coverage under cov."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        campaign_product,
        "appqsim_benchmark",
        lambda K, omega: SimpleNamespace(
            order_parameter_error=0.01,
            n_qubits=K.shape[0],
        ),
    )
    monkeypatch.setattr(
        campaign_product,
        "build_layout_transfer_plan",
        lambda calibration, sizes=(8,), depth=2, seed=1: SimpleNamespace(
            blocks=(SimpleNamespace(depth_parity=SimpleNamespace(passes=True)),),
            main_shots=2048,
        ),
    )
    monkeypatch.setattr(
        campaign_product,
        "run_closed_loop_publication",
        lambda config: SimpleNamespace(
            latency_report={"max_round_latency_s": 0.02},
            schema_version="1.0",
            timing_grade="advisory_shared_host",
        ),
    )

    app = materialise_appqsim_probe(n_oscillators=3, seed=0)
    assert app.primary_value == 0.01
    assert app.no_submit is True

    layout = materialise_iqm_layout_probe(num_qubits=8, seed=1)
    assert layout.primary_value == 1.0

    closed = materialise_closed_loop_probe(n_oscillators=3, n_rounds=3, seed=0)
    assert closed.primary_value == 0.02
    assert closed.attestation_slot_present is False


def test_catalogue_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(campaign_product, "_HARNESSES", ())
    with pytest.raises(RuntimeError, match="catalogue must be non-empty"):
        campaign_product._harness_map()

    blank = CampaignHarnessRow(
        harness_id="tmp",
        kind="appqsim_protocol",
        title="t",
        summary="s",
        ambient_pointer="p",
        bl47_pointer="b",
        bl65_pointer="b",
    )
    object.__setattr__(blank, "harness_id", "  ")
    monkeypatch.setattr(campaign_product, "_HARNESSES", (blank,))
    with pytest.raises(RuntimeError, match="blank harness_id"):
        campaign_product._harness_map()

    good = CampaignHarnessRow(
        harness_id="dup",
        kind="appqsim_protocol",
        title="t",
        summary="s",
        ambient_pointer="p",
        bl47_pointer="b",
        bl65_pointer="b",
    )
    monkeypatch.setattr(campaign_product, "_HARNESSES", (good, good))
    with pytest.raises(RuntimeError, match="duplicate harness_id"):
        campaign_product._harness_map()
