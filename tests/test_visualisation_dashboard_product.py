# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for visualisation dashboard product
"""Real-surface tests for ``scpn_quantum_control.visualisation_dashboard_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.visualisation_dashboard_product as visualisation_dashboard_product
from scpn_quantum_control.visualisation_dashboard_product import (
    VISUALISATION_DASHBOARD_CLAIM_BOUNDARY,
    VISUALISATION_DASHBOARD_PRODUCT_SCHEMA,
    MaterialisedStaticReportProbe,
    PathEligibilityDecision,
    SecretsScanResult,
    VisualisationPanelRow,
    assert_visualisation_dashboard_product_integrity,
    build_visualisation_dashboard_product_registry,
    decide_visualisation_path,
    get_visualisation_panel,
    iter_visualisation_panels,
    list_visualisation_panel_ids,
    map_visualisation_dashboard_public_surfaces,
    materialise_demo_static_report_probe,
    scan_export_for_secrets,
)


def _registry_panels(registry: dict[str, object]) -> list[dict[str, object]]:
    """Narrow a validated registry panel collection for drift fixtures."""
    raw = registry["panels"]
    assert isinstance(raw, list)
    return cast(list[dict[str, object]], raw)


def test_list_and_filters() -> None:
    """Expose the stable catalogue and deterministic posture filters."""
    ids = list_visualisation_panel_ids()
    assert "order_parameter_energy_loss" in ids
    assert "gradient_norm" in ids
    assert "refuse_live_qpu_stream" in ids
    assert ids == list_visualisation_panel_ids()
    mat = iter_visualisation_panels(support_posture="fixture_materialised")
    assert mat
    assert all(row.support_posture == "fixture_materialised" for row in mat)
    refuse = iter_visualisation_panels(support_posture="refuse_only")
    assert refuse


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve known panels while rejecting blank and unknown identifiers."""
    row = get_visualisation_panel("order_parameter_energy_loss")
    assert row.live_qpu is False
    assert row.claim_boundary == VISUALISATION_DASHBOARD_CLAIM_BOUNDARY
    with pytest.raises(ValueError, match="non-empty"):
        get_visualisation_panel("  ")
    with pytest.raises(ValueError, match="unknown panel_id"):
        get_visualisation_panel("not_a_panel")


def test_path_eligibility_and_secrets() -> None:
    """Allow static fixtures while refusing unsafe paths and secret exports."""
    allowed = decide_visualisation_path(fixture_driven=True)
    assert allowed.allowed is True
    qpu = decide_visualisation_path(request_live_qpu_stream=True)
    assert qpu.allowed is False
    saas = decide_visualisation_path(request_saas_dashboard=True)
    assert saas.allowed is False
    no_fix = decide_visualisation_path(fixture_driven=False)
    assert no_fix.allowed is False

    clean = scan_export_for_secrets('{"order":[0.1,0.2]}')
    assert clean.clean is True
    dirty = scan_export_for_secrets("api_key=sk-abcdefghijklmnopqrst")
    assert dirty.clean is False
    assert dirty.findings
    with pytest.raises(ValueError, match="string"):
        scan_export_for_secrets(cast(Any, 123))


def test_materialise_demo_static_report_probe() -> None:
    """Materialise a deterministic secret-clean static report probe."""
    probe = materialise_demo_static_report_probe()
    assert probe.live_qpu is False
    assert probe.secrets_clean is True
    assert probe.series_point_count == 5
    assert probe.gradient_norm_count == 5
    assert len(probe.fixture_digest_sha256) == 64
    assert "order_parameter_energy_loss" in probe.panel_ids
    assert "gradient_norm" in probe.panel_ids
    payload = probe.to_dict()
    assert payload["live_qpu"] is False
    # digest is deterministic for fixed fixture
    again = materialise_demo_static_report_probe()
    assert again.fixture_digest_sha256 == probe.fixture_digest_sha256


def test_public_surfaces_and_registry() -> None:
    """Map public owners and validate the complete dashboard registry."""
    surfaces = map_visualisation_dashboard_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.visualisation_dashboard_product" in paths
    assert "scpn_quantum_control.differentiable_dashboard" in paths

    registry = build_visualisation_dashboard_product_registry()
    assert registry["schema"] == VISUALISATION_DASHBOARD_PRODUCT_SCHEMA
    assert registry["live_qpu_policy"] is False
    validated = assert_visualisation_dashboard_product_integrity(registry)
    assert validated["panel_count"] == len(list_visualisation_panel_ids())
    assert assert_visualisation_dashboard_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_live_qpu() -> None:
    """Reject panel-set drift and any live-QPU policy relaxation."""
    registry = build_visualisation_dashboard_product_registry()
    panels = _registry_panels(registry)

    broken = dict(registry)
    broken["panels"] = panels + [
        {
            "panel_id": "ghost",
            "kind": "gradient_norm",
            "title": "t",
            "summary": "s",
            "module_path": "m",
            "symbol_name": "x",
            "support_posture": "catalogue_only",
            "live_qpu": False,
            "as_of": "2026-07-24",
            "claim_boundary": VISUALISATION_DASHBOARD_CLAIM_BOUNDARY,
        }
    ]
    broken["panel_count"] = len(cast(list[object], broken["panels"]))
    with pytest.raises(ValueError, match="drift"):
        assert_visualisation_dashboard_product_integrity(broken)

    empty: dict[str, object] = {"panels": [], "blank_entry_count": 0, "panel_count": 0}
    with pytest.raises(ValueError, match="non-empty panels"):
        assert_visualisation_dashboard_product_integrity(empty)

    live = dict(registry)
    live_rows = [dict(row) for row in panels]
    live_rows[0]["live_qpu"] = True
    live["panels"] = live_rows
    with pytest.raises(ValueError, match="live_qpu"):
        assert_visualisation_dashboard_product_integrity(live)

    policy = dict(registry)
    policy["live_qpu_policy"] = True
    with pytest.raises(ValueError, match="live_qpu_policy"):
        assert_visualisation_dashboard_product_integrity(policy)


def test_integrity_rejects_blank_invalid() -> None:
    """Reject malformed rows, missing sentinels, duplicates, and count drift."""
    registry = build_visualisation_dashboard_product_registry()
    panels = _registry_panels(registry)

    non_map = dict(registry)
    non_map["panels"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_visualisation_dashboard_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in panels]
    rows[0]["panel_id"] = "  "
    blank_id["panels"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_visualisation_dashboard_product_integrity(blank_id)

    bad_kind = dict(registry)
    krows = [dict(row) for row in panels]
    krows[1]["kind"] = "nope"
    bad_kind["panels"] = krows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_visualisation_dashboard_product_integrity(bad_kind)

    no_symbol = dict(registry)
    srows = [dict(row) for row in panels]
    srows[0]["symbol_name"] = ""
    no_symbol["panels"] = srows
    with pytest.raises(ValueError, match="symbol_name"):
        assert_visualisation_dashboard_product_integrity(no_symbol)

    no_default = dict(registry)
    renamed = [dict(row) for row in panels]
    for row in renamed:
        if row.get("panel_id") == "order_parameter_energy_loss":
            row["panel_id"] = "renamed"
    no_default["panels"] = renamed
    with pytest.raises(ValueError, match="missing order_parameter|drift"):
        assert_visualisation_dashboard_product_integrity(no_default)

    no_refuse = dict(registry)
    without = [dict(row) for row in panels if row.get("panel_id") != "refuse_live_qpu_stream"]
    no_refuse["panels"] = without
    no_refuse["panel_count"] = len(without)
    with pytest.raises(ValueError, match="missing refuse_live_qpu|drift"):
        assert_visualisation_dashboard_product_integrity(no_refuse)

    dup = dict(registry)
    drows = [dict(row) for row in panels]
    drows.append(dict(drows[0]))
    dup["panels"] = drows
    dup["panel_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate panel_id"):
        assert_visualisation_dashboard_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_visualisation_dashboard_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["panel_count"] = 0
    with pytest.raises(ValueError, match="panel_count"):
        assert_visualisation_dashboard_product_integrity(count_mismatch)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("schema", "visualisation_dashboard_product.v1", "product schema"),
        ("claim_boundary", "drifted claim", "claim_boundary"),
        ("policy_note", "drifted policy", "policy_note"),
        ("default_panel_id", "gradient_norm", "default_panel_id"),
    ),
)
def test_integrity_rejects_governed_metadata_drift(
    field: str,
    value: str,
    error: str,
) -> None:
    """Reject stale schemas and drifted governed registry metadata."""
    registry = build_visualisation_dashboard_product_registry()
    broken = dict(registry)
    broken[field] = value
    with pytest.raises(ValueError, match=error):
        assert_visualisation_dashboard_product_integrity(broken)


def test_integrity_rejects_canonical_row_and_surface_drift() -> None:
    """Reject transported panel and public-surface mutations exactly."""
    registry = build_visualisation_dashboard_product_registry()
    panels = [dict(row) for row in _registry_panels(registry)]
    panels[0]["summary"] = "drifted summary"
    row_drift = dict(registry)
    row_drift["panels"] = panels
    with pytest.raises(ValueError, match="panel row 0 drift"):
        assert_visualisation_dashboard_product_integrity(row_drift)

    surface_drift = dict(registry)
    surface_drift["public_surfaces"] = []
    with pytest.raises(ValueError, match="public_surfaces"):
        assert_visualisation_dashboard_product_integrity(surface_drift)


def test_module_exports() -> None:
    """Keep every documented dashboard product entry point public."""
    assert "materialise_demo_static_report_probe" in visualisation_dashboard_product.__all__
    assert "scan_export_for_secrets" in visualisation_dashboard_product.__all__
    assert "list_visualisation_panel_ids" in visualisation_dashboard_product.__all__


def test_row_decision_probe_validation() -> None:
    """Enforce immutable row, decision, scan, and probe invariants."""
    base: dict[str, Any] = {
        "panel_id": "x",
        "kind": "gradient_norm",
        "title": "t",
        "summary": "s",
        "module_path": "m",
        "symbol_name": "fn",
        "support_posture": "fixture_materialised",
    }
    assert VisualisationPanelRow(**base).panel_id == "x"
    with pytest.raises(ValueError, match="claim_boundary"):
        VisualisationPanelRow(**{**base, "claim_boundary": "drifted claim"})
    with pytest.raises(ValueError, match="panel_id"):
        VisualisationPanelRow(**{**base, "panel_id": ""})
    with pytest.raises(ValueError, match="kind"):
        VisualisationPanelRow(**{**base, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="title"):
        VisualisationPanelRow(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        VisualisationPanelRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="module_path"):
        VisualisationPanelRow(**{**base, "module_path": ""})
    with pytest.raises(ValueError, match="symbol_name"):
        VisualisationPanelRow(**{**base, "symbol_name": ""})
    with pytest.raises(ValueError, match="support_posture"):
        VisualisationPanelRow(**{**base, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="live_qpu"):
        VisualisationPanelRow(**{**base, "live_qpu": True})
    with pytest.raises(ValueError, match="as_of"):
        VisualisationPanelRow(**{**base, "as_of": ""})

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
    assert decide_visualisation_path().to_dict()["allowed"] is True
    with pytest.raises(ValueError, match="claim_boundary"):
        PathEligibilityDecision(
            outcome="allowed",
            allowed=True,
            reason="fixture-driven static path",
            blockers=(),
            claim_boundary="drifted claim",
        )

    with pytest.raises(ValueError, match="clean scans cannot list findings"):
        SecretsScanResult(clean=True, findings=("x",))
    with pytest.raises(ValueError, match="dirty scans require findings"):
        SecretsScanResult(clean=False, findings=())
    with pytest.raises(ValueError, match="findings entries"):
        SecretsScanResult(clean=False, findings=("",))
    with pytest.raises(ValueError, match="claim_boundary"):
        SecretsScanResult(clean=True, findings=(), claim_boundary="drifted claim")

    with pytest.raises(ValueError, match="panel_ids must be non-empty"):
        MaterialisedStaticReportProbe(
            panel_ids=(),
            series_point_count=1,
            gradient_norm_count=1,
            fixture_digest_sha256="a" * 64,
            live_qpu=False,
            secrets_clean=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="series_point_count"):
        MaterialisedStaticReportProbe(
            panel_ids=("a",),
            series_point_count=0,
            gradient_norm_count=1,
            fixture_digest_sha256="a" * 64,
            live_qpu=False,
            secrets_clean=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="gradient_norm_count"):
        MaterialisedStaticReportProbe(
            panel_ids=("a",),
            series_point_count=1,
            gradient_norm_count=0,
            fixture_digest_sha256="a" * 64,
            live_qpu=False,
            secrets_clean=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="fixture_digest"):
        MaterialisedStaticReportProbe(
            panel_ids=("a",),
            series_point_count=1,
            gradient_norm_count=1,
            fixture_digest_sha256="short",
            live_qpu=False,
            secrets_clean=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="fixture_digest"):
        MaterialisedStaticReportProbe(
            panel_ids=("a",),
            series_point_count=1,
            gradient_norm_count=1,
            fixture_digest_sha256="G" * 64,
            live_qpu=False,
            secrets_clean=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="live_qpu"):
        MaterialisedStaticReportProbe(
            panel_ids=("a",),
            series_point_count=1,
            gradient_norm_count=1,
            fixture_digest_sha256="a" * 64,
            live_qpu=True,
            secrets_clean=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="secrets_clean"):
        MaterialisedStaticReportProbe(
            panel_ids=("a",),
            series_point_count=1,
            gradient_norm_count=1,
            fixture_digest_sha256="a" * 64,
            live_qpu=False,
            secrets_clean=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedStaticReportProbe(
            panel_ids=("a",),
            series_point_count=1,
            gradient_norm_count=1,
            fixture_digest_sha256="a" * 64,
            live_qpu=False,
            secrets_clean=True,
            demo_label="",
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        MaterialisedStaticReportProbe(
            panel_ids=("a",),
            series_point_count=1,
            gradient_norm_count=1,
            fixture_digest_sha256="a" * 64,
            live_qpu=False,
            secrets_clean=True,
            demo_label="d",
            claim_boundary="drifted claim",
        )


def test_probe_refused_when_path_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop materialisation when the dashboard path policy refuses it."""

    def _refuse(**_kwargs: Any) -> PathEligibilityDecision:
        return PathEligibilityDecision(
            outcome="refused",
            allowed=False,
            reason="forced refuse",
            blockers=("forced",),
        )

    monkeypatch.setattr(visualisation_dashboard_product, "decide_visualisation_path", _refuse)
    with pytest.raises(ValueError, match="refused"):
        materialise_demo_static_report_probe()


def test_secrets_scan_result_to_dict() -> None:
    """SecretsScanResult.to_dict must expose clean flag, findings, and claim boundary."""
    dirty = SecretsScanResult(clean=False, findings=("api_key pattern",))
    payload = dirty.to_dict()
    assert payload["clean"] is False
    assert payload["findings"] == ["api_key pattern"]
    assert payload["claim_boundary"] == VISUALISATION_DASHBOARD_CLAIM_BOUNDARY

    clean = SecretsScanResult(clean=True, findings=())
    assert clean.to_dict()["findings"] == []


def test_materialised_probe_rejects_blank_panel_id_entry() -> None:
    """MaterialisedStaticReportProbe must refuse blank entries in panel_ids."""
    with pytest.raises(ValueError, match="panel_ids entries must be non-empty"):
        MaterialisedStaticReportProbe(
            panel_ids=("order_parameter_energy_loss", "  "),
            series_point_count=1,
            gradient_norm_count=1,
            fixture_digest_sha256="a" * 64,
            live_qpu=False,
            secrets_clean=True,
            demo_label="d",
        )


def test_iter_visualisation_panels_by_kind() -> None:
    """Kind filter must select only matching panel families."""
    grads = iter_visualisation_panels(kind="gradient_norm")
    assert grads
    assert all(row.kind == "gradient_norm" for row in grads)
    assert {row.panel_id for row in grads} == {"gradient_norm"}


def test_catalogue_map_rejects_blank_panel_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """_catalogue_map must raise when a catalogue row has a blank panel_id."""
    good = get_visualisation_panel("gradient_norm")
    blank = VisualisationPanelRow(
        panel_id="tmp",
        kind="gradient_norm",
        title="t",
        summary="s",
        module_path="m",
        symbol_name="fn",
        support_posture="catalogue_only",
    )
    object.__setattr__(blank, "panel_id", "  ")
    monkeypatch.setattr(visualisation_dashboard_product, "_CANONICAL_PANELS", (blank,))
    with pytest.raises(RuntimeError, match="blank panel_id"):
        visualisation_dashboard_product._catalogue_map()
    # restore path not required — monkeypatch undoes; sanity on good id
    assert good.panel_id == "gradient_norm"


def test_catalogue_map_rejects_duplicate_panel_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """_catalogue_map must raise on duplicate panel_id values."""
    row = get_visualisation_panel("gradient_norm")
    monkeypatch.setattr(visualisation_dashboard_product, "_CANONICAL_PANELS", (row, row))
    with pytest.raises(RuntimeError, match="duplicate panel_id"):
        visualisation_dashboard_product._catalogue_map()


def test_catalogue_map_rejects_empty_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    """_catalogue_map must refuse an empty panel catalogue."""
    monkeypatch.setattr(visualisation_dashboard_product, "_CANONICAL_PANELS", ())
    with pytest.raises(RuntimeError, match="non-empty"):
        visualisation_dashboard_product._catalogue_map()


def test_catalogue_map_accepts_valid_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """_catalogue_map must index a valid single-row catalogue by panel_id."""
    good = get_visualisation_panel("gradient_norm")
    monkeypatch.setattr(visualisation_dashboard_product, "_CANONICAL_PANELS", (good,))
    mapped = visualisation_dashboard_product._catalogue_map()
    assert mapped[good.panel_id].panel_id == good.panel_id


def test_materialise_rejects_stale_fixture_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject the superseded deterministic fixture schema without an alias."""

    def _stale_fixture() -> dict[str, object]:
        return {
            "schema": "visualisation_demo_fixture.v1",
            "order_parameter": [0.1],
            "energy_loss": [1.0],
            "gradient_norms": [0.1],
            "live_qpu": False,
        }

    monkeypatch.setattr(visualisation_dashboard_product, "_demo_fixture_payload", _stale_fixture)
    with pytest.raises(ValueError, match="fixture schema"):
        materialise_demo_static_report_probe()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("claim_boundary", "drifted claim", "claim_boundary"),
        ("live_qpu", True, "live_qpu"),
    ),
)
def test_materialise_rejects_fixture_governance_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    error: str,
) -> None:
    """Reject drift in governed fields of the deterministic fixture."""

    def _drifted_fixture() -> dict[str, object]:
        fixture: dict[str, object] = {
            "schema": "visualisation_demo_fixture.v2",
            "order_parameter": [0.1],
            "energy_loss": [1.0],
            "gradient_norms": [0.1],
            "live_qpu": False,
            "claim_boundary": VISUALISATION_DASHBOARD_CLAIM_BOUNDARY,
        }
        fixture[field] = value
        return fixture

    monkeypatch.setattr(
        visualisation_dashboard_product,
        "_demo_fixture_payload",
        _drifted_fixture,
    )
    with pytest.raises(ValueError, match=error):
        materialise_demo_static_report_probe()


def test_materialise_rejects_non_list_series(monkeypatch: pytest.MonkeyPatch) -> None:
    """materialise_demo_static_report_probe refuses non-list series fields."""

    def _bad_fixture() -> dict[str, object]:
        return {
            "schema": "visualisation_demo_fixture.v2",
            "order_parameter": "not-a-list",
            "energy_loss": [1.0],
            "gradient_norms": [0.1],
            "live_qpu": False,
            "claim_boundary": VISUALISATION_DASHBOARD_CLAIM_BOUNDARY,
        }

    monkeypatch.setattr(visualisation_dashboard_product, "_demo_fixture_payload", _bad_fixture)
    with pytest.raises(ValueError, match="must be lists"):
        materialise_demo_static_report_probe()


def test_materialise_rejects_mismatched_order_energy_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """materialise_demo_static_report_probe refuses unequal order/energy series."""

    def _bad_fixture() -> dict[str, object]:
        return {
            "schema": "visualisation_demo_fixture.v2",
            "order_parameter": [0.1, 0.2],
            "energy_loss": [1.0],
            "gradient_norms": [0.1],
            "live_qpu": False,
            "claim_boundary": VISUALISATION_DASHBOARD_CLAIM_BOUNDARY,
        }

    monkeypatch.setattr(visualisation_dashboard_product, "_demo_fixture_payload", _bad_fixture)
    with pytest.raises(ValueError, match="non-empty equal length"):
        materialise_demo_static_report_probe()


def test_materialise_rejects_empty_order_series(monkeypatch: pytest.MonkeyPatch) -> None:
    """materialise_demo_static_report_probe refuses empty equal-length order/energy."""

    def _bad_fixture() -> dict[str, object]:
        return {
            "schema": "visualisation_demo_fixture.v2",
            "order_parameter": [],
            "energy_loss": [],
            "gradient_norms": [0.1],
            "live_qpu": False,
            "claim_boundary": VISUALISATION_DASHBOARD_CLAIM_BOUNDARY,
        }

    monkeypatch.setattr(visualisation_dashboard_product, "_demo_fixture_payload", _bad_fixture)
    with pytest.raises(ValueError, match="non-empty equal length"):
        materialise_demo_static_report_probe()


def test_materialise_rejects_empty_gradient_norms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """materialise_demo_static_report_probe refuses empty gradient_norms."""

    def _bad_fixture() -> dict[str, object]:
        return {
            "schema": "visualisation_demo_fixture.v2",
            "order_parameter": [0.1],
            "energy_loss": [1.0],
            "gradient_norms": [],
            "live_qpu": False,
            "claim_boundary": VISUALISATION_DASHBOARD_CLAIM_BOUNDARY,
        }

    monkeypatch.setattr(visualisation_dashboard_product, "_demo_fixture_payload", _bad_fixture)
    with pytest.raises(ValueError, match="gradient_norms must be non-empty"):
        materialise_demo_static_report_probe()


def test_materialise_rejects_dirty_secrets_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """materialise_demo_static_report_probe fails closed when secrets scan is dirty."""

    def _dirty(_text: str) -> SecretsScanResult:
        return SecretsScanResult(clean=False, findings=("token pattern",))

    monkeypatch.setattr(visualisation_dashboard_product, "scan_export_for_secrets", _dirty)
    with pytest.raises(ValueError, match="failed secrets scan"):
        materialise_demo_static_report_probe()
