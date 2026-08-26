# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for polyglot edge Program-AD product
"""Real-surface tests for ``polyglot_edge_ad_product``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

import scpn_quantum_control.polyglot_edge_ad_product as edge
from scpn_quantum_control.polyglot_edge_ad_product import (
    POLYGLOT_EDGE_AD_CERTIFICATE_SCHEMA,
    POLYGLOT_EDGE_AD_CLAIM_BOUNDARY,
    POLYGLOT_EDGE_AD_PRODUCT_SCHEMA,
    CommittedWasmReplayCertificate,
    EdgeADPathDecision,
    EdgeADRuntimeRow,
    assert_polyglot_edge_ad_product_integrity,
    build_polyglot_edge_ad_product_registry,
    decide_edge_ad_path,
    get_edge_ad_runtime,
    iter_edge_ad_runtimes,
    list_edge_ad_runtime_ids,
    load_committed_wasm_replay_payload,
    map_polyglot_edge_ad_public_surfaces,
    materialise_wasm_replay_certificate,
)
from scpn_quantum_control.studio.program_ad_replay_artifact import (
    PROGRAM_AD_REPLAY_ARTIFACT_ID,
    PROGRAM_AD_REPLAY_SCHEMA,
)


def test_runtime_catalogue_and_filters() -> None:
    """Catalogue rows expose stable support and routing metadata."""
    ids = list_edge_ad_runtime_ids()
    assert ids == (
        "rust_native_replay",
        "browser_wasm_replay",
        "julia_program_ad",
    )
    assert ids == list_edge_ad_runtime_ids()
    wasm = get_edge_ad_runtime("browser_wasm_replay")
    assert wasm.support_posture == "committed_sample_bitexact"
    assert wasm.silent_host_fallback is False
    assert wasm.general_program_ad is False
    assert "replay" in wasm.studio_verb_ids
    assert wasm.to_dict()["runtime_id"] == "browser_wasm_replay"
    unsupported = iter_edge_ad_runtimes(support_posture="boundary_unsupported")
    assert tuple(row.runtime_id for row in unsupported) == ("julia_program_ad",)
    assert iter_edge_ad_runtimes(support_posture="bounded_authority")[0].runtime_id == (
        "rust_native_replay"
    )
    assert tuple(row.runtime_id for row in iter_edge_ad_runtimes()) == ids


def test_get_runtime_fails_closed() -> None:
    """Blank and unknown runtime identifiers fail closed."""
    with pytest.raises(ValueError, match="non-empty"):
        get_edge_ad_runtime(" ")
    with pytest.raises(ValueError, match="unknown runtime_id"):
        get_edge_ad_runtime("ghost")


def test_wasm_path_requires_exact_verified_sample() -> None:
    """Browser WASM admits only the verified committed replay."""
    allowed = decide_edge_ad_path(
        "browser_wasm_replay",
        studio_verb_id="replay",
    )
    assert allowed.allowed is True
    assert allowed.outcome == "allowed"
    assert allowed.host_fallback_used is False
    assert allowed.to_dict()["blockers"] == []

    refused = decide_edge_ad_path(
        "browser_wasm_replay",
        studio_verb_id="differentiate",
        artifact_payload={"schema": "tampered"},
        request_host_fallback=True,
        request_general_program_ad=True,
        committed_sample_only=False,
    )
    assert refused.allowed is False
    assert len(refused.blockers) >= 5
    assert any("artefact" in item for item in refused.blockers)
    assert any("fallback" in item for item in refused.blockers)
    assert any("general" in item for item in refused.blockers)
    assert any("committed" in item for item in refused.blockers)
    assert any("verb" in item for item in refused.blockers)


def test_julia_and_native_path_boundaries() -> None:
    """Native bounded replay is admitted while Julia Program AD is refused."""
    julia = decide_edge_ad_path("julia_program_ad", studio_verb_id="differentiate")
    assert julia.allowed is False
    assert any("Kuramoto-only" in item for item in julia.blockers)

    native = decide_edge_ad_path("rust_native_replay", studio_verb_id="differentiate")
    assert native.allowed is True
    blank = decide_edge_ad_path("rust_native_replay", studio_verb_id=" ")
    assert blank.allowed is False
    assert any("non-empty" in item for item in blank.blockers)


def test_committed_wasm_certificate_is_supported() -> None:
    """The committed rational replay composes valid artifact and parity evidence."""
    certificate = materialise_wasm_replay_certificate()
    assert certificate.schema == POLYGLOT_EDGE_AD_CERTIFICATE_SCHEMA
    assert certificate.artifact_verified is True
    assert certificate.parity_verified is True
    assert certificate.supported is True
    assert certificate.expected_value == 19.0
    assert certificate.expected_gradient == (6.0, 2.0)
    assert certificate.input_sha256.startswith("sha256:")
    assert certificate.blockers == ()
    payload = certificate.to_dict()
    assert payload["supported"] is True
    assert payload["expected_gradient"] == [6.0, 2.0]


def test_tampered_wasm_certificate_fails_closed() -> None:
    """A changed replay digest prevents browser support."""
    payload = load_committed_wasm_replay_payload()
    payload["input_sha256"] = "sha256:" + "0" * 64
    certificate = materialise_wasm_replay_certificate(payload)
    assert certificate.supported is False
    assert certificate.artifact_verified is False
    assert certificate.parity_verified is True
    assert certificate.blockers


def test_certificate_handles_validation_and_parity_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Certificate composition preserves validation and parity blockers."""
    monkeypatch.setattr(
        edge,
        "inspect_program_ad_replay_artifact",
        lambda payload: (_ for _ in ()).throw(RuntimeError("engine missing")),
    )
    failed = materialise_wasm_replay_certificate({"schema": "x"})
    assert failed.supported is False
    assert any("engine missing" in item for item in failed.blockers)

    class _ParityDecision:
        passed = False
        blockers: tuple[str, ...] = ()
        reason = "parity refused"

    monkeypatch.setattr(
        edge,
        "inspect_program_ad_replay_artifact",
        lambda payload: type("V", (), {"passed": False, "errors": ()})(),
    )
    monkeypatch.setattr(edge, "verify_certificate", lambda *a, **k: _ParityDecision())
    parity_failed = materialise_wasm_replay_certificate({})
    assert parity_failed.supported is False
    assert "parity refused" in parity_failed.blockers

    class _ParityPassed:
        passed = True
        blockers: tuple[str, ...] = ()
        reason = "passed"

    monkeypatch.setattr(edge, "verify_certificate", lambda *a, **k: _ParityPassed())
    unsupported = materialise_wasm_replay_certificate({})
    assert unsupported.blockers == ("composed WASM replay certificate is unsupported",)


def test_certificate_rejects_bad_trusted_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nominal validation cannot promote malformed certificate fields."""

    class _Validation:
        passed = True
        errors: tuple[str, ...] = ()

    monkeypatch.setattr(edge, "inspect_program_ad_replay_artifact", lambda payload: _Validation())
    cert = materialise_wasm_replay_certificate({"artifact_id": 3})
    assert cert.artifact_verified is False
    assert any("identifiers" in item for item in cert.blockers)


def test_strict_committed_payload_loader(tmp_path: Path) -> None:
    """Committed JSON loading rejects non-objects, duplicate keys, and NaN."""
    good = tmp_path / "good.json"
    good.write_text('{"a": 1}\n', encoding="utf-8")
    assert load_committed_wasm_replay_payload(good) == {"a": 1}

    root_list = tmp_path / "list.json"
    root_list.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_committed_wasm_replay_payload(root_list)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_committed_wasm_replay_payload(duplicate)

    nonstandard = tmp_path / "nonstandard.json"
    nonstandard.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_committed_wasm_replay_payload(nonstandard)


def test_public_surfaces_and_registry() -> None:
    """Public surface and product registries are complete and valid."""
    surfaces = map_polyglot_edge_ad_public_surfaces()
    assert len(surfaces) == 3
    roles = {str(row["role"]) for row in surfaces}
    assert "polyglot_edge_ad_product_surface" in roles
    assert "committed_wasm_replay_artifact_authority" in roles
    assert "parity_certificate_subset_authority" in roles

    registry = build_polyglot_edge_ad_product_registry()
    assert registry["schema"] == POLYGLOT_EDGE_AD_PRODUCT_SCHEMA
    assert registry["runtime_count"] == 3
    assert registry["silent_host_fallback_policy"] is False
    assert registry["general_program_ad_policy"] is False
    validated = assert_polyglot_edge_ad_product_integrity(registry)
    assert validated["default_runtime_id"] == "browser_wasm_replay"
    assert assert_polyglot_edge_ad_product_integrity()["blank_entry_count"] == 0


def test_registry_integrity_rejects_shape_and_policy_drift() -> None:
    """Registry validation rejects malformed rows and runtime policy drift."""
    registry = build_polyglot_edge_ad_product_registry()
    runtimes = cast(list[dict[str, object]], registry["runtimes"])

    stale_schema = dict(registry)
    stale_schema["schema"] = "polyglot_edge_ad_product.v1"
    with pytest.raises(ValueError, match="unknown edge Program-AD product schema"):
        assert_polyglot_edge_ad_product_integrity(stale_schema)

    claim_drift = dict(registry)
    claim_drift["claim_boundary"] = "legacy planning label"
    with pytest.raises(ValueError, match="claim boundary drift"):
        assert_polyglot_edge_ad_product_integrity(claim_drift)

    bad_runtime_payloads: tuple[object, ...] = ([], "nope")
    for bad in bad_runtime_payloads:
        payload = dict(registry)
        payload["runtimes"] = bad
        with pytest.raises(ValueError, match="non-empty runtimes"):
            assert_polyglot_edge_ad_product_integrity(payload)

    non_map = dict(registry)
    non_map["runtimes"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_polyglot_edge_ad_product_integrity(non_map)

    blank = dict(registry)
    blank_rows = [dict(row) for row in runtimes]
    blank_rows[0]["runtime_id"] = " "
    blank["runtimes"] = blank_rows
    with pytest.raises(ValueError, match="blank runtime"):
        assert_polyglot_edge_ad_product_integrity(blank)

    duplicate = dict(registry)
    duplicate_rows = [dict(row) for row in runtimes]
    duplicate_rows.append(dict(duplicate_rows[0]))
    duplicate["runtimes"] = duplicate_rows
    duplicate["runtime_count"] = len(duplicate_rows)
    with pytest.raises(ValueError, match="duplicate runtime_id"):
        assert_polyglot_edge_ad_product_integrity(duplicate)

    for field, message in (
        ("silent_host_fallback", "runtime silent_host_fallback"),
        ("general_program_ad", "runtime general_program_ad"),
        ("claim_boundary", "runtime claim boundary drift"),
    ):
        drift = dict(registry)
        rows = [dict(row) for row in runtimes]
        rows[0][field] = "drift" if field == "claim_boundary" else True
        drift["runtimes"] = rows
        with pytest.raises(ValueError, match=message):
            assert_polyglot_edge_ad_product_integrity(drift)

    wasm_drift = dict(registry)
    rows = [dict(row) for row in runtimes]
    rows[1]["support_posture"] = "bounded_authority"
    wasm_drift["runtimes"] = rows
    with pytest.raises(ValueError, match="browser WASM"):
        assert_polyglot_edge_ad_product_integrity(wasm_drift)

    julia_drift = dict(registry)
    rows = [dict(row) for row in runtimes]
    rows[2]["support_posture"] = "bounded_authority"
    julia_drift["runtimes"] = rows
    with pytest.raises(ValueError, match="Julia Program-AD"):
        assert_polyglot_edge_ad_product_integrity(julia_drift)


def test_registry_integrity_rejects_counts_and_top_level_policies() -> None:
    """Registry counts, canonical ids, and top-level policies fail closed."""
    registry = build_polyglot_edge_ad_product_registry()
    runtimes = cast(list[dict[str, object]], registry["runtimes"])

    missing = dict(registry)
    missing["runtimes"] = [dict(runtimes[0])]
    missing["runtime_count"] = 1
    with pytest.raises(ValueError, match="browser WASM and Julia"):
        assert_polyglot_edge_ad_product_integrity(missing)

    count = dict(registry)
    count["runtime_count"] = 0
    with pytest.raises(ValueError, match="runtime_count"):
        assert_polyglot_edge_ad_product_integrity(count)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_polyglot_edge_ad_product_integrity(blank_count)

    fallback = dict(registry)
    fallback["silent_host_fallback_policy"] = True
    with pytest.raises(ValueError, match="silent_host_fallback_policy"):
        assert_polyglot_edge_ad_product_integrity(fallback)

    general = dict(registry)
    general["general_program_ad_policy"] = True
    with pytest.raises(ValueError, match="general_program_ad_policy"):
        assert_polyglot_edge_ad_product_integrity(general)

    surface_drift = dict(registry)
    surface_drift["public_surfaces"] = []
    with pytest.raises(ValueError, match="public surface map drift"):
        assert_polyglot_edge_ad_product_integrity(surface_drift)

    drift = dict(registry)
    rows = [dict(row) for row in runtimes]
    rows[0]["runtime_id"] = "renamed"
    drift["runtimes"] = rows
    with pytest.raises(ValueError, match="catalogue drift"):
        assert_polyglot_edge_ad_product_integrity(drift)


def test_runtime_row_validation() -> None:
    """Runtime dataclass validation covers every bounded-support invariant."""
    base: dict[str, Any] = {
        "runtime_id": "rust_native_replay",
        "runtime_kind": "native_rust",
        "title": "t",
        "summary": "s",
        "support_posture": "bounded_authority",
        "authority_pointer": "p",
        "studio_verb_ids": ("replay",),
        "wasm_safe_operations": ("add",),
        "max_ir_bytes": 1,
        "max_inputs": 1,
    }
    assert EdgeADRuntimeRow(**base).title == "t"
    for field, value, message in (
        ("runtime_id", "bad", "runtime_id"),
        ("runtime_kind", "bad", "runtime_kind"),
        ("title", "", "title"),
        ("summary", "", "summary"),
        ("support_posture", "bad", "support_posture"),
        ("authority_pointer", "", "authority_pointer"),
        ("studio_verb_ids", (), "studio_verb_ids"),
        ("max_ir_bytes", -1, "bounds"),
        ("silent_host_fallback", True, "silent_host_fallback"),
        ("general_program_ad", True, "general_program_ad"),
    ):
        with pytest.raises(ValueError, match=message):
            EdgeADRuntimeRow(**{**base, field: value})
    with pytest.raises(ValueError, match="studio_verb_ids"):
        EdgeADRuntimeRow(**{**base, "studio_verb_ids": ("replay", "")})
    with pytest.raises(ValueError, match="unique"):
        EdgeADRuntimeRow(**{**base, "studio_verb_ids": ("replay", "replay")})
    unsupported = {
        **base,
        "runtime_id": "julia_program_ad",
        "runtime_kind": "julia",
        "support_posture": "boundary_unsupported",
        "wasm_safe_operations": (),
        "max_ir_bytes": 0,
        "max_inputs": 0,
    }
    assert EdgeADRuntimeRow(**unsupported).support_posture == "boundary_unsupported"
    with pytest.raises(ValueError, match="zero bounds"):
        EdgeADRuntimeRow(**{**unsupported, "max_inputs": 1})
    with pytest.raises(ValueError, match="wasm-safe"):
        EdgeADRuntimeRow(**{**unsupported, "wasm_safe_operations": ("add",)})
    with pytest.raises(ValueError, match="positive bounds"):
        EdgeADRuntimeRow(**{**base, "max_inputs": 0})


def test_decision_validation() -> None:
    """Path decisions require coherent outcomes, blockers, and no fallback."""
    good = EdgeADPathDecision(
        runtime_id="r",
        outcome="allowed",
        allowed=True,
        reason="ok",
        blockers=(),
    )
    assert good.allowed is True
    for kwargs, message in (
        ({"runtime_id": ""}, "runtime_id"),
        ({"outcome": cast(Any, "bad")}, "outcome"),
        ({"reason": ""}, "reason"),
        ({"outcome": "refused", "allowed": True}, "agree"),
        ({"blockers": ("bad",)}, "cannot list"),
        ({"outcome": "refused", "allowed": False}, "require blockers"),
        ({"outcome": "refused", "allowed": False, "blockers": ("",)}, "non-empty"),
        ({"host_fallback_used": True}, "host_fallback_used"),
    ):
        values: dict[str, Any] = {
            "runtime_id": "r",
            "outcome": "allowed",
            "allowed": True,
            "reason": "ok",
            "blockers": (),
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=message):
            EdgeADPathDecision(**values)


def test_certificate_validation() -> None:
    """Certificate validation refuses inconsistent support evidence."""
    good: dict[str, Any] = {
        "schema": POLYGLOT_EDGE_AD_CERTIFICATE_SCHEMA,
        "artifact_id": PROGRAM_AD_REPLAY_ARTIFACT_ID,
        "artifact_schema": PROGRAM_AD_REPLAY_SCHEMA,
        "input_sha256": "sha256:" + "a" * 64,
        "expected_value": 19.0,
        "expected_gradient": (6.0, 2.0),
        "parity_family_id": "value_and_gradient_replay",
        "artifact_verified": True,
        "parity_verified": True,
        "supported": True,
        "blockers": (),
    }
    assert CommittedWasmReplayCertificate(**good).supported is True
    cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"schema": "bad"}, "schema"),
        ({"parity_family_id": ""}, "parity_family_id"),
        ({"blockers": ("",)}, "blockers"),
        ({"artifact_verified": False}, "both verifications"),
        ({"blockers": ("bad",)}, "cannot list"),
        ({"artifact_id": "bad"}, "artifact_id"),
        ({"artifact_schema": "bad"}, "artifact_schema"),
        ({"input_sha256": "bad"}, "SHA-256"),
        ({"expected_value": None}, "expected value"),
    )
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            CommittedWasmReplayCertificate(**{**good, **changes})
    unsupported = {**good, "supported": False, "artifact_verified": False, "blockers": ()}
    with pytest.raises(ValueError, match="require blockers"):
        CommittedWasmReplayCertificate(**unsupported)


def test_trusted_artifact_field_validation() -> None:
    """Trusted artifact extraction validates identifiers and numeric fields."""
    base: dict[str, object] = {
        "artifact_id": "id",
        "schema": "schema",
        "input_sha256": "sha256:" + "a" * 64,
        "expected": {"value": 1.0, "gradient": [2.0]},
    }
    assert edge._trusted_artifact_fields(base)[3:] == (1.0, (2.0,))
    for payload, message in (
        ({**base, "artifact_id": 1}, "identifiers"),
        ({**base, "input_sha256": 1}, "input_sha256"),
        ({**base, "expected": []}, "mapping"),
        ({**base, "expected": {"value": True, "gradient": [2.0]}}, "numeric"),
        ({**base, "expected": {"value": 1.0, "gradient": []}}, "non-empty"),
        ({**base, "expected": {"value": 1.0, "gradient": [True]}}, "numeric"),
    ):
        with pytest.raises(ValueError, match=message):
            edge._trusted_artifact_fields(payload)


def test_catalogue_defensive_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defensive catalogue construction rejects empty, blank, and duplicate ids."""
    monkeypatch.setattr(edge, "_RUNTIMES", ())
    with pytest.raises(RuntimeError, match="must be non-empty"):
        edge._runtime_map()

    row = EdgeADRuntimeRow(
        runtime_id="rust_native_replay",
        runtime_kind="native_rust",
        title="t",
        summary="s",
        support_posture="bounded_authority",
        authority_pointer="p",
        studio_verb_ids=("replay",),
        wasm_safe_operations=("add",),
        max_ir_bytes=1,
        max_inputs=1,
    )
    object.__setattr__(row, "runtime_id", " ")
    monkeypatch.setattr(edge, "_RUNTIMES", (row,))
    with pytest.raises(RuntimeError, match="blank runtime_id"):
        edge._runtime_map()
    object.__setattr__(row, "runtime_id", "rust_native_replay")
    monkeypatch.setattr(edge, "_RUNTIMES", (row, row))
    with pytest.raises(RuntimeError, match="duplicate runtime_id"):
        edge._runtime_map()


def test_module_exports_and_claim_boundary() -> None:
    """Public exports include core entry points and retain the Julia boundary."""
    assert "materialise_wasm_replay_certificate" in edge.__all__
    assert "decide_edge_ad_path" in edge.__all__
    assert "julia" in POLYGLOT_EDGE_AD_CLAIM_BOUNDARY.lower()
    assert json.loads(json.dumps(build_polyglot_edge_ad_product_registry()))
