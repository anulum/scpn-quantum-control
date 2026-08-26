# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for compiler boundary product
"""Real-surface tests for ``scpn_quantum_control.compiler_boundary_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.compiler_boundary_product as compiler_boundary_product
from scpn_quantum_control.compiler_boundary_product import (
    COMPILER_BOUNDARY_CLAIM_BOUNDARY,
    COMPILER_BOUNDARY_PRODUCT_SCHEMA,
    CompilerBoundaryRow,
    MaterialisedCompilerBoundaryProbe,
    PathEligibilityDecision,
    assert_compiler_boundary_product_integrity,
    build_compiler_boundary_product_registry,
    decide_compiler_path,
    get_compiler_boundary,
    iter_compiler_boundaries,
    list_compiler_ids,
    map_compiler_boundary_public_surfaces,
    materialise_compiler_boundary_probe,
    materialise_demo_compiler_boundary_probe,
)


def _registry_compilers(registry: dict[str, object]) -> list[dict[str, object]]:
    """Narrow a validated registry compiler collection for drift fixtures."""
    raw = registry["compilers"]
    assert isinstance(raw, list)
    return cast(list[dict[str, object]], raw)


def test_list_and_filters() -> None:
    """Expose the stable compiler catalogue and deterministic filters."""
    ids = list_compiler_ids()
    assert "qir" in ids
    assert "cudaq" in ids
    assert "catalyst_external" in ids
    assert "mlir_enzyme_in_tree" in ids
    assert len(ids) == 5
    permanent = iter_compiler_boundaries(status="permanent_boundary")
    assert permanent
    assert all(row.status == "permanent_boundary" for row in permanent)
    assert all(row.import_export_allowed is False for row in permanent)
    empty = iter_compiler_boundaries(status="supported")
    assert empty == ()
    policy = iter_compiler_boundaries(support_posture="policy_only")
    assert policy


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve known compilers while rejecting blank and unknown ids."""
    row = get_compiler_boundary("qir")
    assert row.claim_boundary == COMPILER_BOUNDARY_CLAIM_BOUNDARY
    assert row.import_export_allowed is True
    assert row.invent_green_runtime is False
    cudaq = get_compiler_boundary("cudaq")
    assert cudaq.status == "permanent_boundary"
    assert cudaq.import_export_allowed is False
    with pytest.raises(ValueError, match="non-empty"):
        get_compiler_boundary("  ")
    with pytest.raises(ValueError, match="unknown compiler_id"):
        get_compiler_boundary("not_a_compiler")


def test_decide_compiler_path() -> None:
    """Allow validate-only paths and refuse invent-green execution."""
    qir = decide_compiler_path("qir", request_import_export=True)
    assert qir.allowed is True

    cudaq_export = decide_compiler_path("cudaq", request_import_export=True)
    assert cudaq_export.allowed is False
    assert any(
        "import/export" in b.lower() or "permanent" in b.lower() for b in cudaq_export.blockers
    )

    invent = decide_compiler_path("cudaq", invent_green_full_runtime=True)
    assert invent.allowed is False
    assert any("invent-green" in b.lower() or "runtime" in b.lower() for b in invent.blockers)

    submit = decide_compiler_path("qir", invent_green_provider_submit=True)
    assert submit.allowed is False
    assert any("submit" in b.lower() or "provider" in b.lower() for b in submit.blockers)

    mlir = decide_compiler_path("mlir_enzyme_in_tree", request_import_export=True)
    assert mlir.allowed is True


def test_boundary_probe() -> None:
    """Materialise bounded ambient probes without promotion claims."""
    probe = materialise_demo_compiler_boundary_probe()
    assert probe.invent_green_cudaq_runtime is False
    assert probe.invent_green_qir_provider_submit is False
    assert probe.catalyst_runner_status == "runtime_gap"
    assert probe.catalyst_promotion_ready is False
    assert "LLVM" in probe.llvm_claim_gate_boundary or "JIT" in probe.llvm_claim_gate_boundary
    payload = probe.to_dict()
    assert payload["invent_green_cudaq_runtime"] is False

    success = materialise_compiler_boundary_probe(catalyst_runner_status="success")
    assert success.catalyst_runner_status == "success"


def test_public_surfaces_and_registry() -> None:
    """Publish deterministic surfaces and a validated product registry."""
    surfaces = map_compiler_boundary_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.compiler_boundary_product" in paths
    assert "scpn_quantum_control.compiler.mlir_llvm_jit_claim_gate" in paths

    registry = build_compiler_boundary_product_registry()
    assert registry["schema"] == COMPILER_BOUNDARY_PRODUCT_SCHEMA
    assert registry["schema"] == "compiler_boundary_product.v2"
    assert registry["invent_green_runtime_policy"] is False
    validated = assert_compiler_boundary_product_integrity(registry)
    assert validated["compiler_count"] == 5
    assert assert_compiler_boundary_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_policy() -> None:
    """Reject catalogue drift and dishonest runtime policy flags."""
    registry = build_compiler_boundary_product_registry()
    compilers = _registry_compilers(registry)

    broken = dict(registry)
    broken["compilers"] = compilers + [
        {
            "compiler_id": "ghost",
            "title": "t",
            "summary": "s",
            "status": "adapter",
            "ambient_pointer": "p",
            "route_matrix_pointer": "r",
            "import_export_allowed": True,
            "invent_green_runtime": False,
            "support_posture": "metadata_only",
            "as_of": "2026-07-24",
            "claim_boundary": COMPILER_BOUNDARY_CLAIM_BOUNDARY,
        }
    ]
    broken["compiler_count"] = len(cast(list[object], broken["compilers"]))
    with pytest.raises(ValueError, match="drift"):
        assert_compiler_boundary_product_integrity(broken)

    stale_schema = dict(registry)
    stale_schema["schema"] = "compiler_boundary_product.v1"
    with pytest.raises(ValueError, match="unexpected compiler boundary product schema"):
        assert_compiler_boundary_product_integrity(stale_schema)

    unexpected_key = dict(registry)
    unexpected_key["legacy_alias"] = "deprecated"
    with pytest.raises(ValueError, match="registry keys drift"):
        assert_compiler_boundary_product_integrity(unexpected_key)

    claim_drift = dict(registry)
    claim_drift["claim_boundary"] = "legacy planning label"
    with pytest.raises(ValueError, match="claim boundary drift"):
        assert_compiler_boundary_product_integrity(claim_drift)

    surface_drift = dict(registry)
    surface_drift["public_surfaces"] = []
    with pytest.raises(ValueError, match="public surface map drift"):
        assert_compiler_boundary_product_integrity(surface_drift)

    note_drift = dict(registry)
    note_drift["policy_note"] = "legacy planning label"
    with pytest.raises(ValueError, match="policy note drift"):
        assert_compiler_boundary_product_integrity(note_drift)

    empty = dict(registry)
    empty["compilers"] = []
    empty["compiler_count"] = 0
    with pytest.raises(ValueError, match="non-empty compilers"):
        assert_compiler_boundary_product_integrity(empty)

    policy = dict(registry)
    policy["invent_green_runtime_policy"] = True
    with pytest.raises(ValueError, match="invent_green_runtime_policy"):
        assert_compiler_boundary_product_integrity(policy)

    submit = dict(registry)
    submit["invent_green_qir_provider_submit_policy"] = True
    with pytest.raises(ValueError, match="invent_green_qir_provider_submit_policy"):
        assert_compiler_boundary_product_integrity(submit)


def test_integrity_rejects_blank_invalid() -> None:
    """Reject malformed rows, duplicates, blanks, and count drift."""
    registry = build_compiler_boundary_product_registry()
    compilers = _registry_compilers(registry)

    non_map = dict(registry)
    non_map["compilers"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_compiler_boundary_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in compilers]
    rows[0]["compiler_id"] = "  "
    blank_id["compilers"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_compiler_boundary_product_integrity(blank_id)

    bad_status = dict(registry)
    brows = [dict(row) for row in compilers]
    brows[0]["status"] = "marketing_tick"
    bad_status["compilers"] = brows
    with pytest.raises(ValueError, match="unknown status"):
        assert_compiler_boundary_product_integrity(bad_status)

    invent = dict(registry)
    irows = [dict(row) for row in compilers]
    irows[0]["invent_green_runtime"] = True
    invent["compilers"] = irows
    with pytest.raises(ValueError, match="invent_green_runtime"):
        assert_compiler_boundary_product_integrity(invent)

    no_ambient = dict(registry)
    arows = [dict(row) for row in compilers]
    arows[0]["ambient_pointer"] = ""
    no_ambient["compilers"] = arows
    with pytest.raises(ValueError, match="ambient_pointer"):
        assert_compiler_boundary_product_integrity(no_ambient)

    claim_drift = dict(registry)
    crows = [dict(row) for row in compilers]
    crows[0]["claim_boundary"] = "legacy planning label"
    claim_drift["compilers"] = crows
    with pytest.raises(ValueError, match="claim boundary drift"):
        assert_compiler_boundary_product_integrity(claim_drift)

    pointer_drift = dict(registry)
    pointer_rows = [dict(row) for row in compilers]
    pointer_rows[0]["ambient_pointer"] = "legacy planning pointer"
    pointer_drift["compilers"] = pointer_rows
    with pytest.raises(ValueError, match="catalogue row drift"):
        assert_compiler_boundary_product_integrity(pointer_drift)

    perm_export = dict(registry)
    prows = [dict(row) for row in compilers]
    for row in prows:
        if row.get("compiler_id") == "cudaq":
            row["import_export_allowed"] = True
    perm_export["compilers"] = prows
    with pytest.raises(ValueError, match="import_export_allowed"):
        assert_compiler_boundary_product_integrity(perm_export)

    no_cudaq = dict(registry)
    without = [dict(row) for row in compilers if row.get("compiler_id") != "cudaq"]
    no_cudaq["compilers"] = without
    no_cudaq["compiler_count"] = len(without)
    with pytest.raises(ValueError, match="missing cudaq|drift"):
        assert_compiler_boundary_product_integrity(no_cudaq)

    no_qir = dict(registry)
    without_q = [dict(row) for row in compilers if row.get("compiler_id") != "qir"]
    no_qir["compilers"] = without_q
    no_qir["compiler_count"] = len(without_q)
    with pytest.raises(ValueError, match="missing qir|drift"):
        assert_compiler_boundary_product_integrity(no_qir)

    dup = dict(registry)
    drows = [dict(row) for row in compilers]
    drows.append(dict(drows[0]))
    dup["compilers"] = drows
    dup["compiler_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate compiler_id"):
        assert_compiler_boundary_product_integrity(dup)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_compiler_boundary_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["compiler_count"] = 0
    with pytest.raises(ValueError, match="compiler_count"):
        assert_compiler_boundary_product_integrity(count_mismatch)


def test_module_exports() -> None:
    """Keep the supported compiler-boundary API explicitly exported."""
    assert "materialise_demo_compiler_boundary_probe" in compiler_boundary_product.__all__
    assert "decide_compiler_path" in compiler_boundary_product.__all__
    assert "list_compiler_ids" in compiler_boundary_product.__all__


def test_row_decision_probe_validation() -> None:
    """Enforce row, path-decision, and materialised-probe invariants."""
    base: dict[str, Any] = {
        "compiler_id": "x",
        "title": "t",
        "summary": "s",
        "status": "adapter",
        "ambient_pointer": "p",
        "route_matrix_pointer": "r",
        "import_export_allowed": True,
    }
    assert CompilerBoundaryRow(**base).compiler_id == "x"
    assert CompilerBoundaryRow(**base).to_dict()["compiler_id"] == "x"
    with pytest.raises(ValueError, match="compiler_id"):
        CompilerBoundaryRow(**{**base, "compiler_id": ""})
    with pytest.raises(ValueError, match="title"):
        CompilerBoundaryRow(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        CompilerBoundaryRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="status"):
        CompilerBoundaryRow(**{**base, "status": cast(Any, "nope")})
    with pytest.raises(ValueError, match="ambient_pointer"):
        CompilerBoundaryRow(**{**base, "ambient_pointer": ""})
    with pytest.raises(ValueError, match="route_matrix_pointer"):
        CompilerBoundaryRow(**{**base, "route_matrix_pointer": ""})
    with pytest.raises(ValueError, match="invent_green_runtime"):
        CompilerBoundaryRow(**{**base, "invent_green_runtime": True})
    with pytest.raises(ValueError, match="permanent_boundary"):
        CompilerBoundaryRow(
            **{
                **base,
                "status": "permanent_boundary",
                "import_export_allowed": True,
            }
        )
    with pytest.raises(ValueError, match="support_posture"):
        CompilerBoundaryRow(**{**base, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="as_of"):
        CompilerBoundaryRow(**{**base, "as_of": ""})

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
    assert decide_compiler_path("qir").to_dict()["allowed"] is True

    with pytest.raises(ValueError, match="catalyst_runner_status"):
        MaterialisedCompilerBoundaryProbe(
            catalyst_runner_status="",
            catalyst_promotion_ready=False,
            llvm_claim_gate_boundary="b",
            invent_green_cudaq_runtime=False,
            invent_green_qir_provider_submit=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="llvm_claim_gate_boundary"):
        MaterialisedCompilerBoundaryProbe(
            catalyst_runner_status="runtime_gap",
            catalyst_promotion_ready=False,
            llvm_claim_gate_boundary="",
            invent_green_cudaq_runtime=False,
            invent_green_qir_provider_submit=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_cudaq_runtime"):
        MaterialisedCompilerBoundaryProbe(
            catalyst_runner_status="runtime_gap",
            catalyst_promotion_ready=False,
            llvm_claim_gate_boundary="b",
            invent_green_cudaq_runtime=True,
            invent_green_qir_provider_submit=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_qir_provider_submit"):
        MaterialisedCompilerBoundaryProbe(
            catalyst_runner_status="runtime_gap",
            catalyst_promotion_ready=False,
            llvm_claim_gate_boundary="b",
            invent_green_cudaq_runtime=False,
            invent_green_qir_provider_submit=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedCompilerBoundaryProbe(
            catalyst_runner_status="runtime_gap",
            catalyst_promotion_ready=False,
            llvm_claim_gate_boundary="b",
            invent_green_cudaq_runtime=False,
            invent_green_qir_provider_submit=False,
            demo_label="",
        )


def test_catalogue_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed when the internal catalogue is empty or ambiguous."""
    monkeypatch.setattr(compiler_boundary_product, "_CANONICAL", ())
    with pytest.raises(RuntimeError, match="catalogue must be non-empty"):
        compiler_boundary_product._catalogue_map()

    blank = CompilerBoundaryRow(
        compiler_id="tmp",
        title="t",
        summary="s",
        status="adapter",
        ambient_pointer="p",
        route_matrix_pointer="r",
        import_export_allowed=True,
    )
    object.__setattr__(blank, "compiler_id", "  ")
    monkeypatch.setattr(compiler_boundary_product, "_CANONICAL", (blank,))
    with pytest.raises(RuntimeError, match="blank compiler_id"):
        compiler_boundary_product._catalogue_map()

    good = CompilerBoundaryRow(
        compiler_id="dup",
        title="t",
        summary="s",
        status="adapter",
        ambient_pointer="p",
        route_matrix_pointer="r",
        import_export_allowed=True,
    )
    monkeypatch.setattr(compiler_boundary_product, "_CANONICAL", (good, good))
    with pytest.raises(RuntimeError, match="duplicate compiler_id"):
        compiler_boundary_product._catalogue_map()
