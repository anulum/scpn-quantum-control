# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for KYMA mechanism benchmark product
"""Real-surface tests for ``kyma_mechanism_benchmark_product``."""

from __future__ import annotations

import builtins
from typing import Any, cast

import pytest

import scpn_quantum_control.kyma_mechanism_benchmark_product as kyma_product
from scpn_quantum_control.kyma_mechanism_benchmark_product import (
    KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY,
    KYMA_MECHANISM_BENCHMARK_PRODUCT_SCHEMA,
    KYMA_V2_PROTOCOL_ID,
    FrozenDesignConstants,
    KymaSuiteRow,
    MaterialisedMechanismCertificateProbe,
    PathEligibilityDecision,
    assert_kyma_mechanism_benchmark_product_integrity,
    build_kyma_mechanism_benchmark_product_registry,
    decide_kyma_path,
    get_frozen_design_constants,
    get_kyma_suite,
    iter_kyma_suites,
    list_kyma_suite_ids,
    load_frozen_design_constants,
    map_kyma_mechanism_benchmark_public_surfaces,
    materialise_demo_mechanism_certificate_probe,
    materialise_mechanism_certificate_probe,
)


def test_list_and_filters() -> None:
    """List stable suite identifiers and filter rows by KYMA generation."""
    ids = list_kyma_suite_ids()
    assert ids == ("kyma_v1", "kyma_v2")
    assert ids == list_kyma_suite_ids()
    v2 = iter_kyma_suites(kind="kyma_v2")
    assert len(v2) == 1
    assert v2[0].suite_id == "kyma_v2"
    empty = iter_kyma_suites(kind="kyma_v1")
    assert empty[0].suite_id == "kyma_v1"


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve known suites and reject blank or unknown identifiers."""
    row = get_kyma_suite("kyma_v2")
    assert row.claim_boundary == KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY
    assert row.mechanism_only is True
    assert row.invent_green_advantage is False
    assert row.protocol_id == KYMA_V2_PROTOCOL_ID
    v1 = get_kyma_suite("kyma_v1")
    assert "7f6b" in v1.protocol_id or "PREREGISTRATION" in v1.protocol_id
    with pytest.raises(ValueError, match="non-empty"):
        get_kyma_suite("  ")
    with pytest.raises(ValueError, match="unknown suite_id"):
        get_kyma_suite("not_a_suite")


def test_frozen_design_constants() -> None:
    """Keep frozen design constants digest-stable and ambient-compatible."""
    frozen = get_frozen_design_constants()
    reloaded = load_frozen_design_constants()
    assert frozen.content_digest == reloaded.content_digest
    assert len(frozen.content_digest) == 64
    assert frozen.realise_fraction == 0.95
    assert frozen.non_sep_target == 0.40
    assert frozen.g_sync_grid
    assert frozen.steps_grid
    assert frozen.k_bridge_grid
    payload = frozen.to_dict()
    assert payload["content_digest"] == frozen.content_digest
    # Ambient parity only when the optional JAX ambient path is importable.
    # Base CI matrix does not install JAX; framework overlay jobs do.
    try:
        load_frozen_design_constants(verify_ambient=True)
    except ImportError:
        pytest.importorskip("jax")
    except Exception:
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from scpn_quantum_control.kyma_mechanism_benchmark_product import "
                "load_frozen_design_constants; "
                "load_frozen_design_constants(verify_ambient=True); "
                "print('AMBIENT_OK')",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 and (
            "No module named 'jax'" in (proc.stderr or "")
            or 'No module named "jax"' in (proc.stderr or "")
        ):
            pytest.importorskip("jax")
        assert proc.returncode == 0, proc.stderr
        assert "AMBIENT_OK" in proc.stdout


def test_decide_kyma_path() -> None:
    """Allow honest mechanism paths and refuse prohibited claim routes."""
    ok = decide_kyma_path("kyma_v2")
    assert ok.allowed is True

    adv = decide_kyma_path("kyma_v2", invent_green_advantage=True)
    assert adv.allowed is False
    assert any("advantage" in b.lower() or "protocol" in b.lower() for b in adv.blockers)

    retune = decide_kyma_path("kyma_v2", post_hoc_constant_retune=True)
    assert retune.allowed is False
    assert any("retun" in b.lower() or "freeze" in b.lower() for b in retune.blockers)

    student = decide_kyma_path("kyma_v2", design_from_student_held_out=True)
    assert student.allowed is False
    assert any("held-out" in b.lower() or "student" in b.lower() for b in student.blockers)


def test_mechanism_certificate_probe() -> None:
    """Materialise deterministic bounded mechanism-certificate evidence."""
    with pytest.raises(ValueError, match="seed"):
        materialise_mechanism_certificate_probe(seed=-1)

    probe = materialise_demo_mechanism_certificate_probe()
    assert probe.suite_id == "kyma_v2"
    assert probe.protocol_id == KYMA_V2_PROTOCOL_ID
    assert probe.invent_green_advantage is False
    assert probe.design_from_student_held_out is False
    assert 0.0 <= probe.r1_realisability <= 1.0
    assert 0.0 <= probe.r2_realisability <= 1.0
    assert 0.0 <= probe.non_separability_rate <= 1.0
    assert probe.meets_realise_target is True
    assert probe.meets_non_sep_target is True
    assert len(probe.design_constants_digest) == 64
    payload = probe.to_dict()
    assert payload["invent_green_advantage"] is False
    # Ambient teacher path labels itself; product-local fallback is honest when
    # JAX is absent from the base matrix.
    assert (
        "ambient_kyma_v2" in probe.demo_label
        or "product_local_frozen_design_demo" in probe.demo_label
    )

    again = materialise_mechanism_certificate_probe(seed=0)
    assert again.r1_realisability == probe.r1_realisability


def test_materialise_certificate_stub_ambient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise product certificate assembly when ambient returns fixed rates."""
    import sys
    import types

    class _Cfg:
        pass

    class _Batch:
        pass

    fake_task = types.ModuleType("scpn_quantum_control.benchmarks.kyma_v2.task")
    fake_task.__dict__["ProbeConfigV2"] = lambda: _Cfg()
    fake_task.__dict__["build_trials"] = lambda cfg, seed: _Batch()

    fake_design = types.ModuleType("scpn_quantum_control.benchmarks.kyma_v2.design")
    fake_design.__dict__["single_relation_realisability"] = lambda cfg, batch: (1.0, 1.0)
    fake_design.__dict__["non_separability_rate"] = lambda cfg, batch: 0.5

    monkeypatch.setitem(
        sys.modules,
        "scpn_quantum_control.benchmarks.kyma_v2.task",
        fake_task,
    )
    monkeypatch.setitem(
        sys.modules,
        "scpn_quantum_control.benchmarks.kyma_v2.design",
        fake_design,
    )

    probe = materialise_mechanism_certificate_probe(seed=0)
    assert probe.r1_realisability == 1.0
    assert probe.r2_realisability == 1.0
    assert probe.non_separability_rate == 0.5
    assert probe.meets_realise_target is True
    assert probe.meets_non_sep_target is True
    assert probe.invent_green_advantage is False


def test_materialise_certificate_import_failures_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover base-matrix fallback without depending on local JAX presence."""
    real_import = builtins.__import__

    def raise_on_design(
        exc: Exception,
    ) -> Any:
        def guarded_import(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if name == "benchmarks.kyma_v2.design" and level == 1:
                raise exc
            return real_import(name, globals, locals, fromlist, level)

        return guarded_import

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", raise_on_design(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="import failed: RuntimeError: boom"):
            materialise_mechanism_certificate_probe()

    missing_unrelated = ModuleNotFoundError("No module named 'optax'")
    missing_unrelated.name = "optax"
    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", raise_on_design(missing_unrelated))
        with pytest.raises(RuntimeError, match="import failed.*optax"):
            materialise_mechanism_certificate_probe()

    missing_jax = ModuleNotFoundError("No module named 'jax'")
    missing_jax.name = "jax"
    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", raise_on_design(missing_jax))
        with pytest.raises(RuntimeError, match="unavailable for custom config"):
            materialise_mechanism_certificate_probe(config=object())

    missing_transitive_jax = ModuleNotFoundError("JAX transitive import unavailable")
    missing_transitive_jax.name = "optax"
    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", raise_on_design(missing_transitive_jax))
        probe = materialise_mechanism_certificate_probe(seed=3)
    assert probe.demo_label.endswith("seed_3")
    assert probe.meets_realise_target is True
    assert probe.meets_non_sep_target is True


def test_public_surfaces_and_registry() -> None:
    """Publish complete deterministic surface and registry catalogues."""
    surfaces = map_kyma_mechanism_benchmark_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.kyma_mechanism_benchmark_product" in paths
    assert "scpn_quantum_control.benchmarks.kyma_v2.design" in paths

    registry = build_kyma_mechanism_benchmark_product_registry()
    assert registry["schema"] == KYMA_MECHANISM_BENCHMARK_PRODUCT_SCHEMA
    assert registry["invent_green_advantage_policy"] is False
    assert registry["kyma_v2_protocol_id"] == KYMA_V2_PROTOCOL_ID
    validated = assert_kyma_mechanism_benchmark_product_integrity(registry)
    assert validated["suite_count"] == 2
    assert assert_kyma_mechanism_benchmark_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_policy() -> None:
    """Reject suite drift and invent-green, retune, or held-out policies."""
    registry = build_kyma_mechanism_benchmark_product_registry()
    suites = cast(list[dict[str, object]], registry["suites"])

    wrong_schema = dict(registry)
    wrong_schema["schema"] = "kyma_mechanism_benchmark_product.v1"
    with pytest.raises(ValueError, match="schema mismatch"):
        assert_kyma_mechanism_benchmark_product_integrity(wrong_schema)

    broken = dict(registry)
    broken["suites"] = suites + [
        {
            "suite_id": "ghost",
            "kind": "kyma_v2",
            "title": "t",
            "summary": "s",
            "ambient_pointer": "p",
            "protocol_id": "x",
            "mechanism_only": True,
            "invent_green_advantage": False,
            "support_posture": "local_research",
            "as_of": "2026-07-24",
            "claim_boundary": KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY,
        }
    ]
    broken["suite_count"] = len(cast(list[object], broken["suites"]))
    with pytest.raises(ValueError, match="drift"):
        assert_kyma_mechanism_benchmark_product_integrity(broken)

    empty: dict[str, object] = {
        "schema": KYMA_MECHANISM_BENCHMARK_PRODUCT_SCHEMA,
        "suites": [],
        "blank_entry_count": 0,
        "suite_count": 0,
        "frozen_design_constants": registry["frozen_design_constants"],
    }
    with pytest.raises(ValueError, match="non-empty suites"):
        assert_kyma_mechanism_benchmark_product_integrity(empty)

    policy = dict(registry)
    policy["invent_green_advantage_policy"] = True
    with pytest.raises(ValueError, match="invent_green_advantage_policy"):
        assert_kyma_mechanism_benchmark_product_integrity(policy)

    retune = dict(registry)
    retune["post_hoc_retune_policy"] = True
    with pytest.raises(ValueError, match="post_hoc_retune_policy"):
        assert_kyma_mechanism_benchmark_product_integrity(retune)

    student = dict(registry)
    student["design_from_student_held_out_policy"] = True
    with pytest.raises(ValueError, match="design_from_student_held_out_policy"):
        assert_kyma_mechanism_benchmark_product_integrity(student)


def test_integrity_rejects_blank_invalid() -> None:
    """Reject malformed, blank, duplicate, and count-drifted registry rows."""
    registry = build_kyma_mechanism_benchmark_product_registry()
    suites = cast(list[dict[str, object]], registry["suites"])

    non_map = dict(registry)
    non_map["suites"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_kyma_mechanism_benchmark_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in suites]
    rows[0]["suite_id"] = "  "
    blank_id["suites"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_kyma_mechanism_benchmark_product_integrity(blank_id)

    invent = dict(registry)
    irows = [dict(row) for row in suites]
    irows[0]["invent_green_advantage"] = True
    invent["suites"] = irows
    with pytest.raises(ValueError, match="invent_green_advantage"):
        assert_kyma_mechanism_benchmark_product_integrity(invent)

    mechanism = dict(registry)
    mrows = [dict(row) for row in suites]
    mrows[0]["mechanism_only"] = False
    mechanism["suites"] = mrows
    with pytest.raises(ValueError, match="mechanism_only"):
        assert_kyma_mechanism_benchmark_product_integrity(mechanism)

    no_protocol = dict(registry)
    prows = [dict(row) for row in suites]
    prows[0]["protocol_id"] = ""
    no_protocol["suites"] = prows
    with pytest.raises(ValueError, match="protocol_id"):
        assert_kyma_mechanism_benchmark_product_integrity(no_protocol)

    no_v2 = dict(registry)
    without = [dict(row) for row in suites if row.get("suite_id") != "kyma_v2"]
    no_v2["suites"] = without
    no_v2["suite_count"] = len(without)
    with pytest.raises(ValueError, match="missing kyma_v2|drift"):
        assert_kyma_mechanism_benchmark_product_integrity(no_v2)

    dup = dict(registry)
    drows = [dict(row) for row in suites]
    drows.append(dict(drows[0]))
    dup["suites"] = drows
    dup["suite_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate suite_id"):
        assert_kyma_mechanism_benchmark_product_integrity(dup)

    digest_bad = dict(registry)
    frozen = dict(cast(dict[str, object], registry["frozen_design_constants"]))
    frozen["content_digest"] = "0" * 64
    digest_bad["frozen_design_constants"] = frozen
    with pytest.raises(ValueError, match="digest drift"):
        assert_kyma_mechanism_benchmark_product_integrity(digest_bad)

    no_frozen = dict(registry)
    no_frozen["frozen_design_constants"] = "nope"
    with pytest.raises(ValueError, match="frozen_design_constants must be a mapping"):
        assert_kyma_mechanism_benchmark_product_integrity(no_frozen)

    missing_key = dict(registry)
    frozen2 = dict(cast(dict[str, object], registry["frozen_design_constants"]))
    del frozen2["g_sync_grid"]
    missing_key["frozen_design_constants"] = frozen2
    with pytest.raises(ValueError, match="missing"):
        assert_kyma_mechanism_benchmark_product_integrity(missing_key)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_kyma_mechanism_benchmark_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["suite_count"] = 0
    with pytest.raises(ValueError, match="suite_count"):
        assert_kyma_mechanism_benchmark_product_integrity(count_mismatch)

    protocol_bad = dict(registry)
    protocol_bad["kyma_v2_protocol_id"] = "wrong"
    with pytest.raises(ValueError, match="kyma_v2_protocol_id"):
        assert_kyma_mechanism_benchmark_product_integrity(protocol_bad)


def test_module_exports() -> None:
    """Keep the documented KYMA product functions publicly exported."""
    assert "materialise_demo_mechanism_certificate_probe" in kyma_product.__all__
    assert "decide_kyma_path" in kyma_product.__all__
    assert "get_frozen_design_constants" in kyma_product.__all__


def test_row_decision_probe_validation() -> None:
    """Validate every suite, decision, constant, and probe invariant."""
    base: dict[str, Any] = {
        "suite_id": "x",
        "kind": "kyma_v2",
        "title": "t",
        "summary": "s",
        "ambient_pointer": "p",
        "protocol_id": "proto",
    }
    assert KymaSuiteRow(**base).suite_id == "x"
    assert KymaSuiteRow(**base).to_dict()["suite_id"] == "x"
    with pytest.raises(ValueError, match="suite_id"):
        KymaSuiteRow(**{**base, "suite_id": ""})
    with pytest.raises(ValueError, match="kind"):
        KymaSuiteRow(**{**base, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="title"):
        KymaSuiteRow(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        KymaSuiteRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="ambient_pointer"):
        KymaSuiteRow(**{**base, "ambient_pointer": ""})
    with pytest.raises(ValueError, match="protocol_id"):
        KymaSuiteRow(**{**base, "protocol_id": ""})
    with pytest.raises(ValueError, match="mechanism_only"):
        KymaSuiteRow(**{**base, "mechanism_only": False})
    with pytest.raises(ValueError, match="invent_green_advantage"):
        KymaSuiteRow(**{**base, "invent_green_advantage": True})
    with pytest.raises(ValueError, match="support_posture"):
        KymaSuiteRow(**{**base, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="as_of"):
        KymaSuiteRow(**{**base, "as_of": ""})

    frozen_ok = FrozenDesignConstants(
        g_sync_grid=(1.0,),
        steps_grid=(60,),
        k_bridge_grid=(0.5,),
        realise_fraction=0.95,
        non_sep_target=0.4,
        balance_max_class_fraction=0.4,
        content_digest="a" * 64,
    )
    assert frozen_ok.content_digest == "a" * 64
    with pytest.raises(ValueError, match="g_sync_grid"):
        FrozenDesignConstants(
            g_sync_grid=(),
            steps_grid=(60,),
            k_bridge_grid=(0.5,),
            realise_fraction=0.95,
            non_sep_target=0.4,
            balance_max_class_fraction=0.4,
            content_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="steps_grid"):
        FrozenDesignConstants(
            g_sync_grid=(1.0,),
            steps_grid=(),
            k_bridge_grid=(0.5,),
            realise_fraction=0.95,
            non_sep_target=0.4,
            balance_max_class_fraction=0.4,
            content_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="k_bridge_grid"):
        FrozenDesignConstants(
            g_sync_grid=(1.0,),
            steps_grid=(60,),
            k_bridge_grid=(),
            realise_fraction=0.95,
            non_sep_target=0.4,
            balance_max_class_fraction=0.4,
            content_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="realise_fraction"):
        FrozenDesignConstants(
            g_sync_grid=(1.0,),
            steps_grid=(60,),
            k_bridge_grid=(0.5,),
            realise_fraction=0.0,
            non_sep_target=0.4,
            balance_max_class_fraction=0.4,
            content_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="non_sep_target"):
        FrozenDesignConstants(
            g_sync_grid=(1.0,),
            steps_grid=(60,),
            k_bridge_grid=(0.5,),
            realise_fraction=0.95,
            non_sep_target=1.5,
            balance_max_class_fraction=0.4,
            content_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="balance_max_class_fraction"):
        FrozenDesignConstants(
            g_sync_grid=(1.0,),
            steps_grid=(60,),
            k_bridge_grid=(0.5,),
            realise_fraction=0.95,
            non_sep_target=0.4,
            balance_max_class_fraction=0.0,
            content_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="content_digest"):
        FrozenDesignConstants(
            g_sync_grid=(1.0,),
            steps_grid=(60,),
            k_bridge_grid=(0.5,),
            realise_fraction=0.95,
            non_sep_target=0.4,
            balance_max_class_fraction=0.4,
            content_digest="",
        )
    with pytest.raises(ValueError, match="64-char"):
        FrozenDesignConstants(
            g_sync_grid=(1.0,),
            steps_grid=(60,),
            k_bridge_grid=(0.5,),
            realise_fraction=0.95,
            non_sep_target=0.4,
            balance_max_class_fraction=0.4,
            content_digest="abc",
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
    assert decide_kyma_path("kyma_v2").to_dict()["allowed"] is True

    with pytest.raises(ValueError, match="suite_id"):
        MaterialisedMechanismCertificateProbe(
            suite_id="",
            protocol_id="p",
            design_constants_digest="a" * 64,
            r1_realisability=1.0,
            r2_realisability=1.0,
            non_separability_rate=0.5,
            meets_realise_target=True,
            meets_non_sep_target=True,
            invent_green_advantage=False,
            design_from_student_held_out=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="protocol_id"):
        MaterialisedMechanismCertificateProbe(
            suite_id="kyma_v2",
            protocol_id="",
            design_constants_digest="a" * 64,
            r1_realisability=1.0,
            r2_realisability=1.0,
            non_separability_rate=0.5,
            meets_realise_target=True,
            meets_non_sep_target=True,
            invent_green_advantage=False,
            design_from_student_held_out=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="design_constants_digest"):
        MaterialisedMechanismCertificateProbe(
            suite_id="kyma_v2",
            protocol_id="p",
            design_constants_digest="",
            r1_realisability=1.0,
            r2_realisability=1.0,
            non_separability_rate=0.5,
            meets_realise_target=True,
            meets_non_sep_target=True,
            invent_green_advantage=False,
            design_from_student_held_out=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="r1_realisability"):
        MaterialisedMechanismCertificateProbe(
            suite_id="kyma_v2",
            protocol_id="p",
            design_constants_digest="a" * 64,
            r1_realisability=1.5,
            r2_realisability=1.0,
            non_separability_rate=0.5,
            meets_realise_target=True,
            meets_non_sep_target=True,
            invent_green_advantage=False,
            design_from_student_held_out=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_advantage"):
        MaterialisedMechanismCertificateProbe(
            suite_id="kyma_v2",
            protocol_id="p",
            design_constants_digest="a" * 64,
            r1_realisability=1.0,
            r2_realisability=1.0,
            non_separability_rate=0.5,
            meets_realise_target=True,
            meets_non_sep_target=True,
            invent_green_advantage=True,
            design_from_student_held_out=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="design_from_student_held_out"):
        MaterialisedMechanismCertificateProbe(
            suite_id="kyma_v2",
            protocol_id="p",
            design_constants_digest="a" * 64,
            r1_realisability=1.0,
            r2_realisability=1.0,
            non_separability_rate=0.5,
            meets_realise_target=True,
            meets_non_sep_target=True,
            invent_green_advantage=False,
            design_from_student_held_out=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedMechanismCertificateProbe(
            suite_id="kyma_v2",
            protocol_id="p",
            design_constants_digest="a" * 64,
            r1_realisability=1.0,
            r2_realisability=1.0,
            non_separability_rate=0.5,
            meets_realise_target=True,
            meets_non_sep_target=True,
            invent_green_advantage=False,
            design_from_student_held_out=False,
            demo_label="",
        )


def test_iter_suites_without_kind_filter() -> None:
    """Unfiltered suite iter returns both catalogue rows (kind is None branch)."""
    all_rows = iter_kyma_suites()
    assert len(all_rows) == len(list_kyma_suite_ids())
    assert {row.suite_id for row in all_rows} == set(list_kyma_suite_ids())


def test_mechanism_certificate_probe_to_dict() -> None:
    """MaterialisedMechanismCertificateProbe.to_dict exposes the public fields."""
    probe = MaterialisedMechanismCertificateProbe(
        suite_id="kyma_v2",
        protocol_id=KYMA_V2_PROTOCOL_ID,
        design_constants_digest="a" * 64,
        r1_realisability=1.0,
        r2_realisability=1.0,
        non_separability_rate=0.5,
        meets_realise_target=True,
        meets_non_sep_target=True,
        invent_green_advantage=False,
        design_from_student_held_out=False,
        demo_label="unit",
    )
    payload = probe.to_dict()
    assert payload["suite_id"] == "kyma_v2"
    assert payload["protocol_id"] == KYMA_V2_PROTOCOL_ID
    assert payload["invent_green_advantage"] is False
    assert payload["demo_label"] == "unit"
    assert payload["claim_boundary"] == KYMA_MECHANISM_BENCHMARK_CLAIM_BOUNDARY


def test_assert_mirrored_constants_match_ambient_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambient parity path succeeds when design mirrors product constants."""
    import sys
    import types

    fake_design = types.ModuleType("scpn_quantum_control.benchmarks.kyma_v2.design")
    fake_design.__dict__["G_SYNC_GRID"] = kyma_product._G_SYNC_GRID
    fake_design.__dict__["STEPS_GRID"] = kyma_product._STEPS_GRID
    fake_design.__dict__["K_BRIDGE_GRID"] = kyma_product._K_BRIDGE_GRID
    fake_design.__dict__["REALISE_FRACTION"] = kyma_product._REALISE_FRACTION
    fake_design.__dict__["NON_SEP_TARGET"] = kyma_product._NON_SEP_TARGET
    fake_design.__dict__["BALANCE_MAX_CLASS_FRACTION"] = kyma_product._BALANCE_MAX_CLASS_FRACTION
    monkeypatch.setitem(
        sys.modules,
        "scpn_quantum_control.benchmarks.kyma_v2.design",
        fake_design,
    )
    # Direct call exercises the match path under pytest-cov (no JAX import needed).
    kyma_product._assert_mirrored_constants_match_ambient()
    frozen = load_frozen_design_constants(verify_ambient=True)
    assert len(frozen.content_digest) == 64


def test_assert_mirrored_constants_detects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambient parity path fails closed when a mirrored constant drifts."""
    import sys
    import types

    fake_design = types.ModuleType("scpn_quantum_control.benchmarks.kyma_v2.design")
    fake_design.__dict__["G_SYNC_GRID"] = (9.9,)  # deliberate drift
    fake_design.__dict__["STEPS_GRID"] = kyma_product._STEPS_GRID
    fake_design.__dict__["K_BRIDGE_GRID"] = kyma_product._K_BRIDGE_GRID
    fake_design.__dict__["REALISE_FRACTION"] = kyma_product._REALISE_FRACTION
    fake_design.__dict__["NON_SEP_TARGET"] = kyma_product._NON_SEP_TARGET
    fake_design.__dict__["BALANCE_MAX_CLASS_FRACTION"] = kyma_product._BALANCE_MAX_CLASS_FRACTION
    monkeypatch.setitem(
        sys.modules,
        "scpn_quantum_control.benchmarks.kyma_v2.design",
        fake_design,
    )
    with pytest.raises(RuntimeError, match="design constant drift.*G_SYNC_GRID"):
        kyma_product._assert_mirrored_constants_match_ambient()


def test_catalogue_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject empty, blank, and duplicate internal suite catalogues."""
    monkeypatch.setattr(kyma_product, "_SUITES", ())
    with pytest.raises(RuntimeError, match="catalogue must be non-empty"):
        kyma_product._suite_map()

    blank = KymaSuiteRow(
        suite_id="tmp",
        kind="kyma_v2",
        title="t",
        summary="s",
        ambient_pointer="p",
        protocol_id="proto",
    )
    object.__setattr__(blank, "suite_id", "  ")
    monkeypatch.setattr(kyma_product, "_SUITES", (blank,))
    with pytest.raises(RuntimeError, match="blank suite_id"):
        kyma_product._suite_map()

    good = KymaSuiteRow(
        suite_id="dup",
        kind="kyma_v2",
        title="t",
        summary="s",
        ambient_pointer="p",
        protocol_id="proto",
    )
    monkeypatch.setattr(kyma_product, "_SUITES", (good, good))
    with pytest.raises(RuntimeError, match="duplicate suite_id"):
        kyma_product._suite_map()
