# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for cloud-native deployment product
"""Real-surface tests for ``cloud_native_deployment_product``."""

from __future__ import annotations

from typing import Any, cast

import pytest

import scpn_quantum_control.cloud_native_deployment_product as deploy_product
from scpn_quantum_control.cloud_native_deployment_product import (
    CLOUD_NATIVE_DEPLOYMENT_CLAIM_BOUNDARY,
    CLOUD_NATIVE_DEPLOYMENT_PRODUCT_SCHEMA,
    DeploymentPatternRow,
    MaterialisedDeployDryRunProbe,
    PathEligibilityDecision,
    ThreatModelRow,
    assert_cloud_native_deployment_product_integrity,
    build_cloud_native_deployment_product_registry,
    compute_spec_digest,
    decide_deploy_path,
    get_deployment_pattern,
    iter_deployment_patterns,
    list_deployment_pattern_ids,
    list_threat_ids,
    map_cloud_native_deployment_public_surfaces,
    materialise_demo_deploy_dry_run_probe,
    materialise_deploy_dry_run_probe,
)


def test_list_and_filters() -> None:
    """List deployment patterns and threats, then filter patterns by kind."""
    ids = list_deployment_pattern_ids()
    assert "batch_worker" in ids
    assert "stable_core_gate" in ids
    assert "offline_research" in ids
    assert len(ids) == 3
    threats = list_threat_ids()
    assert "secret_leakage" in threats
    assert "always_on_qpu" in threats
    assert len(threats) == 5
    batch = iter_deployment_patterns(kind="batch_worker")
    assert len(batch) == 1
    empty = iter_deployment_patterns(kind="stable_core_gate")
    assert empty[0].pattern_id == "stable_core_gate"


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve known patterns and reject blank or unknown identifiers."""
    row = get_deployment_pattern("batch_worker")
    assert row.claim_boundary == CLOUD_NATIVE_DEPLOYMENT_CLAIM_BOUNDARY
    assert row.allows_always_on_qpu is False
    assert row.secret_env_allowed is False
    assert row.live_cluster_create is False
    with pytest.raises(ValueError, match="non-empty"):
        get_deployment_pattern("  ")
    with pytest.raises(ValueError, match="unknown pattern_id"):
        get_deployment_pattern("not_a_pattern")


def test_decide_deploy_path() -> None:
    """Allow dry runs and refuse QPU, cluster, secret, or credential routes."""
    ok = decide_deploy_path("batch_worker")
    assert ok.allowed is True

    qpu = decide_deploy_path("batch_worker", invent_green_always_on_qpu=True)
    assert qpu.allowed is False
    assert any("qpu" in b.lower() for b in qpu.blockers)

    cluster = decide_deploy_path("batch_worker", invent_green_live_cluster=True)
    assert cluster.allowed is False
    assert any("cluster" in b.lower() for b in cluster.blockers)

    secret = decide_deploy_path("batch_worker", inject_secret_env=True)
    assert secret.allowed is False
    assert any("secret" in b.lower() for b in secret.blockers)

    creds = decide_deploy_path("batch_worker", load_credentials=True)
    assert creds.allowed is False
    assert any("credential" in b.lower() for b in creds.blockers)


def test_deploy_dry_run_probe() -> None:
    """Materialise bounded manifests and reject secret-like environment keys."""
    probe = materialise_demo_deploy_dry_run_probe()
    assert probe.pattern_id == "batch_worker"
    assert len(probe.manifest_sha256) == 64
    assert "deployment.yaml" in probe.file_names
    assert "service.yaml" in probe.file_names
    assert "docker-compose.yaml" in probe.file_names
    assert probe.invent_green_live_cluster is False
    assert probe.invent_green_always_on_qpu is False
    assert probe.secret_env_present is False
    assert "cluster" in probe.ambient_claim_boundary or "manifest" in (
        probe.ambient_claim_boundary
    )
    payload = probe.to_dict()
    assert payload["invent_green_live_cluster"] is False

    again = materialise_deploy_dry_run_probe("stable_core_gate")
    assert again.pattern_id == "stable_core_gate"
    assert len(again.manifest_sha256) == 64

    with pytest.raises(ValueError, match="secret-like"):
        materialise_deploy_dry_run_probe("batch_worker", env={"API_KEY": "x"})
    with pytest.raises(ValueError, match="secret-like"):
        materialise_deploy_dry_run_probe("batch_worker", env={"MY_SECRET": "x"})


def test_spec_digest() -> None:
    """Keep deployment specification digests deterministic and validated."""
    d1 = compute_spec_digest(
        name="scpn-batch-worker",
        image="img:1",
        command=("scpn-bench", "gate"),
    )
    d2 = compute_spec_digest(
        name="scpn-batch-worker",
        image="img:1",
        command=("scpn-bench", "gate"),
    )
    assert d1 == d2
    assert len(d1) == 64
    d3 = compute_spec_digest(
        name="other",
        image="img:1",
        command=("scpn-bench", "gate"),
    )
    assert d3 != d1
    with pytest.raises(ValueError, match="name"):
        compute_spec_digest(name="", image="img:1", command=("a",))
    with pytest.raises(ValueError, match="image"):
        compute_spec_digest(name="n", image="", command=("a",))
    with pytest.raises(ValueError, match="command"):
        compute_spec_digest(name="n", image="img:1", command=())
    with pytest.raises(ValueError, match="replicas"):
        compute_spec_digest(name="n", image="img:1", command=("a",), replicas=0)


def test_public_surfaces_and_registry() -> None:
    """Publish complete deterministic surface and registry catalogues."""
    surfaces = map_cloud_native_deployment_public_surfaces()
    assert surfaces
    paths = {row["module_path"] for row in surfaces}
    assert "scpn_quantum_control.cloud_native_deployment_product" in paths
    assert "scpn_quantum_control.deployment.cloud_native" in paths

    registry = build_cloud_native_deployment_product_registry()
    assert registry["schema"] == CLOUD_NATIVE_DEPLOYMENT_PRODUCT_SCHEMA
    assert registry["allows_always_on_qpu_policy"] is False
    assert registry["secret_env_allowed_policy"] is False
    assert registry["live_cluster_create_policy"] is False
    validated = assert_cloud_native_deployment_product_integrity(registry)
    assert validated["pattern_count"] == 3
    assert validated["threat_count"] == 5
    assert assert_cloud_native_deployment_product_integrity()["blank_entry_count"] == 0


def test_integrity_rejects_drift_and_policy() -> None:
    """Reject pattern drift and permissive QPU, secret, or cluster policies."""
    registry = build_cloud_native_deployment_product_registry()
    patterns = cast(list[dict[str, object]], registry["patterns"])

    wrong_schema = dict(registry)
    wrong_schema["schema"] = "cloud_native_deployment_product.v1"
    with pytest.raises(ValueError, match="schema mismatch"):
        assert_cloud_native_deployment_product_integrity(wrong_schema)

    broken = dict(registry)
    broken["patterns"] = patterns + [
        {
            "pattern_id": "ghost",
            "kind": "batch_worker",
            "title": "t",
            "summary": "s",
            "default_command": ["cmd"],
            "allows_always_on_qpu": False,
            "secret_env_allowed": False,
            "live_cluster_create": False,
            "hardware_safety_pointer": "p",
            "compute_plan_pointer": "p",
            "support_posture": "policy_only",
            "as_of": "2026-07-24",
            "claim_boundary": CLOUD_NATIVE_DEPLOYMENT_CLAIM_BOUNDARY,
        }
    ]
    broken["pattern_count"] = len(cast(list[object], broken["patterns"]))
    with pytest.raises(ValueError, match="drift"):
        assert_cloud_native_deployment_product_integrity(broken)

    empty: dict[str, object] = {
        "schema": CLOUD_NATIVE_DEPLOYMENT_PRODUCT_SCHEMA,
        "patterns": [],
        "threats": registry["threats"],
        "blank_entry_count": 0,
        "pattern_count": 0,
    }
    with pytest.raises(ValueError, match="non-empty patterns"):
        assert_cloud_native_deployment_product_integrity(empty)

    no_threats = dict(registry)
    no_threats["threats"] = []
    no_threats["threat_count"] = 0
    with pytest.raises(ValueError, match="non-empty threats"):
        assert_cloud_native_deployment_product_integrity(no_threats)

    policy = dict(registry)
    policy["allows_always_on_qpu_policy"] = True
    with pytest.raises(ValueError, match="allows_always_on_qpu_policy"):
        assert_cloud_native_deployment_product_integrity(policy)

    secret_policy = dict(registry)
    secret_policy["secret_env_allowed_policy"] = True
    with pytest.raises(ValueError, match="secret_env_allowed_policy"):
        assert_cloud_native_deployment_product_integrity(secret_policy)

    live_policy = dict(registry)
    live_policy["live_cluster_create_policy"] = True
    with pytest.raises(ValueError, match="live_cluster_create_policy"):
        assert_cloud_native_deployment_product_integrity(live_policy)


def test_integrity_rejects_blank_invalid() -> None:
    """Reject malformed, blank, duplicate, and count-drifted registry rows."""
    registry = build_cloud_native_deployment_product_registry()
    patterns = cast(list[dict[str, object]], registry["patterns"])
    threats = cast(list[dict[str, object]], registry["threats"])

    non_map = dict(registry)
    non_map["patterns"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_cloud_native_deployment_product_integrity(non_map)

    blank_id = dict(registry)
    rows = [dict(row) for row in patterns]
    rows[0]["pattern_id"] = "  "
    blank_id["patterns"] = rows
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_cloud_native_deployment_product_integrity(blank_id)

    qpu = dict(registry)
    qrows = [dict(row) for row in patterns]
    qrows[0]["allows_always_on_qpu"] = True
    qpu["patterns"] = qrows
    with pytest.raises(ValueError, match="allows_always_on_qpu"):
        assert_cloud_native_deployment_product_integrity(qpu)

    secret = dict(registry)
    srows = [dict(row) for row in patterns]
    srows[0]["secret_env_allowed"] = True
    secret["patterns"] = srows
    with pytest.raises(ValueError, match="secret_env_allowed"):
        assert_cloud_native_deployment_product_integrity(secret)

    live = dict(registry)
    lrows = [dict(row) for row in patterns]
    lrows[0]["live_cluster_create"] = True
    live["patterns"] = lrows
    with pytest.raises(ValueError, match="live_cluster_create"):
        assert_cloud_native_deployment_product_integrity(live)

    no_cmd = dict(registry)
    crows = [dict(row) for row in patterns]
    crows[0]["default_command"] = []
    no_cmd["patterns"] = crows
    with pytest.raises(ValueError, match="default_command"):
        assert_cloud_native_deployment_product_integrity(no_cmd)

    no_batch = dict(registry)
    without = [dict(row) for row in patterns if row.get("pattern_id") != "batch_worker"]
    no_batch["patterns"] = without
    no_batch["pattern_count"] = len(without)
    with pytest.raises(ValueError, match="missing batch_worker|drift"):
        assert_cloud_native_deployment_product_integrity(no_batch)

    dup = dict(registry)
    drows = [dict(row) for row in patterns]
    drows.append(dict(drows[0]))
    dup["patterns"] = drows
    dup["pattern_count"] = len(drows)
    with pytest.raises(ValueError, match="duplicate pattern_id"):
        assert_cloud_native_deployment_product_integrity(dup)

    threat_non_map = dict(registry)
    threat_non_map["threats"] = [cast(Any, "nope")]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_cloud_native_deployment_product_integrity(threat_non_map)

    threat_blank = dict(registry)
    trows = [dict(row) for row in threats]
    trows[0]["threat_id"] = ""
    threat_blank["threats"] = trows
    with pytest.raises(ValueError, match="blank or invalid threat_id"):
        assert_cloud_native_deployment_product_integrity(threat_blank)

    threat_fail = dict(registry)
    tf = [dict(row) for row in threats]
    tf[0]["fail_closed"] = False
    threat_fail["threats"] = tf
    with pytest.raises(ValueError, match="fail_closed"):
        assert_cloud_native_deployment_product_integrity(threat_fail)

    threat_dup = dict(registry)
    td = [dict(row) for row in threats]
    td.append(dict(td[0]))
    threat_dup["threats"] = td
    threat_dup["threat_count"] = len(td)
    with pytest.raises(ValueError, match="duplicate threat_id"):
        assert_cloud_native_deployment_product_integrity(threat_dup)

    threat_drift = dict(registry)
    pruned = [dict(row) for row in threats if row.get("threat_id") != "secret_leakage"]
    threat_drift["threats"] = pruned
    threat_drift["threat_count"] = len(pruned)
    with pytest.raises(ValueError, match="threat set drift"):
        assert_cloud_native_deployment_product_integrity(threat_drift)

    blank_count = dict(registry)
    blank_count["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_cloud_native_deployment_product_integrity(blank_count)

    count_mismatch = dict(registry)
    count_mismatch["pattern_count"] = 0
    with pytest.raises(ValueError, match="pattern_count"):
        assert_cloud_native_deployment_product_integrity(count_mismatch)

    threat_count_bad = dict(registry)
    threat_count_bad["threat_count"] = 0
    with pytest.raises(ValueError, match="threat_count"):
        assert_cloud_native_deployment_product_integrity(threat_count_bad)


def test_module_exports() -> None:
    """Keep the documented deployment-product functions publicly exported."""
    assert "materialise_demo_deploy_dry_run_probe" in deploy_product.__all__
    assert "decide_deploy_path" in deploy_product.__all__
    assert "list_deployment_pattern_ids" in deploy_product.__all__


def test_row_decision_probe_validation() -> None:
    """Validate every pattern, threat, decision, and dry-run probe invariant."""
    base: dict[str, Any] = {
        "pattern_id": "x",
        "kind": "batch_worker",
        "title": "t",
        "summary": "s",
        "default_command": ("scpn-bench", "gate"),
    }
    assert DeploymentPatternRow(**base).pattern_id == "x"
    assert DeploymentPatternRow(**base).to_dict()["pattern_id"] == "x"
    with pytest.raises(ValueError, match="pattern_id"):
        DeploymentPatternRow(**{**base, "pattern_id": ""})
    with pytest.raises(ValueError, match="kind"):
        DeploymentPatternRow(**{**base, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="title"):
        DeploymentPatternRow(**{**base, "title": ""})
    with pytest.raises(ValueError, match="summary"):
        DeploymentPatternRow(**{**base, "summary": ""})
    with pytest.raises(ValueError, match="default_command"):
        DeploymentPatternRow(**{**base, "default_command": ()})
    with pytest.raises(ValueError, match="allows_always_on_qpu"):
        DeploymentPatternRow(**{**base, "allows_always_on_qpu": True})
    with pytest.raises(ValueError, match="secret_env_allowed"):
        DeploymentPatternRow(**{**base, "secret_env_allowed": True})
    with pytest.raises(ValueError, match="live_cluster_create"):
        DeploymentPatternRow(**{**base, "live_cluster_create": True})
    with pytest.raises(ValueError, match="hardware_safety_pointer"):
        DeploymentPatternRow(**{**base, "hardware_safety_pointer": ""})
    with pytest.raises(ValueError, match="compute_plan_pointer"):
        DeploymentPatternRow(**{**base, "compute_plan_pointer": ""})
    with pytest.raises(ValueError, match="support_posture"):
        DeploymentPatternRow(**{**base, "support_posture": cast(Any, "nope")})
    with pytest.raises(ValueError, match="as_of"):
        DeploymentPatternRow(**{**base, "as_of": ""})

    threat_base: dict[str, Any] = {
        "threat_id": "t",
        "kind": "secret_leakage",
        "title": "title",
        "mitigation": "mitigation",
    }
    assert ThreatModelRow(**threat_base).threat_id == "t"
    with pytest.raises(ValueError, match="threat_id"):
        ThreatModelRow(**{**threat_base, "threat_id": ""})
    with pytest.raises(ValueError, match="kind"):
        ThreatModelRow(**{**threat_base, "kind": cast(Any, "nope")})
    with pytest.raises(ValueError, match="title"):
        ThreatModelRow(**{**threat_base, "title": ""})
    with pytest.raises(ValueError, match="mitigation"):
        ThreatModelRow(**{**threat_base, "mitigation": ""})
    with pytest.raises(ValueError, match="fail_closed"):
        ThreatModelRow(**{**threat_base, "fail_closed": False})

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
    assert decide_deploy_path("batch_worker").to_dict()["allowed"] is True

    with pytest.raises(ValueError, match="pattern_id"):
        MaterialisedDeployDryRunProbe(
            pattern_id="",
            manifest_sha256="a" * 64,
            file_names=("deployment.yaml",),
            ambient_claim_boundary="b",
            invent_green_live_cluster=False,
            invent_green_always_on_qpu=False,
            secret_env_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="manifest_sha256"):
        MaterialisedDeployDryRunProbe(
            pattern_id="p",
            manifest_sha256="",
            file_names=("deployment.yaml",),
            ambient_claim_boundary="b",
            invent_green_live_cluster=False,
            invent_green_always_on_qpu=False,
            secret_env_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="64-char"):
        MaterialisedDeployDryRunProbe(
            pattern_id="p",
            manifest_sha256="abc",
            file_names=("deployment.yaml",),
            ambient_claim_boundary="b",
            invent_green_live_cluster=False,
            invent_green_always_on_qpu=False,
            secret_env_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="file_names"):
        MaterialisedDeployDryRunProbe(
            pattern_id="p",
            manifest_sha256="a" * 64,
            file_names=(),
            ambient_claim_boundary="b",
            invent_green_live_cluster=False,
            invent_green_always_on_qpu=False,
            secret_env_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="file_names entries"):
        MaterialisedDeployDryRunProbe(
            pattern_id="p",
            manifest_sha256="a" * 64,
            file_names=("ok", "  "),
            ambient_claim_boundary="b",
            invent_green_live_cluster=False,
            invent_green_always_on_qpu=False,
            secret_env_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="ambient_claim_boundary"):
        MaterialisedDeployDryRunProbe(
            pattern_id="p",
            manifest_sha256="a" * 64,
            file_names=("deployment.yaml",),
            ambient_claim_boundary="",
            invent_green_live_cluster=False,
            invent_green_always_on_qpu=False,
            secret_env_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_live_cluster"):
        MaterialisedDeployDryRunProbe(
            pattern_id="p",
            manifest_sha256="a" * 64,
            file_names=("deployment.yaml",),
            ambient_claim_boundary="b",
            invent_green_live_cluster=True,
            invent_green_always_on_qpu=False,
            secret_env_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="invent_green_always_on_qpu"):
        MaterialisedDeployDryRunProbe(
            pattern_id="p",
            manifest_sha256="a" * 64,
            file_names=("deployment.yaml",),
            ambient_claim_boundary="b",
            invent_green_live_cluster=False,
            invent_green_always_on_qpu=True,
            secret_env_present=False,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="secret_env_present"):
        MaterialisedDeployDryRunProbe(
            pattern_id="p",
            manifest_sha256="a" * 64,
            file_names=("deployment.yaml",),
            ambient_claim_boundary="b",
            invent_green_live_cluster=False,
            invent_green_always_on_qpu=False,
            secret_env_present=True,
            demo_label="d",
        )
    with pytest.raises(ValueError, match="demo_label"):
        MaterialisedDeployDryRunProbe(
            pattern_id="p",
            manifest_sha256="a" * 64,
            file_names=("deployment.yaml",),
            ambient_claim_boundary="b",
            invent_green_live_cluster=False,
            invent_green_always_on_qpu=False,
            secret_env_present=False,
            demo_label="",
        )


def test_catalogue_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject empty, blank, and duplicate internal pattern catalogues."""
    monkeypatch.setattr(deploy_product, "_PATTERNS", ())
    with pytest.raises(RuntimeError, match="catalogue must be non-empty"):
        deploy_product._pattern_map()

    blank = DeploymentPatternRow(
        pattern_id="tmp",
        kind="batch_worker",
        title="t",
        summary="s",
        default_command=("cmd",),
    )
    object.__setattr__(blank, "pattern_id", "  ")
    monkeypatch.setattr(deploy_product, "_PATTERNS", (blank,))
    with pytest.raises(RuntimeError, match="blank pattern_id"):
        deploy_product._pattern_map()

    good = DeploymentPatternRow(
        pattern_id="dup",
        kind="batch_worker",
        title="t",
        summary="s",
        default_command=("cmd",),
    )
    monkeypatch.setattr(deploy_product, "_PATTERNS", (good, good))
    with pytest.raises(RuntimeError, match="duplicate pattern_id"):
        deploy_product._pattern_map()


def test_iter_deployment_patterns_without_kind_filter() -> None:
    """Unfiltered pattern iter returns the full catalogue (kind is None branch)."""
    all_rows = iter_deployment_patterns()
    assert len(all_rows) == len(list_deployment_pattern_ids())
    assert {row.pattern_id for row in all_rows} == set(list_deployment_pattern_ids())


def test_deploy_dry_run_accepts_non_secret_env() -> None:
    """Safe env keys pass the secret pre-check (loop continue branch)."""
    probe = materialise_deploy_dry_run_probe(
        "batch_worker",
        env={"LOG_LEVEL": "info", "WORKER_NAME": "demo"},
    )
    assert probe.pattern_id == "batch_worker"
    assert probe.invent_green_live_cluster is False
